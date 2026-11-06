# collector\market_policy.py
"""Reviewed BTC/ETH derivative market policies for aggr.trade collection.

The original chart URLs mix spot and derivative instruments. This collector
must never infer the production CVD universe only from suffixes or from every
market returned by /products.

Version 1 uses explicit allowlists proven to be accepted by the supplied
historical GET requests and classified from the supplied aggr exchange/catalog
implementations.

Policy rules
------------

* Futures/perpetual derivatives only.
* Spot markets are excluded.
* Options are excluded and are not requested.
* The exact Binance USDT perpetual is the only execution-price source.
* All selected vbuy/vsell values are USD-like notionals because aggr-server
  stores price multiplied by exchange-normalized base size.
* Market tiers are diagnostic labels only. Total CVD sums every selected row.
* A policy change requires a new policy version and therefore a new manifest
  market-list identity.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Iterable

MARKET_POLICY_SCHEMA_VERSION = "smp_aggr_market_policy.v1"
COLLECTOR_SCHEMA_VERSION = "smp_aggr_collector.v1"
SOURCE_INTERVAL_MS = 5000

_ALLOWED_TIERS = {
    "USDT",
    "USDC",
    "USD_INVERSE",
}


@dataclass(frozen=True, order=True)
class MarketSpec:
    """One reviewed derivative instrument in a collection policy."""

    market: str
    tier: str

    def __post_init__(self) -> None:
        market = str(self.market or "").strip()
        tier = str(self.tier or "").strip().upper()

        if not market or ":" not in market:
            raise ValueError(f"Invalid aggr market identifier: {self.market!r}")

        if tier not in _ALLOWED_TIERS:
            raise ValueError(f"Unsupported market tier {tier!r} for {market!r}")

        object.__setattr__(self, "market", market)
        object.__setattr__(self, "tier", tier)


@dataclass(frozen=True)
class MarketPolicy:
    """Complete immutable market policy for one base asset."""

    base: str
    version: str
    price_market: str
    markets: tuple[MarketSpec, ...]

    def __post_init__(self) -> None:
        base = str(self.base or "").strip().upper()
        version = str(self.version or "").strip()
        price_market = str(self.price_market or "").strip()

        if base not in {"BTC", "ETH"}:
            raise ValueError("Version 1 market policies support BTC and ETH only")

        if not version:
            raise ValueError("Market policy version must be non-empty")

        if not self.markets:
            raise ValueError("Market policy must contain at least one market")

        market_names = [item.market for item in self.markets]

        if len(market_names) != len(set(market_names)):
            duplicates = sorted(
                {name for name in market_names if market_names.count(name) > 1}
            )
            raise ValueError(f"Market policy contains duplicate names: {duplicates}")

        if price_market not in set(market_names):
            raise ValueError(
                "price_market must be included in the reviewed market list"
            )

        object.__setattr__(self, "base", base)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "price_market", price_market)

    @property
    def selected_markets(self) -> tuple[str, ...]:
        """Return deterministic lexicographically sorted market names."""

        return tuple(sorted(item.market for item in self.markets))

    @property
    def request_markets(self) -> tuple[str, ...]:
        """Return deterministic markets used in the historical URL."""

        return self.selected_markets

    @property
    def tier_by_market(self) -> dict[str, str]:
        return {item.market: item.tier for item in self.markets}

    @property
    def market_list_hash(self) -> str:
        """Hash every policy field which affects archive interpretation."""

        payload = {
            "schema": MARKET_POLICY_SCHEMA_VERSION,
            "base": self.base,
            "version": self.version,
            "price_market": self.price_market,
            "markets": [
                {
                    "market": market,
                    "tier": self.tier_by_market[market],
                }
                for market in self.selected_markets
            ],
        }

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def assert_response_markets_allowed(
        self,
        response_markets: Iterable[str],
    ) -> None:
        observed = {
            str(value or "").strip()
            for value in response_markets
            if str(value or "").strip()
        }

        unknown = sorted(observed - set(self.selected_markets))

        if unknown:
            raise ValueError(
                "aggr response contains market(s) outside the reviewed "
                f"{self.base} policy: {unknown[:20]}"
            )


@dataclass(frozen=True, order=True)
class MarketPolicyEra:
    """One policy beginning at an exact UTC-hour boundary."""

    activation_start_ms: int
    policy: MarketPolicy

    def __post_init__(self) -> None:
        activation = self.activation_start_ms

        if isinstance(activation, bool) or not isinstance(activation, int):
            raise ValueError("Policy-era activation_start_ms must be an integer")

        if activation < 0 or activation % 3_600_000 != 0:
            raise ValueError(
                "Policy-era activation_start_ms must be a non-negative "
                "UTC-hour boundary"
            )

        if not isinstance(self.policy, MarketPolicy):
            raise TypeError("Policy era requires a MarketPolicy")


@dataclass(frozen=True, slots=True)
class MarketPolicySegment:
    """One half-open request segment owned by one policy era."""

    start_ms: int
    end_ms: int
    activation_start_ms: int
    policy: MarketPolicy


@dataclass(frozen=True)
class MarketPolicyRegistry:
    """Ordered policy history for one canonical base."""

    base: str
    eras: tuple[MarketPolicyEra, ...]

    def __post_init__(self) -> None:
        base = str(self.base or "").strip().upper()

        if base not in {"BTC", "ETH"}:
            raise ValueError("Market policy registries support BTC and ETH only")

        eras = tuple(sorted(self.eras))

        if not eras:
            raise ValueError("Market policy registry must contain at least one era")

        if eras[0].activation_start_ms != 0:
            raise ValueError(
                "Market policy registry must begin with a zero-epoch "
                "compatibility era"
            )

        activation_values = [era.activation_start_ms for era in eras]

        if len(activation_values) != len(set(activation_values)):
            raise ValueError(
                "Market policy registry contains duplicate activation times"
            )

        versions: set[str] = set()

        for era in eras:
            if era.policy.base != base:
                raise ValueError(
                    "Every policy era must belong to registry base " f"{base!r}"
                )

            if era.policy.version in versions:
                raise ValueError(
                    "Market policy versions must be unique inside one "
                    f"registry: {era.policy.version!r}"
                )

            versions.add(era.policy.version)

        object.__setattr__(self, "base", base)
        object.__setattr__(self, "eras", eras)

    @property
    def latest_policy(self) -> MarketPolicy:
        return self.eras[-1].policy

    def era_for_timestamp_ms(
        self,
        timestamp_ms: int,
    ) -> MarketPolicyEra:
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise ValueError(
                "Policy lookup timestamp_ms must be a non-negative integer"
            )

        starts = [era.activation_start_ms for era in self.eras]
        index = bisect_right(starts, timestamp_ms) - 1

        if index < 0:
            raise RuntimeError("No market policy era owns the requested timestamp")

        return self.eras[index]

    def segments_for_range(
        self,
        from_ms: int,
        to_ms: int,
    ) -> tuple[MarketPolicySegment, ...]:
        if (
            isinstance(from_ms, bool)
            or not isinstance(from_ms, int)
            or isinstance(to_ms, bool)
            or not isinstance(to_ms, int)
        ):
            raise ValueError("Policy-segment boundaries must be integers")

        if (
            from_ms <= 0
            or to_ms <= from_ms
            or from_ms % SOURCE_INTERVAL_MS != 0
            or to_ms % SOURCE_INTERVAL_MS != 0
        ):
            raise ValueError(
                "Policy-segment range must be a positive non-empty "
                "five-second-aligned [from, to) range"
            )

        starts = [era.activation_start_ms for era in self.eras]

        output: list[MarketPolicySegment] = []
        cursor = from_ms

        while cursor < to_ms:
            era_index = bisect_right(starts, cursor) - 1
            era = self.eras[era_index]

            next_activation = (
                starts[era_index + 1] if era_index + 1 < len(starts) else to_ms
            )
            segment_end = min(to_ms, next_activation)

            if segment_end <= cursor:
                raise RuntimeError("Market policy registry produced an empty segment")

            output.append(
                MarketPolicySegment(
                    start_ms=cursor,
                    end_ms=segment_end,
                    activation_start_ms=era.activation_start_ms,
                    policy=era.policy,
                )
            )
            cursor = segment_end

        return tuple(output)


_BTC_MARKETS = (
    # Linear USDT derivatives.
    MarketSpec("BINANCE_FUTURES:btcusdt", "USDT"),
    MarketSpec("BITFINEX:BTCF0:USTF0", "USDT"),
    MarketSpec("BITGET:BTCUSDT", "USDT"),
    MarketSpec("BITMEX:XBT_USDT", "USDT"),
    MarketSpec("BITMEX:XBTUSDT", "USDT"),
    MarketSpec("BITUNIX:BTCUSDT", "USDT"),
    MarketSpec("BYBIT:BTCUSDT", "USDT"),
    MarketSpec("GATEIO:BTC_USDT", "USDT"),
    MarketSpec("HUOBI:BTC-USDT", "USDT"),
    MarketSpec("KUCOIN:XBTUSDTM", "USDT"),
    MarketSpec("MEXC:BTC_USDT", "USDT"),
    MarketSpec("OKEX:BTC-USDT-SWAP", "USDT"),
    # Linear USDC derivatives.
    MarketSpec("DERIBIT:BTC_USDC-PERPETUAL", "USDC"),
    MarketSpec("OKEX:BTC-USDC-SWAP", "USDC"),
    # USD/inverse/coin-margined derivatives, normalized by aggr adapters.
    MarketSpec("BINANCE_FUTURES:btcusd_perp", "USD_INVERSE"),
    MarketSpec("BITMEX:XBTUSD", "USD_INVERSE"),
    MarketSpec("BYBIT:BTCUSD", "USD_INVERSE"),
    MarketSpec("COINBASE:BTC-PERP-INTX", "USD_INVERSE"),
    MarketSpec("DERIBIT:BTC-PERPETUAL", "USD_INVERSE"),
    MarketSpec("HUOBI:BTC-USD", "USD_INVERSE"),
    MarketSpec("KRAKEN:PI_XBTUSD", "USD_INVERSE"),
    MarketSpec("KUCOIN:XBTUSDM", "USD_INVERSE"),
    MarketSpec("MEXC:BTC_USD", "USD_INVERSE"),
    MarketSpec("OKEX:BTC-USD-SWAP", "USD_INVERSE"),
)


_ETH_MARKETS = (
    # Linear USDT derivatives.
    MarketSpec("BINANCE_FUTURES:ethusdt", "USDT"),
    MarketSpec("BITFINEX:ETHF0:USTF0", "USDT"),
    MarketSpec("BITGET:ETHUSDT", "USDT"),
    MarketSpec("BITMEX:ETH_USDT", "USDT"),
    MarketSpec("BITMEX:ETHUSDT", "USDT"),
    MarketSpec("BITUNIX:ETHUSDT", "USDT"),
    MarketSpec("BYBIT:ETHUSDT", "USDT"),
    MarketSpec("HUOBI:ETH-USDT", "USDT"),
    MarketSpec("KUCOIN:ETHUSDTM", "USDT"),
    MarketSpec("OKEX:ETH-USDT-SWAP", "USDT"),
    # Linear USDC derivatives.
    MarketSpec("BINANCE_FUTURES:ethusdc", "USDC"),
    MarketSpec("DERIBIT:ETH_USDC-PERPETUAL", "USDC"),
    MarketSpec("KUCOIN:ETHUSDCM", "USDC"),
    # USD/inverse/coin-margined derivatives, normalized by aggr adapters.
    MarketSpec("BINANCE_FUTURES:ethusd_perp", "USD_INVERSE"),
    MarketSpec("BITGET:ETHPERP", "USD_INVERSE"),
    MarketSpec("BITGET:ETHUSD", "USD_INVERSE"),
    MarketSpec("BITMEX:ETHUSD", "USD_INVERSE"),
    MarketSpec("BYBIT:ETHUSD", "USD_INVERSE"),
    MarketSpec("COINBASE:ETH-PERP-INTX", "USD_INVERSE"),
    MarketSpec("DERIBIT:ETH-PERPETUAL", "USD_INVERSE"),
    MarketSpec("DYDX:ETH-USD", "USD_INVERSE"),
    MarketSpec("HUOBI:ETH-USD", "USD_INVERSE"),
    MarketSpec("HYPERLIQUID:ETH", "USD_INVERSE"),
    MarketSpec("KRAKEN:PF_ETHUSD", "USD_INVERSE"),
    MarketSpec("KRAKEN:PI_ETHUSD", "USD_INVERSE"),
    MarketSpec("KUCOIN:ETHUSDM", "USD_INVERSE"),
    MarketSpec("OKEX:ETH-USD-SWAP", "USD_INVERSE"),
)


BTC_POLICY = MarketPolicy(
    base="BTC",
    version="btc-derivatives-v1",
    price_market="BINANCE_FUTURES:btcusdt",
    markets=_BTC_MARKETS,
)

ETH_POLICY = MarketPolicy(
    base="ETH",
    version="eth-derivatives-v1",
    price_market="BINANCE_FUTURES:ethusdt",
    markets=_ETH_MARKETS,
)


# Option B: transition infrastructure exists, but no future production policy
# is activated. A real future policy must be added as another explicit era
# after its market catalog and activation hour are reviewed.
BTC_POLICY_REGISTRY = MarketPolicyRegistry(
    base="BTC",
    eras=(
        MarketPolicyEra(
            activation_start_ms=0,
            policy=BTC_POLICY,
        ),
    ),
)

ETH_POLICY_REGISTRY = MarketPolicyRegistry(
    base="ETH",
    eras=(
        MarketPolicyEra(
            activation_start_ms=0,
            policy=ETH_POLICY,
        ),
    ),
)


_POLICY_REGISTRIES = {
    "BTC": BTC_POLICY_REGISTRY,
    "ETH": ETH_POLICY_REGISTRY,
}


def get_market_policy_registry(
    base: str,
) -> MarketPolicyRegistry:
    base_u = str(base or "").strip().upper()

    try:
        return _POLICY_REGISTRIES[base_u]
    except KeyError as exc:
        raise ValueError("Collector market policy supports BTC and ETH only") from exc


def get_market_policy(
    base: str,
    *,
    partition_start_ms: int | None = None,
) -> MarketPolicy:
    registry = get_market_policy_registry(base)

    if partition_start_ms is None:
        return registry.latest_policy

    return registry.era_for_timestamp_ms(
        partition_start_ms,
    ).policy


def market_policy_segments(
    base: str,
    *,
    from_ms: int,
    to_ms: int,
) -> tuple[MarketPolicySegment, ...]:
    return get_market_policy_registry(base).segments_for_range(
        from_ms,
        to_ms,
    )

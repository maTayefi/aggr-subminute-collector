# collector\market_policy.py
"""Reviewed BTC/ETH aggr.trade derivative-market policies.

The production request URLs supplied by the project owner contain both spot and
derivative markets. The collector must not reuse those URLs unchanged.

Every included market below has an explicit tier and was reviewed against the
supplied aggr-server catalog/exchange-adapter behavior. Version 1 does not
dynamically add products merely because a new name appears in /products.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


SOURCE_INTERVAL_MS = 5000
COLLECTOR_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class MarketSpec:
    market: str
    tier: str

    def __post_init__(self) -> None:
        market = str(self.market or "").strip()
        tier = str(self.tier or "").strip().upper()

        if not market or ":" not in market:
            raise ValueError(f"Invalid aggr market identifier: {self.market!r}")

        if tier not in {"USDT", "USDC", "USD"}:
            raise ValueError(
                f"Unsupported market tier {self.tier!r} for {market!r}"
            )

        object.__setattr__(self, "market", market)
        object.__setattr__(self, "tier", tier)


@dataclass(frozen=True, slots=True)
class MarketPolicy:
    base: str
    version: str
    price_market: str
    markets: tuple[MarketSpec, ...]

    def __post_init__(self) -> None:
        base = str(self.base or "").strip().upper()
        version = str(self.version or "").strip()
        price_market = str(self.price_market or "").strip()

        if base not in {"BTC", "ETH"}:
            raise ValueError(f"Unsupported policy base: {base!r}")

        if not version:
            raise ValueError("Market policy version must be non-empty")

        if not self.markets:
            raise ValueError("Market policy requires at least one market")

        names = [spec.market for spec in self.markets]

        if len(names) != len(set(names)):
            duplicates = sorted(
                name for name in set(names) if names.count(name) > 1
            )
            raise ValueError(
                f"Market policy contains duplicate markets: {duplicates}"
            )

        if price_market not in names:
            raise ValueError(
                "The exact Binance USDT perpetual price market must also be "
                "inside the selected CVD universe"
            )

        object.__setattr__(self, "base", base)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "price_market", price_market)

    @property
    def selected_markets(self) -> tuple[str, ...]:
        return tuple(spec.market for spec in self.markets)

    @property
    def tier_by_market(self) -> dict[str, str]:
        return {spec.market: spec.tier for spec in self.markets}

    @property
    def market_list_hash(self) -> str:
        payload = {
            "base": self.base,
            "version": self.version,
            "price_market": self.price_market,
            "markets": [
                {
                    "market": spec.market,
                    "tier": spec.tier,
                }
                for spec in self.markets
            ],
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()


def _specs(
    tier: str,
    markets: Iterable[str],
) -> tuple[MarketSpec, ...]:
    return tuple(MarketSpec(market, tier) for market in markets)


BTC_POLICY = MarketPolicy(
    base="BTC",
    version="btc-v1",
    price_market="BINANCE_FUTURES:btcusdt",
    markets=(
        *_specs(
            "USDT",
            (
                "ASTER:BTCUSDT",
                "BINANCE_FUTURES:btcusdt",
                "BITFINEX:BTCF0:USTF0",
                "BITGET:BTCUSDT",
                "BITMART:BTCUSDT",
                "BITMEX:XBTUSDT",
                "BITMEX:XBT_USDT",
                "BITUNIX:BTCUSDT",
                "BYBIT:BTCUSDT",
                "GATEIO:BTC_USDT",
                "HUOBI:BTC-USDT",
                "KUCOIN:XBTUSDTM",
                "MEXC:BTC_USDT",
                "OKEX:BTC-USDT-SWAP",
                "PHEMEX:BTCUSDT",
            ),
        ),
        *_specs(
            "USDC",
            (
                "BINANCE_FUTURES:btcusdc",
                "BITMART:BTCUSDC",
                "DERIBIT:BTC_USDC-PERPETUAL",
                "KUCOIN:XBTUSDCM",
                "OKEX:BTC-USDC-SWAP",
                "PHEMEX:BTCUSDC",
            ),
        ),
        *_specs(
            "USD",
            (
                "BINANCE_FUTURES:btcusd_perp",
                "BITGET:BTCUSD",
                "BITGET:BTCPERP",
                "BITMART:BTCUSD",
                "BITMEX:XBTUSD",
                "BYBIT:BTCUSD",
                "COINBASE:BTC-PERP-INTX",
                "DERIBIT:BTC-PERPETUAL",
                "DYDX:BTC-USD",
                "GATEIO:BTC_USD",
                "HUOBI:BTC-USD",
                "HYPERLIQUID:BTC",
                "KRAKEN:PF_XBTUSD",
                "KRAKEN:PI_XBTUSD",
                "KUCOIN:XBTUSDM",
                "MEXC:BTC_USD",
                "OKEX:BTC-USD-SWAP",
                "PHEMEX:BTCUSD",
            ),
        ),
    ),
)


ETH_POLICY = MarketPolicy(
    base="ETH",
    version="eth-v1",
    price_market="BINANCE_FUTURES:ethusdt",
    markets=(
        *_specs(
            "USDT",
            (
                "ASTER:ETHUSDT",
                "BINANCE_FUTURES:ethusdt",
                "BITFINEX:ETHF0:USTF0",
                "BITGET:ETHUSDT",
                "BITMART:ETHUSDT",
                "BITMEX:ETHUSDT",
                "BITMEX:ETH_USDT",
                "BITUNIX:ETHUSDT",
                "BYBIT:ETHUSDT",
                "GATEIO:ETH_USDT",
                "HUOBI:ETH-USDT",
                "KUCOIN:ETHUSDTM",
                "MEXC:ETH_USDT",
                "OKEX:ETH-USDT-SWAP",
                "PHEMEX:ETHUSDT",
            ),
        ),
        *_specs(
            "USDC",
            (
                "BINANCE_FUTURES:ethusdc",
                "BITMART:ETHUSDC",
                "DERIBIT:ETH_USDC-PERPETUAL",
                "KUCOIN:ETHUSDCM",
                "PHEMEX:ETHUSDC",
            ),
        ),
        *_specs(
            "USD",
            (
                "BINANCE_FUTURES:ethusd_perp",
                "BITGET:ETHUSD",
                "BITGET:ETHPERP",
                "BITMART:ETHUSD",
                "BITMEX:ETHUSD",
                "BYBIT:ETHUSD",
                "COINBASE:ETH-PERP-INTX",
                "DERIBIT:ETH-PERPETUAL",
                "DYDX:ETH-USD",
                "HUOBI:ETH-USD",
                "HYPERLIQUID:ETH",
                "KRAKEN:PF_ETHUSD",
                "KRAKEN:PI_ETHUSD",
                "KUCOIN:ETHUSDM",
                "MEXC:ETH_USD",
                "OKEX:ETH-USD-SWAP",
                "PHEMEX:ETHUSD",
            ),
        ),
    ),
)


_POLICIES = {
    "BTC": BTC_POLICY,
    "ETH": ETH_POLICY,
}


def get_market_policy(base: str) -> MarketPolicy:
    base_u = str(base or "").strip().upper()

    try:
        return _POLICIES[base_u]
    except KeyError as exc:
        raise ValueError(
            "Collector version 1 supports BTC and ETH only"
        ) from exc
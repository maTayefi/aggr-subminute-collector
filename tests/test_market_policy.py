# tests\test_market_policy.py
from __future__ import annotations

import pytest

from collector.market_policy import (
    BTC_POLICY,
    BTC_POLICY_REGISTRY,
    MARKET_POLICY_SCHEMA_VERSION,
    MarketPolicy,
    MarketPolicyEra,
    MarketPolicyRegistry,
    MarketSpec,
    get_market_policy,
)


@pytest.mark.parametrize(
    ("base", "price_market"),
    [
        ("BTC", "BINANCE_FUTURES:btcusdt"),
        ("ETH", "BINANCE_FUTURES:ethusdt"),
    ],
)
def test_policy_has_exact_binance_usdt_perpetual_price_source(
    base: str,
    price_market: str,
) -> None:
    policy = get_market_policy(base)

    assert policy.price_market == price_market
    assert price_market in policy.selected_markets


@pytest.mark.parametrize("base", ["BTC", "ETH"])
def test_policy_is_deterministic_and_hashable(base: str) -> None:
    first = get_market_policy(base)
    second = get_market_policy(base.lower())

    assert first is second
    assert first.selected_markets == tuple(sorted(first.selected_markets))

    assert len(first.market_list_hash) == 64
    assert first.market_list_hash == second.market_list_hash
    assert all(char in "0123456789abcdef" for char in first.market_list_hash)


@pytest.mark.parametrize("base", ["BTC", "ETH"])
def test_policy_excludes_explicit_spot_markets(base: str) -> None:
    policy = get_market_policy(base)
    markets = set(policy.selected_markets)

    forbidden_suffixes = (
        "-SPOT",
        "/USD",
        "/USDC",
        "/USDT",
    )

    assert not any(market.endswith(forbidden_suffixes) for market in markets)

    forbidden_exact = {
        "BINANCE:btcusdt",
        "BINANCE:ethusdt",
        "BINANCE_US:btcusdt",
        "BYBIT:BTCUSDT-SPOT",
        "BYBIT:ETHUSDT-SPOT",
        "COINBASE:BTC-USD",
        "COINBASE:ETH-USD",
        "MEXC:BTCUSDT",
        "POLONIEX:BTC_USDT",
        "POLONIEX:ETH_USDT",
    }

    assert not (markets & forbidden_exact)


@pytest.mark.parametrize("base", ["BTC", "ETH"])
def test_every_policy_tier_is_reviewed(base: str) -> None:
    policy = get_market_policy(base)

    assert set(policy.tier_by_market.values()).issubset(
        {
            "USDT",
            "USDC",
            "USD_INVERSE",
        }
    )

    assert "USDT" in set(policy.tier_by_market.values())
    assert "USD_INVERSE" in set(policy.tier_by_market.values())


def test_policy_rejects_unknown_response_market() -> None:
    policy = get_market_policy("BTC")

    with pytest.raises(
        ValueError,
        match="outside the reviewed",
    ):
        policy.assert_response_markets_allowed(
            [
                policy.price_market,
                "BINANCE:btcusdt",
            ]
        )


def test_policy_hash_changes_when_market_tier_changes() -> None:
    original = get_market_policy("BTC")

    modified_specs = tuple(
        MarketSpec(
            item.market,
            ("USDC" if item.market == "BITGET:BTCUSDT" else item.tier),
        )
        for item in original.markets
    )

    modified = MarketPolicy(
        base=original.base,
        version=original.version,
        price_market=original.price_market,
        markets=modified_specs,
    )

    assert modified.market_list_hash != original.market_list_hash


def test_market_policy_schema_identity_is_frozen() -> None:
    assert MARKET_POLICY_SCHEMA_VERSION == "smp_aggr_market_policy.v1"


def test_frozen_policy_hashes_match_consumer_release_contract() -> None:
    btc = get_market_policy("BTC")
    eth = get_market_policy("ETH")

    assert (
        btc.market_list_hash
        == "752dcf5147041ad1aa4c4f5404190f140e40d9282dc2541a3a03a7267dbcd05a"
    )
    assert (
        eth.market_list_hash
        == "05589b35e9a959cc8d6015af0b166dc58794a12dd5928578c6e62b915aa79906"
    )


def _synthetic_future_btc_policy() -> MarketPolicy:
    current = get_market_policy("BTC")

    return MarketPolicy(
        base="BTC",
        version="btc-derivatives-synthetic-future",
        price_market=current.price_market,
        markets=current.markets,
    )


def test_production_policy_registry_does_not_activate_future_policy() -> None:
    assert len(BTC_POLICY_REGISTRY.eras) == 1
    assert BTC_POLICY_REGISTRY.eras[0].activation_start_ms == 0
    assert BTC_POLICY_REGISTRY.latest_policy is BTC_POLICY


def test_policy_registry_splits_exactly_at_utc_hour_boundary() -> None:
    activation = 1_786_564_800_000
    future = _synthetic_future_btc_policy()

    registry = MarketPolicyRegistry(
        base="BTC",
        eras=(
            MarketPolicyEra(
                activation_start_ms=0,
                policy=BTC_POLICY,
            ),
            MarketPolicyEra(
                activation_start_ms=activation,
                policy=future,
            ),
        ),
    )

    segments = registry.segments_for_range(
        activation - 3_600_000,
        activation + 3_600_000,
    )

    assert len(segments) == 2

    assert segments[0].start_ms == activation - 3_600_000
    assert segments[0].end_ms == activation
    assert segments[0].policy is BTC_POLICY

    assert segments[1].start_ms == activation
    assert segments[1].end_ms == activation + 3_600_000
    assert segments[1].policy is future


@pytest.mark.parametrize(
    "activation",
    [
        -3_600_000,
        1,
        3_599_999,
        True,
    ],
)
def test_policy_era_requires_exact_nonnegative_utc_hour(
    activation,
) -> None:
    with pytest.raises(ValueError, match="UTC-hour boundary|integer"):
        MarketPolicyEra(
            activation_start_ms=activation,
            policy=BTC_POLICY,
        )

# tests\test_market_policy.py
from __future__ import annotations

import re

import pytest

from collector.market_policy import (
    BTC_POLICY,
    ETH_POLICY,
    SOURCE_INTERVAL_MS,
    get_market_policy,
)


def test_version_one_resolution_is_five_seconds() -> None:
    assert SOURCE_INTERVAL_MS == 5000


@pytest.mark.parametrize(
    ("policy", "expected_price"),
    [
        (BTC_POLICY, "BINANCE_FUTURES:btcusdt"),
        (ETH_POLICY, "BINANCE_FUTURES:ethusdt"),
    ],
)
def test_price_market_is_exact_binance_usdt_perpetual(
    policy,
    expected_price,
) -> None:
    assert policy.price_market == expected_price
    assert expected_price in policy.selected_markets
    assert policy.tier_by_market[expected_price] == "USDT"


@pytest.mark.parametrize("policy", [BTC_POLICY, ETH_POLICY])
def test_market_policy_contains_no_explicit_spot_products(policy) -> None:
    forbidden = (
        "-SPOT",
        "BINANCE:",
        "BINANCE_US:",
        "BITSTAMP:",
        "POLONIEX:",
    )

    failures = [
        market
        for market in policy.selected_markets
        if any(token in market for token in forbidden)
    ]

    assert failures == []


@pytest.mark.parametrize("policy", [BTC_POLICY, ETH_POLICY])
def test_market_policy_identity_is_stable_and_complete(policy) -> None:
    assert policy.version
    assert len(policy.selected_markets) == len(set(policy.selected_markets))
    assert set(policy.tier_by_market.values()) == {"USDT", "USDC", "USD"}
    assert re.fullmatch(r"[0-9a-f]{64}", policy.market_list_hash)


def test_unknown_base_fails_closed() -> None:
    with pytest.raises(ValueError, match="BTC and ETH only"):
        get_market_policy("SOL")
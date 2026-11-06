# tests\test_aggr_client.py
from __future__ import annotations

from urllib.parse import unquote

import pytest

from collector.aggr_client import (
    build_historical_url,
    parse_aggr_historical_payload,
)
from collector.market_policy import BTC_POLICY

FROM_MS = 1_786_560_615_000
TO_MS = 1_786_560_645_000


def _payload(rows: list[list]) -> dict:
    return {
        "format": "point",
        "columns": {
            "time": 0,
            "cbuy": 1,
            "close": 2,
            "csell": 3,
            "high": 4,
            "lbuy": 5,
            "low": 6,
            "lsell": 7,
            "market": 8,
            "open": 9,
            "vbuy": 10,
            "vsell": 11,
        },
        "results": rows,
    }


def _btc_row(
    *,
    timestamp: int = 1_786_560_615,
    market: str = "BINANCE_FUTURES:btcusdt",
    vbuy: float = 13_186.0144,
    vsell: float = 29_922.0624,
) -> list:
    return [
        timestamp,
        11,
        63_394.3,
        7,
        63_394.3,
        None,
        63_394.2,
        None,
        market,
        63_394.2,
        vbuy,
        vsell,
    ]


def test_builds_encoded_get_url_with_half_open_boundaries() -> None:
    markets = (
        "BINANCE_FUTURES:btcusdt",
        "DERIBIT:BTC_USDC-PERPETUAL",
    )

    url = build_historical_url(
        from_ms=FROM_MS,
        to_ms=TO_MS,
        timeframe_ms=5000,
        markets=markets,
    )

    assert url.startswith(
        "https://api.aggr.trade/historical/" f"{FROM_MS}/{TO_MS}/5000/"
    )

    decoded = unquote(url.rsplit("/", 1)[-1])
    assert decoded == "+".join(markets)
    assert "%3A" in url
    assert "%2B" in url


def test_rejects_unaligned_url_boundaries() -> None:
    with pytest.raises(ValueError, match="align"):
        build_historical_url(
            from_ms=FROM_MS + 1,
            to_ms=TO_MS,
            timeframe_ms=5000,
            markets=BTC_POLICY.selected_markets,
        )


def test_parser_uses_dynamic_column_indexes() -> None:
    payload = {
        "format": "point",
        "columns": {
            "market": 0,
            "time": 1,
            "open": 2,
            "high": 3,
            "low": 4,
            "close": 5,
            "cbuy": 6,
            "csell": 7,
            "vbuy": 8,
            "vsell": 9,
            "lbuy": 10,
            "lsell": 11,
        },
        "results": [
            [
                "BINANCE_FUTURES:btcusdt",
                1_786_560_615,
                63_394.2,
                63_394.3,
                63_394.2,
                63_394.3,
                11,
                7,
                13_186.0144,
                29_922.0624,
                None,
                None,
            ]
        ],
    }

    parsed = parse_aggr_historical_payload(
        payload,
        policy=BTC_POLICY,
        from_ms=FROM_MS,
        to_ms=TO_MS,
    )

    assert len(parsed.points) == 1
    point = parsed.points[0]

    assert point.market == "BINANCE_FUTURES:btcusdt"
    assert point.ts_utc_seconds == 1_786_560_615
    assert point.vbuy == pytest.approx(13_186.0144)
    assert point.vsell == pytest.approx(29_922.0624)
    assert point.market_tier == "USDT"


def test_exact_duplicates_are_collapsed() -> None:
    row = _btc_row()

    parsed = parse_aggr_historical_payload(
        _payload([row, list(row)]),
        policy=BTC_POLICY,
        from_ms=FROM_MS,
        to_ms=TO_MS,
    )

    assert len(parsed.points) == 1
    assert parsed.source_row_count == 2
    assert parsed.exact_duplicates_dropped == 1


def test_conflicting_duplicates_fail_closed() -> None:
    first = _btc_row(vbuy=100.0)
    second = _btc_row(vbuy=101.0)

    with pytest.raises(ValueError, match="Conflicting AGGR rows"):
        parse_aggr_historical_payload(
            _payload([first, second]),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_sub_nanoscopic_duplicate_difference_is_still_a_conflict() -> None:
    first = _btc_row(vbuy=100.0)
    second = _btc_row(vbuy=100.0 + 5e-10)

    with pytest.raises(ValueError, match="Conflicting AGGR rows"):
        parse_aggr_historical_payload(
            _payload([first, second]),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_market_outside_reviewed_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside reviewed policy"):
        parse_aggr_historical_payload(
            _payload(
                [
                    _btc_row(
                        market="BINANCE:btcusdt",
                    )
                ]
            ),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_result_end_boundary_is_exclusive() -> None:
    with pytest.raises(ValueError, match="outside requested"):
        parse_aggr_historical_payload(
            _payload(
                [
                    _btc_row(
                        timestamp=TO_MS // 1000,
                    )
                ]
            ),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_liquidation_only_row_has_zero_trade_flow() -> None:
    row = [
        1_786_560_615,
        None,
        None,
        None,
        None,
        2500.0,
        None,
        0.0,
        "BINANCE_FUTURES:btcusdt",
        None,
        None,
        None,
    ]

    parsed = parse_aggr_historical_payload(
        _payload([row]),
        policy=BTC_POLICY,
        from_ms=FROM_MS,
        to_ms=TO_MS,
    )

    point = parsed.points[0]

    assert point.open is None
    assert point.cbuy == 0
    assert point.csell == 0
    assert point.vbuy == 0.0
    assert point.vsell == 0.0
    assert point.lbuy == pytest.approx(2500.0)


def test_parser_rejects_null_ohlc_with_positive_trade_flow() -> None:
    row = _btc_row()

    # OHLC field indexes in _payload():
    # close=2, high=4, low=6, open=9.
    for index in (2, 4, 6, 9):
        row[index] = None

    with pytest.raises(
        ValueError,
        match="fully null OHLC.*zero ordinary trade",
    ):
        parse_aggr_historical_payload(
            _payload([row]),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_partial_null_trade_fields_are_rejected() -> None:
    row = _btc_row()
    row[1] = None

    with pytest.raises(ValueError, match="fully present or fully null"):
        parse_aggr_historical_payload(
            _payload([row]),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_malformed_ohlc_is_rejected() -> None:
    row = _btc_row()
    row[4] = 63_000.0  # high below open/close

    with pytest.raises(ValueError, match="high is below"):
        parse_aggr_historical_payload(
            _payload([row]),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_rejects_fractional_millisecond_boundary_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="millisecond integer"):
        build_historical_url(
            from_ms=float(FROM_MS) + 0.5,
            to_ms=TO_MS,
            timeframe_ms=5000,
            markets=BTC_POLICY.selected_markets,
        )


def test_parser_rejects_non_five_second_timeframe() -> None:
    with pytest.raises(
        ValueError,
        match="requires timeframe_ms=5000",
    ):
        parse_aggr_historical_payload(
            _payload([_btc_row()]),
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
            timeframe_ms=10_000,
        )


def test_rejects_fractional_timeframe_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="timeframe_ms must be an integer"):
        build_historical_url(
            from_ms=FROM_MS,
            to_ms=TO_MS,
            timeframe_ms=5000.9,
            markets=BTC_POLICY.selected_markets,
        )


def test_rejects_fractional_response_column_index() -> None:
    payload = _payload([_btc_row()])
    payload["columns"]["vbuy"] = 10.9

    with pytest.raises(ValueError, match="column index.*vbuy.*invalid"):
        parse_aggr_historical_payload(
            payload,
            policy=BTC_POLICY,
            from_ms=FROM_MS,
            to_ms=TO_MS,
        )


def test_successful_empty_response_accepts_empty_column_map() -> None:
    parsed = parse_aggr_historical_payload(
        {
            "format": "point",
            "columns": {},
            "results": [],
        },
        policy=BTC_POLICY,
        from_ms=FROM_MS,
        to_ms=TO_MS,
    )

    assert parsed.points == ()
    assert parsed.source_row_count == 0
    assert parsed.exact_duplicates_dropped == 0
    assert parsed.request_from_ms == FROM_MS
    assert parsed.request_to_ms == TO_MS


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_http_status_honors_retry_after(
    monkeypatch,
    status_code: int,
) -> None:
    import httpx

    import collector.aggr_client as aggr_client

    sleeps: list[float] = []

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def get(self, url: str) -> httpx.Response:
            self.calls += 1
            request = httpx.Request("GET", url)

            if self.calls == 1:
                return httpx.Response(
                    status_code,
                    headers={"Retry-After": "7"},
                    request=request,
                )

            return httpx.Response(
                200,
                json={
                    "format": "point",
                    "columns": {},
                    "results": [],
                },
                request=request,
            )

    fake = FakeClient()

    monkeypatch.setattr(
        aggr_client.httpx,
        "Client",
        lambda **kwargs: fake,
    )
    monkeypatch.setattr(
        aggr_client.time,
        "sleep",
        lambda delay: sleeps.append(float(delay)),
    )

    client = aggr_client.AggrHistoricalClient(
        max_attempts=2,
        initial_backoff_seconds=0.25,
    )

    start_ms = 1_800_000_000_000

    response = client.fetch(
        policy=BTC_POLICY,
        from_ms=start_ms,
        to_ms=start_ms + 5_000,
    )

    assert response.points == ()
    assert fake.calls == 2
    assert sleeps == pytest.approx([7.0])

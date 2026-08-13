# collector\aggr_client.py
"""Strict client for the aggr.trade five-second historical GET endpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Iterable
from urllib.parse import quote

import httpx

from collector.market_policy import (
    COLLECTOR_SCHEMA_VERSION,
    SOURCE_INTERVAL_MS,
    MarketPolicy,
)
from collector.models import (
    CanonicalMarketPoint,
    nonnegative_count,
    nonnegative_float,
    points_identical,
    validate_nullable_ohlc,
)


AGGR_HISTORICAL_BASE_URL = "https://api.aggr.trade/historical"

_REQUIRED_COLUMNS = {
    "time",
    "cbuy",
    "close",
    "csell",
    "high",
    "lbuy",
    "low",
    "lsell",
    "market",
    "open",
    "vbuy",
    "vsell",
}


@dataclass(frozen=True, slots=True)
class AggrHistoricalResponse:
    points: tuple[CanonicalMarketPoint, ...]
    exact_duplicates_dropped: int
    source_row_count: int
    request_from_ms: int
    request_to_ms: int

    @property
    def unique_row_count(self) -> int:
        return len(self.points)


def _strict_ms_boundary(
    name: str,
    value: Any,
    *,
    interval_ms: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer timestamp, not bool")

    try:
        output = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a UNIX millisecond integer; got {value!r}"
        ) from exc

    if output <= 0:
        raise ValueError(f"{name} must be positive; got {output}")

    if output % interval_ms != 0:
        raise ValueError(
            f"{name} must align to {interval_ms} milliseconds; got {output}"
        )

    return output


def build_historical_url(
    *,
    from_ms: int,
    to_ms: int,
    timeframe_ms: int,
    markets: Iterable[str],
    base_url: str = AGGR_HISTORICAL_BASE_URL,
) -> str:
    interval = int(timeframe_ms)

    if interval != SOURCE_INTERVAL_MS:
        raise ValueError(
            f"Collector version 1 requires timeframe_ms={SOURCE_INTERVAL_MS}"
        )

    start = _strict_ms_boundary(
        "from_ms",
        from_ms,
        interval_ms=interval,
    )
    end = _strict_ms_boundary(
        "to_ms",
        to_ms,
        interval_ms=interval,
    )

    if end <= start:
        raise ValueError(
            "to_ms must be greater than from_ms under [from, to) semantics"
        )

    normalized_markets = [
        str(market or "").strip()
        for market in markets
        if str(market or "").strip()
    ]

    if not normalized_markets:
        raise ValueError("At least one market is required")

    if len(normalized_markets) != len(set(normalized_markets)):
        raise ValueError("Historical request contains duplicate market names")

    encoded_markets = quote(
        "+".join(normalized_markets),
        safe="",
    )

    return (
        f"{str(base_url).rstrip('/')}/"
        f"{start}/{end}/{interval}/{encoded_markets}"
    )


def _column_map(payload: dict[str, Any]) -> dict[str, int]:
    columns = payload.get("columns")

    if not isinstance(columns, dict):
        raise ValueError("AGGR response columns must be an object")

    missing = sorted(_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ValueError(
            f"AGGR response is missing required columns: {missing}"
        )

    output: dict[str, int] = {}

    for name in _REQUIRED_COLUMNS:
        raw_index = columns.get(name)

        if isinstance(raw_index, bool):
            raise ValueError(
                f"AGGR column index for {name!r} must be an integer"
            )

        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"AGGR column index for {name!r} is invalid: {raw_index!r}"
            ) from exc

        if index < 0:
            raise ValueError(
                f"AGGR column index for {name!r} must be non-negative"
            )

        output[name] = index

    if len(set(output.values())) != len(output):
        raise ValueError(
            "AGGR response maps multiple required fields to the same index"
        )

    return output


def _row_value(
    row: list[Any],
    columns: dict[str, int],
    name: str,
) -> Any:
    index = columns[name]

    if index >= len(row):
        raise ValueError(
            f"AGGR row has no value for column {name!r} at index {index}"
        )

    return row[index]


def _timestamp_seconds(
    raw: Any,
    *,
    row_number: int,
    from_ms: int,
    to_ms: int,
    interval_ms: int,
) -> int:
    if isinstance(raw, bool):
        raise ValueError(
            f"AGGR row {row_number}: time must be numeric, not bool"
        )

    try:
        numeric = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"AGGR row {row_number}: invalid time {raw!r}"
        ) from exc

    if not numeric.is_integer():
        raise ValueError(
            f"AGGR row {row_number}: time must be whole UNIX seconds"
        )

    timestamp = int(numeric)
    interval_seconds = interval_ms // 1000

    if timestamp % interval_seconds != 0:
        raise ValueError(
            f"AGGR row {row_number}: time {timestamp} is not aligned to "
            f"{interval_seconds} seconds"
        )

    timestamp_ms = timestamp * 1000

    if timestamp_ms < from_ms or timestamp_ms >= to_ms:
        raise ValueError(
            f"AGGR row {row_number}: time {timestamp} is outside requested "
            f"[{from_ms}, {to_ms}) millisecond range"
        )

    return timestamp


def _flow_fields(
    row: list[Any],
    columns: dict[str, int],
    *,
    context: str,
) -> tuple[int, int, float, float, float | None, float | None]:
    raw_cbuy = _row_value(row, columns, "cbuy")
    raw_csell = _row_value(row, columns, "csell")
    raw_vbuy = _row_value(row, columns, "vbuy")
    raw_vsell = _row_value(row, columns, "vsell")
    raw_lbuy = _row_value(row, columns, "lbuy")
    raw_lsell = _row_value(row, columns, "lsell")

    trade_fields = (raw_cbuy, raw_csell, raw_vbuy, raw_vsell)
    trade_null_count = sum(value is None for value in trade_fields)

    if trade_null_count == 4:
        # Influx can emit a liquidation-only bucket where trade fields were not
        # written. Liquidation activity is separate from ordinary CVD, so its
        # buy/sell trade flow is a known zero for this row.
        if raw_lbuy is None and raw_lsell is None:
            raise ValueError(
                f"{context}: all trade and liquidation fields are null"
            )

        cbuy = 0
        csell = 0
        vbuy = 0.0
        vsell = 0.0

    elif trade_null_count:
        raise ValueError(
            f"{context}: trade count/volume fields must be either fully "
            "present or fully null for a liquidation-only row"
        )

    else:
        cbuy_value = nonnegative_count(
            raw_cbuy,
            name=f"{context}.cbuy",
        )
        csell_value = nonnegative_count(
            raw_csell,
            name=f"{context}.csell",
        )
        vbuy_value = nonnegative_float(
            raw_vbuy,
            name=f"{context}.vbuy",
        )
        vsell_value = nonnegative_float(
            raw_vsell,
            name=f"{context}.vsell",
        )

        assert cbuy_value is not None
        assert csell_value is not None
        assert vbuy_value is not None
        assert vsell_value is not None

        cbuy = cbuy_value
        csell = csell_value
        vbuy = vbuy_value
        vsell = vsell_value

    lbuy = nonnegative_float(
        raw_lbuy,
        name=f"{context}.lbuy",
        allow_none=True,
    )
    lsell = nonnegative_float(
        raw_lsell,
        name=f"{context}.lsell",
        allow_none=True,
    )

    return cbuy, csell, vbuy, vsell, lbuy, lsell


def parse_aggr_historical_payload(
    payload: Any,
    *,
    policy: MarketPolicy,
    from_ms: int,
    to_ms: int,
    timeframe_ms: int = SOURCE_INTERVAL_MS,
) -> AggrHistoricalResponse:
    """Parse one complete AGGR response into deterministic canonical points."""

    interval = int(timeframe_ms)

    start = _strict_ms_boundary(
        "from_ms",
        from_ms,
        interval_ms=interval,
    )
    end = _strict_ms_boundary(
        "to_ms",
        to_ms,
        interval_ms=interval,
    )

    if end <= start:
        raise ValueError("to_ms must be greater than from_ms")

    if not isinstance(payload, dict):
        raise ValueError("AGGR response must be one JSON object")

    if payload.get("format") != "point":
        raise ValueError(
            f"Unsupported AGGR response format: {payload.get('format')!r}"
        )

    columns = _column_map(payload)
    results = payload.get("results")

    if not isinstance(results, list):
        raise ValueError("AGGR response results must be a list")

    selected = set(policy.selected_markets)
    tier_by_market = policy.tier_by_market

    unique: dict[tuple[int, str], CanonicalMarketPoint] = {}
    exact_duplicates = 0

    for row_number, raw_row in enumerate(results, start=1):
        if not isinstance(raw_row, list):
            raise ValueError(
                f"AGGR result row {row_number} must be a list"
            )

        context = f"AGGR row {row_number}"

        timestamp = _timestamp_seconds(
            _row_value(raw_row, columns, "time"),
            row_number=row_number,
            from_ms=start,
            to_ms=end,
            interval_ms=interval,
        )

        market = str(
            _row_value(raw_row, columns, "market") or ""
        ).strip()

        if not market:
            raise ValueError(f"{context}: market must be non-empty")

        if market not in selected:
            raise ValueError(
                f"{context}: API returned market {market!r} outside reviewed "
                f"policy {policy.version!r}"
            )

        o, h, l_, c = validate_nullable_ohlc(
            open_=_row_value(raw_row, columns, "open"),
            high=_row_value(raw_row, columns, "high"),
            low=_row_value(raw_row, columns, "low"),
            close=_row_value(raw_row, columns, "close"),
            context=context,
        )

        cbuy, csell, vbuy, vsell, lbuy, lsell = _flow_fields(
            raw_row,
            columns,
            context=context,
        )

        point = CanonicalMarketPoint(
            ts_utc_seconds=timestamp,
            market=market,
            open=o,
            high=h,
            low=l_,
            close=c,
            cbuy=cbuy,
            csell=csell,
            vbuy=vbuy,
            vsell=vsell,
            lbuy=lbuy,
            lsell=lsell,
            source_interval_ms=interval,
            collector_schema_version=COLLECTOR_SCHEMA_VERSION,
            market_policy_version=policy.version,
            market_tier=tier_by_market[market],
        )

        existing = unique.get(point.key)

        if existing is None:
            unique[point.key] = point
            continue

        if points_identical(existing, point):
            exact_duplicates += 1
            continue

        raise ValueError(
            "Conflicting AGGR rows share the same (time, market) key: "
            f"{point.key!r}"
        )

    points = tuple(
        unique[key]
        for key in sorted(unique)
    )

    return AggrHistoricalResponse(
        points=points,
        exact_duplicates_dropped=exact_duplicates,
        source_row_count=len(results),
        request_from_ms=start,
        request_to_ms=end,
    )


class AggrHistoricalClient:
    """Retrying synchronous client suitable for a GitHub Actions worker."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        initial_backoff_seconds: float = 1.0,
        base_url: str = AGGR_HISTORICAL_BASE_URL,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        if initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be > 0")

        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self.initial_backoff_seconds = float(initial_backoff_seconds)
        self.base_url = str(base_url).rstrip("/")

    def fetch(
        self,
        *,
        policy: MarketPolicy,
        from_ms: int,
        to_ms: int,
        timeframe_ms: int = SOURCE_INTERVAL_MS,
    ) -> AggrHistoricalResponse:
        url = build_historical_url(
            from_ms=from_ms,
            to_ms=to_ms,
            timeframe_ms=timeframe_ms,
            markets=policy.selected_markets,
            base_url=self.base_url,
        )

        headers = {
            "Accept": "application/json, */*",
            "Origin": "https://charts.aggr.trade",
            "Referer": "https://charts.aggr.trade/",
            "User-Agent": (
                "Mozilla/5.0 (compatible; "
                "smp-aggr-subminute-collector/1.0)"
            ),
        }

        last_error: BaseException | None = None

        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers=headers,
        ) as client:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    response = client.get(url)

                    if response.status_code == 429:
                        retry_after_raw = response.headers.get("Retry-After")
                        try:
                            retry_after = float(retry_after_raw or 0.0)
                        except ValueError:
                            retry_after = 0.0

                        raise _RetryableAggrError(
                            f"AGGR returned HTTP 429 for {policy.base}",
                            retry_after_seconds=max(0.0, retry_after),
                        )

                    if 500 <= response.status_code <= 599:
                        raise _RetryableAggrError(
                            f"AGGR returned HTTP {response.status_code} "
                            f"for {policy.base}"
                        )

                    response.raise_for_status()

                    try:
                        payload = response.json()
                    except json.JSONDecodeError as exc:
                        raise _RetryableAggrError(
                            f"AGGR returned invalid JSON for {policy.base}"
                        ) from exc

                    return parse_aggr_historical_payload(
                        payload,
                        policy=policy,
                        from_ms=from_ms,
                        to_ms=to_ms,
                        timeframe_ms=timeframe_ms,
                    )

                except (
                    httpx.TimeoutException,
                    httpx.TransportError,
                    _RetryableAggrError,
                ) as exc:
                    last_error = exc

                    if attempt >= self.max_attempts:
                        break

                    delay = self.initial_backoff_seconds * (2 ** (attempt - 1))

                    if isinstance(exc, _RetryableAggrError):
                        delay = max(delay, exc.retry_after_seconds)

                    time.sleep(min(delay, 30.0))

        raise RuntimeError(
            f"AGGR historical GET failed after {self.max_attempts} attempt(s) "
            f"for base={policy.base}, range=[{from_ms}, {to_ms})"
        ) from last_error


class _RetryableAggrError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = float(retry_after_seconds)
# collector\partition.py
"""Deterministic hourly Parquet storage for canonical aggr.trade points."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from collections.abc import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

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

PARQUET_SCHEMA_VERSION = "smp_aggr_parquet.v1"

PARQUET_COLUMNS = (
    "ts_utc",
    "market",
    "open",
    "high",
    "low",
    "close",
    "cbuy",
    "csell",
    "vbuy",
    "vsell",
    "lbuy",
    "lsell",
    "source_interval_ms",
    "collector_schema_version",
    "market_policy_version",
    "market_tier",
)

PARQUET_SCHEMA = pa.schema(
    [
        # Parquet timestamp logical types are persisted at millisecond
        # precision here. Every collector timestamp is still required to align
        # exactly to a five-second boundary.
        pa.field("ts_utc", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=True),
        pa.field("high", pa.float64(), nullable=True),
        pa.field("low", pa.float64(), nullable=True),
        pa.field("close", pa.float64(), nullable=True),
        pa.field("cbuy", pa.int64(), nullable=False),
        pa.field("csell", pa.int64(), nullable=False),
        pa.field("vbuy", pa.float64(), nullable=False),
        pa.field("vsell", pa.float64(), nullable=False),
        pa.field("lbuy", pa.float64(), nullable=True),
        pa.field("lsell", pa.float64(), nullable=True),
        pa.field("source_interval_ms", pa.int32(), nullable=False),
        pa.field("collector_schema_version", pa.string(), nullable=False),
        pa.field("market_policy_version", pa.string(), nullable=False),
        pa.field("market_tier", pa.string(), nullable=False),
    ],
    metadata={
        b"schema": PARQUET_SCHEMA_VERSION.encode("ascii"),
        b"source_interval_ms": str(SOURCE_INTERVAL_MS).encode("ascii"),
        b"collector_schema_version": COLLECTOR_SCHEMA_VERSION.encode("ascii"),
    },
)


@dataclass(frozen=True, slots=True)
class HourlyPartition:
    """One validated deterministic hourly Parquet object."""

    base: str
    partition_start_ms: int
    partition_end_ms: int
    data_file: str
    local_path: Path
    data_sha256: str
    row_count: int
    points: tuple[CanonicalMarketPoint, ...]
    exact_duplicates_dropped: int


def _strict_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        text = value.strip()
        if text and text.isdigit():
            return int(text)

    raise ValueError(f"{name} must be an integer; got {value!r}")


def _hour_start_ms(value: Any) -> int:
    timestamp_ms = _strict_int("partition_start_ms", value)

    if timestamp_ms <= 0:
        raise ValueError("partition_start_ms must be positive")

    if timestamp_ms % 3_600_000 != 0:
        raise ValueError(
            "partition_start_ms must align to a UTC hour; " f"got {timestamp_ms}"
        )

    return timestamp_ms


def partition_data_path(
    base: str,
    partition_start_ms: int,
) -> str:
    base_u = str(base or "").strip().upper()
    if base_u not in {"BTC", "ETH"}:
        raise ValueError("Partition base must be BTC or ETH")

    start_ms = _hour_start_ms(partition_start_ms)
    timestamp = datetime.fromtimestamp(
        start_ms / 1000,
        tz=timezone.utc,
    )

    return (
        f"market_points/{base_u}/{SOURCE_INTERVAL_MS}/"
        f"{timestamp:%Y/%m/%d/%H}.parquet"
    )


def partition_start_ms_for_point(
    point: CanonicalMarketPoint,
) -> int:
    if not isinstance(point, CanonicalMarketPoint):
        raise TypeError("partition_start_ms_for_point requires CanonicalMarketPoint")

    raw_timestamp = point.ts_utc_seconds

    if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, int):
        raise ValueError(
            "Point timestamp must be an exact UNIX-seconds integer; "
            f"got {raw_timestamp!r}"
        )

    interval_seconds = SOURCE_INTERVAL_MS // 1000

    if raw_timestamp <= 0 or raw_timestamp % interval_seconds != 0:
        raise ValueError(
            f"Point timestamp must align to {interval_seconds} seconds; "
            f"got {raw_timestamp!r}"
        )

    timestamp_ms = raw_timestamp * 1000
    return timestamp_ms - (timestamp_ms % 3_600_000)


def _assert_point_owned(
    point: CanonicalMarketPoint,
    *,
    policy: MarketPolicy,
    partition_start_ms: int,
) -> None:
    raw_timestamp = point.ts_utc_seconds

    if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, int):
        raise ValueError(
            f"Point timestamp must be an exact UNIX-seconds integer; "
            f"got {raw_timestamp!r}"
        )

    interval_seconds = SOURCE_INTERVAL_MS // 1000

    if raw_timestamp <= 0 or raw_timestamp % interval_seconds != 0:
        raise ValueError(
            f"Point timestamp must align to {interval_seconds} seconds; "
            f"got {raw_timestamp!r}"
        )

    partition_end_ms = partition_start_ms + 3_600_000
    timestamp_ms = raw_timestamp * 1000

    if timestamp_ms < partition_start_ms or timestamp_ms >= partition_end_ms:
        raise ValueError(
            f"Point {point.key!r} is outside hourly partition "
            f"[{partition_start_ms}, {partition_end_ms})"
        )

    if point.market not in set(policy.selected_markets):
        raise ValueError(
            f"Point market {point.market!r} is outside policy " f"{policy.version!r}"
        )

    if point.source_interval_ms != SOURCE_INTERVAL_MS:
        raise ValueError(
            f"Point {point.key!r} has source_interval_ms="
            f"{point.source_interval_ms}; expected {SOURCE_INTERVAL_MS}"
        )

    if point.collector_schema_version != COLLECTOR_SCHEMA_VERSION:
        raise ValueError(
            f"Point {point.key!r} has collector schema "
            f"{point.collector_schema_version!r}"
        )

    if point.market_policy_version != policy.version:
        raise ValueError(
            f"Point {point.key!r} has market policy "
            f"{point.market_policy_version!r}; expected {policy.version!r}"
        )

    expected_tier = policy.tier_by_market[point.market]
    if point.market_tier != expected_tier:
        raise ValueError(
            f"Point {point.key!r} has tier {point.market_tier!r}; "
            f"expected {expected_tier!r}"
        )

    context = f"Canonical point {point.key!r}"

    # CanonicalMarketPoint type annotations are not runtime validation. The
    # partition writer is a final archive trust boundary and must reject a
    # malformed directly constructed dataclass instead of relying only on the
    # normal AGGR response parser.
    validated_ohlc = validate_nullable_ohlc(
        open_=point.open,
        high=point.high,
        low=point.low,
        close=point.close,
        context=context,
    )

    cbuy = nonnegative_count(
        point.cbuy,
        name=f"{context}.cbuy",
    )
    csell = nonnegative_count(
        point.csell,
        name=f"{context}.csell",
    )
    vbuy = nonnegative_float(
        point.vbuy,
        name=f"{context}.vbuy",
    )
    vsell = nonnegative_float(
        point.vsell,
        name=f"{context}.vsell",
    )
    lbuy = nonnegative_float(
        point.lbuy,
        name=f"{context}.lbuy",
        allow_none=True,
    )
    lsell = nonnegative_float(
        point.lsell,
        name=f"{context}.lsell",
        allow_none=True,
    )

    if validated_ohlc[0] is None:
        assert cbuy is not None
        assert csell is not None
        assert vbuy is not None
        assert vsell is not None

        if cbuy > 0 or csell > 0 or vbuy > 0.0 or vsell > 0.0:
            raise ValueError(
                f"{context}: fully null OHLC is permitted only for a "
                "liquidation-only bucket with zero ordinary trade flow"
            )

        if not (
            (lbuy is not None and lbuy > 0.0) or (lsell is not None and lsell > 0.0)
        ):
            raise ValueError(
                f"{context}: fully null OHLC requires positive liquidation " "activity"
            )


def merge_partition_points(
    existing_points: Iterable[CanonicalMarketPoint],
    incoming_points: Iterable[CanonicalMarketPoint],
    *,
    policy: MarketPolicy,
    partition_start_ms: int,
    replace_existing_conflicts: bool = False,
) -> tuple[tuple[CanonicalMarketPoint, ...], int]:
    """Merge one hourly object by ``(ts_utc, market)``.

    Exact duplicate rows collapse.

    By default, every conflicting value fails closed. Reconciliation may set
    ``replace_existing_conflicts=True`` after the complete incoming AGGR
    response has already passed strict duplicate validation. In that mode, the
    newly fetched row is authoritative over an older archived row with the same
    key.

    Conflicts between two incoming rows remain fatal in every mode.
    """

    start_ms = _hour_start_ms(partition_start_ms)
    unique: dict[tuple[int, str], CanonicalMarketPoint] = {}
    incoming_keys_seen: set[tuple[int, str]] = set()
    exact_duplicates = 0

    def _validate_point(point: CanonicalMarketPoint) -> None:
        if not isinstance(point, CanonicalMarketPoint):
            raise TypeError(
                "Hourly partition merge accepts CanonicalMarketPoint values"
            )

        _assert_point_owned(
            point,
            policy=policy,
            partition_start_ms=start_ms,
        )

    for point in tuple(existing_points):
        _validate_point(point)

        previous = unique.get(point.key)

        if previous is None:
            unique[point.key] = point
            continue

        if points_identical(previous, point):
            exact_duplicates += 1
            continue

        raise ValueError(
            "Conflicting existing hourly partition rows share the same "
            f"(ts_utc, market) key: {point.key!r}"
        )

    for point in tuple(incoming_points):
        _validate_point(point)

        previous = unique.get(point.key)

        if previous is None:
            unique[point.key] = point
            incoming_keys_seen.add(point.key)
            continue

        if points_identical(previous, point):
            exact_duplicates += 1
            incoming_keys_seen.add(point.key)
            continue

        if point.key in incoming_keys_seen:
            raise ValueError(
                "Conflicting incoming hourly partition rows share the same "
                f"(ts_utc, market) key: {point.key!r}"
            )

        if replace_existing_conflicts:
            # parse_aggr_historical_payload() already proved that the complete
            # incoming response contains at most one canonical value per key.
            # A differing value here is therefore a source correction relative
            # to the older archived snapshot, not an unresolved duplicate
            # inside the new response.
            unique[point.key] = point
            incoming_keys_seen.add(point.key)
            continue

        raise ValueError(
            "Conflicting hourly partition rows share the same "
            f"(ts_utc, market) key: {point.key!r}"
        )

    return (
        tuple(unique[key] for key in sorted(unique)),
        exact_duplicates,
    )


def _point_to_arrow_record(
    point: CanonicalMarketPoint,
) -> dict[str, Any]:
    return {
        "ts_utc": datetime.fromtimestamp(
            point.ts_utc_seconds,
            tz=timezone.utc,
        ),
        "market": point.market,
        "open": point.open,
        "high": point.high,
        "low": point.low,
        "close": point.close,
        "cbuy": point.cbuy,
        "csell": point.csell,
        "vbuy": point.vbuy,
        "vsell": point.vsell,
        "lbuy": point.lbuy,
        "lsell": point.lsell,
        "source_interval_ms": point.source_interval_ms,
        "collector_schema_version": point.collector_schema_version,
        "market_policy_version": point.market_policy_version,
        "market_tier": point.market_tier,
    }


def points_to_arrow_table(
    points: Iterable[CanonicalMarketPoint],
) -> pa.Table:
    records = [_point_to_arrow_record(point) for point in points]

    return pa.Table.from_pylist(
        records,
        schema=PARQUET_SCHEMA,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _atomic_parquet_write(
    path: Path,
    table: pa.Table,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)

    try:
        # Keep initial-close failures inside the cleanup boundary.
        # A failed close must not be retried using the same descriptor
        # number because it may already have been released and reused.
        os.close(descriptor)

        pq.write_table(
            table,
            temp_path,
            version="2.6",
            compression="zstd",
            compression_level=9,
            use_dictionary=[
                "market",
                "collector_schema_version",
                "market_policy_version",
                "market_tier",
            ],
            write_statistics=True,
            data_page_version="2.0",
            use_deprecated_int96_timestamps=False,
            coerce_timestamps="ms",
            allow_truncated_timestamps=False,
            row_group_size=65_536,
        )

        os.replace(temp_path, path)

    except BaseException as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            exc.add_note(
                "Could not remove unpublished Parquet temporary "
                f"file {temp_path}: {cleanup_error}"
            )

        raise


def write_hourly_partition(
    root: Path,
    *,
    policy: MarketPolicy,
    partition_start_ms: int,
    points: Iterable[CanonicalMarketPoint],
) -> HourlyPartition:
    """Validate and atomically write one canonical hourly Parquet file."""

    start_ms = _hour_start_ms(partition_start_ms)

    merged, exact_duplicates = merge_partition_points(
        (),
        points,
        policy=policy,
        partition_start_ms=start_ms,
    )

    data_file = partition_data_path(
        policy.base,
        start_ms,
    )
    local_path = Path(root) / data_file

    table = points_to_arrow_table(merged)
    _atomic_parquet_write(local_path, table)

    return HourlyPartition(
        base=policy.base,
        partition_start_ms=start_ms,
        partition_end_ms=start_ms + 3_600_000,
        data_file=data_file,
        local_path=local_path,
        data_sha256=sha256_file(local_path),
        row_count=len(merged),
        points=merged,
        exact_duplicates_dropped=exact_duplicates,
    )


def _timestamp_seconds(value: Any) -> int:
    if not isinstance(value, datetime):
        raise ValueError(f"Parquet ts_utc must decode as a datetime; got {value!r}")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Parquet ts_utc values must be timezone-aware")

    value_utc = value.astimezone(timezone.utc)

    # PARQUET_SCHEMA stores millisecond timestamps, so a malformed or manually
    # produced file can contain subsecond values even though collector output
    # must align to exact five-second buckets. Never truncate those values with
    # int(datetime.timestamp()).
    if value_utc.microsecond != 0:
        raise ValueError(
            "Parquet ts_utc values must be whole-second aligned; " f"got {value_utc!r}"
        )

    timestamp = int(value_utc.timestamp())
    interval_seconds = SOURCE_INTERVAL_MS // 1000

    if timestamp <= 0 or timestamp % interval_seconds != 0:
        raise ValueError(
            f"Parquet ts_utc values must align to {interval_seconds} seconds; "
            f"got {value_utc!r}"
        )

    return timestamp


def _point_from_record(
    record: Mapping[str, Any],
    *,
    policy: MarketPolicy,
) -> CanonicalMarketPoint:
    market = str(record.get("market") or "").strip()

    if market not in set(policy.selected_markets):
        raise ValueError(
            f"Parquet row market {market!r} is outside policy " f"{policy.version!r}"
        )

    context = f"Parquet row ({record.get('ts_utc')!r}, {market!r})"

    o, h, l_, c = validate_nullable_ohlc(
        open_=record.get("open"),
        high=record.get("high"),
        low=record.get("low"),
        close=record.get("close"),
        context=context,
    )

    cbuy = nonnegative_count(record.get("cbuy"), name=f"{context}.cbuy")
    csell = nonnegative_count(record.get("csell"), name=f"{context}.csell")
    vbuy = nonnegative_float(record.get("vbuy"), name=f"{context}.vbuy")
    vsell = nonnegative_float(record.get("vsell"), name=f"{context}.vsell")
    lbuy = nonnegative_float(
        record.get("lbuy"),
        name=f"{context}.lbuy",
        allow_none=True,
    )
    lsell = nonnegative_float(
        record.get("lsell"),
        name=f"{context}.lsell",
        allow_none=True,
    )

    assert cbuy is not None
    assert csell is not None
    assert vbuy is not None
    assert vsell is not None

    interval = _strict_int(
        "Parquet source_interval_ms",
        record.get("source_interval_ms"),
    )
    collector_version = str(record.get("collector_schema_version") or "").strip()
    policy_version = str(record.get("market_policy_version") or "").strip()
    tier = str(record.get("market_tier") or "").strip()

    if interval != SOURCE_INTERVAL_MS:
        raise ValueError("Parquet source_interval_ms is inconsistent")

    if collector_version != COLLECTOR_SCHEMA_VERSION:
        raise ValueError("Parquet collector_schema_version is inconsistent")

    if policy_version != policy.version:
        raise ValueError("Parquet market_policy_version is inconsistent")

    if tier != policy.tier_by_market[market]:
        raise ValueError("Parquet market_tier is inconsistent")

    return CanonicalMarketPoint(
        ts_utc_seconds=_timestamp_seconds(record.get("ts_utc")),
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
        collector_schema_version=collector_version,
        market_policy_version=policy_version,
        market_tier=tier,
    )


def read_hourly_partition(
    path: Path,
    *,
    policy: MarketPolicy,
    partition_start_ms: int,
    expected_row_count: int | None = None,
) -> tuple[CanonicalMarketPoint, ...]:
    """Validate physical rows, then return canonical unique market points."""

    start_ms = _hour_start_ms(partition_start_ms)

    if expected_row_count is not None:
        if (
            isinstance(expected_row_count, bool)
            or not isinstance(expected_row_count, int)
            or expected_row_count < 0
        ):
            raise ValueError("expected_row_count must be a nonnegative integer")

    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise ValueError(f"Could not read Parquet partition {path}: {exc}") from exc

    if not table.schema.equals(PARQUET_SCHEMA, check_metadata=True):
        raise ValueError(
            f"Parquet schema mismatch for {path}; expected " f"{PARQUET_SCHEMA_VERSION}"
        )

    if expected_row_count is not None and table.num_rows != expected_row_count:
        raise ValueError(
            f"Parquet row_count mismatch for {path}: "
            f"manifest={expected_row_count}, parquet={table.num_rows}"
        )

    points = tuple(
        _point_from_record(record, policy=policy) for record in table.to_pylist()
    )

    merged, _duplicates = merge_partition_points(
        (),
        points,
        policy=policy,
        partition_start_ms=start_ms,
    )

    return merged

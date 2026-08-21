from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collector.market_policy import (
    BTC_POLICY,
    COLLECTOR_SCHEMA_VERSION,
    SOURCE_INTERVAL_MS,
)
from collector.models import CanonicalMarketPoint
from collector.partition import (
    merge_partition_points,
    partition_data_path,
    partition_start_ms_for_point,
    read_hourly_partition,
    write_hourly_partition,
)

PARTITION_START_MS = 1_786_557_600_000


def _point(
    *,
    timestamp: int = 1_786_560_615,
    market: str = "BINANCE_FUTURES:btcusdt",
    vbuy: float = 100.0,
) -> CanonicalMarketPoint:
    return CanonicalMarketPoint(
        ts_utc_seconds=timestamp,
        market=market,
        open=63_394.2,
        high=63_394.3,
        low=63_394.2,
        close=63_394.3,
        cbuy=11,
        csell=7,
        vbuy=vbuy,
        vsell=29_922.0624,
        lbuy=None,
        lsell=None,
        source_interval_ms=SOURCE_INTERVAL_MS,
        collector_schema_version=COLLECTOR_SCHEMA_VERSION,
        market_policy_version=BTC_POLICY.version,
        market_tier=BTC_POLICY.tier_by_market[market],
    )


def test_partition_path_is_deterministic() -> None:
    assert (
        partition_data_path(
            "btc",
            PARTITION_START_MS,
        )
        == "market_points/BTC/5000/2026/08/12/18.parquet"
    )


def test_partition_merge_collapses_exact_duplicates() -> None:
    point = _point()

    merged, dropped = merge_partition_points(
        [point],
        [point],
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
    )

    assert merged == (point,)
    assert dropped == 1


def test_partition_merge_rejects_conflicts() -> None:
    first = _point(vbuy=100.0)
    second = replace(first, vbuy=101.0)

    with pytest.raises(ValueError, match="Conflicting hourly partition"):
        merge_partition_points(
            [first],
            [second],
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
        )


def test_authoritative_refetch_replaces_older_archived_value() -> None:
    archived = _point(vbuy=100.0)
    refetched = replace(archived, vbuy=101.0)

    merged, dropped = merge_partition_points(
        [archived],
        [refetched],
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        replace_existing_conflicts=True,
    )

    assert merged == (refetched,)
    assert dropped == 0


def test_authoritative_refetch_still_rejects_conflicting_incoming_rows() -> None:
    archived = _point(vbuy=100.0)
    first_incoming = replace(archived, vbuy=101.0)
    second_incoming = replace(archived, vbuy=102.0)

    with pytest.raises(
        ValueError,
        match="Conflicting incoming hourly partition",
    ):
        merge_partition_points(
            [archived],
            [first_incoming, second_incoming],
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
            replace_existing_conflicts=True,
        )


def test_partition_merge_rejects_sub_nanoscopic_conflict() -> None:
    first = _point(vbuy=100.0)
    second = replace(first, vbuy=100.0 + 5e-10)

    with pytest.raises(ValueError, match="Conflicting hourly partition"):
        merge_partition_points(
            [first],
            [second],
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
        )


def test_partition_write_is_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    points = (
        _point(),
        _point(
            timestamp=1_786_560_620,
            market="BYBIT:BTCUSDT",
            vbuy=200.0,
        ),
    )

    first = write_hourly_partition(
        tmp_path / "first",
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        points=points,
    )
    second = write_hourly_partition(
        tmp_path / "second",
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        points=points,
    )

    assert first.data_sha256 == second.data_sha256
    assert first.local_path.read_bytes() == second.local_path.read_bytes()
    assert first.row_count == 2

    import pyarrow as pa
    import pyarrow.parquet as pq

    stored_table = pq.read_table(first.local_path)
    assert stored_table.schema.field("ts_utc").type == pa.timestamp(
        "ms",
        tz="UTC",
    )

    loaded = read_hourly_partition(
        first.local_path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
    )

    assert loaded == points


def test_partition_rejects_point_from_another_hour(
    tmp_path: Path,
) -> None:
    point = _point(timestamp=1_786_564_800)

    with pytest.raises(ValueError, match="outside hourly partition"):
        write_hourly_partition(
            tmp_path,
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
            points=[point],
        )


def test_partition_rejects_non_five_second_point_timestamp(
    tmp_path: Path,
) -> None:
    point = _point(timestamp=1_786_560_616)

    with pytest.raises(ValueError, match="align to 5 seconds"):
        write_hourly_partition(
            tmp_path,
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
            points=[point],
        )


def test_partition_rejects_non_integer_point_timestamp(
    tmp_path: Path,
) -> None:
    point = _point()

    # Type annotations are not runtime validation. A direct caller can still
    # construct a dataclass carrying a fractional timestamp.
    malformed = replace(
        point,
        ts_utc_seconds=1_786_560_615.5,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="exact UNIX-seconds integer"):
        write_hourly_partition(
            tmp_path,
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
            points=[malformed],
        )


def test_partition_start_helper_rejects_fractional_timestamp() -> None:
    malformed = replace(
        _point(),
        ts_utc_seconds=1_786_560_615.5,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="exact UNIX-seconds integer"):
        partition_start_ms_for_point(malformed)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cbuy": 1.5}, r"cbuy.*integer"),
        ({"csell": -1}, r"csell.*non-negative"),
        ({"vbuy": -0.01}, r"vbuy.*non-negative"),
        ({"vsell": float("inf")}, r"vsell.*finite"),
        ({"lbuy": -1.0}, r"lbuy.*non-negative"),
        ({"open": None}, r"fully present or fully null"),
        ({"high": 50.0}, r"high is below"),
    ],
)
def test_partition_writer_rejects_malformed_direct_canonical_point(
    tmp_path: Path,
    changes: dict,
    message: str,
) -> None:
    malformed = replace(
        _point(),
        **changes,
    )

    with pytest.raises(ValueError, match=message):
        write_hourly_partition(
            tmp_path,
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
            points=[malformed],
        )


def test_partition_writer_rejects_null_ohlc_with_positive_trade_flow(
    tmp_path: Path,
) -> None:
    malformed = replace(
        _point(),
        open=None,
        high=None,
        low=None,
        close=None,
    )

    with pytest.raises(
        ValueError,
        match="fully null OHLC.*zero ordinary trade flow",
    ):
        write_hourly_partition(
            tmp_path,
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
            points=[malformed],
        )


def test_partition_reader_rejects_subsecond_parquet_timestamp(
    tmp_path: Path,
) -> None:
    partition = write_hourly_partition(
        tmp_path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        points=[_point()],
    )

    table = pq.read_table(partition.local_path)
    timestamps = table.column("ts_utc").to_pylist()

    # The physical Parquet schema allows millisecond precision. Simulate a
    # malformed/manually produced object that is one millisecond off the
    # canonical five-second bucket.
    timestamps[0] = timestamps[0].replace(microsecond=1_000)

    timestamp_array = pa.array(
        timestamps,
        type=table.schema.field("ts_utc").type,
    )
    malformed_table = table.set_column(
        table.schema.get_field_index("ts_utc"),
        table.schema.field("ts_utc"),
        timestamp_array,
    )

    pq.write_table(
        malformed_table,
        partition.local_path,
        version="2.6",
        compression="zstd",
    )

    with pytest.raises(ValueError, match="whole-second aligned"):
        read_hourly_partition(
            partition.local_path,
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
        )

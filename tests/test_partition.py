from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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

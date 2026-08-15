from __future__ import annotations

import json
from pathlib import Path

from collector.manifest import (
    PARTITION_MANIFEST_SCHEMA,
    CoveredRange,
    build_partition_manifest,
    partition_manifest_path,
    read_partition_manifest,
    write_partition_manifest,
)
from collector.market_policy import BTC_POLICY
from collector.partition import write_hourly_partition
from tests.test_partition import PARTITION_START_MS, _point


def test_manifest_path_is_deterministic() -> None:
    assert partition_manifest_path(
        "BTC",
        PARTITION_START_MS,
    ).startswith("coverage/BTC/5000/")


def test_manifest_combines_adjacent_coverage_ranges(
    tmp_path: Path,
) -> None:
    partition = write_hourly_partition(
        tmp_path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        points=[_point()],
    )

    manifest = build_partition_manifest(
        partition,
        policy=BTC_POLICY,
        covered_ranges=[
            CoveredRange(
                PARTITION_START_MS,
                PARTITION_START_MS + 30_000,
            ),
            CoveredRange(
                PARTITION_START_MS + 30_000,
                PARTITION_START_MS + 60_000,
            ),
        ],
    )

    assert manifest.schema == PARTITION_MANIFEST_SCHEMA
    assert manifest.covered_ranges == (
        CoveredRange(
            PARTITION_START_MS,
            PARTITION_START_MS + 60_000,
        ),
    )
    assert manifest.market_list_hash == BTC_POLICY.market_list_hash
    assert manifest.selected_markets == BTC_POLICY.selected_markets
    assert manifest.price_market == BTC_POLICY.price_market


def test_manifest_bytes_are_deterministic(
    tmp_path: Path,
) -> None:
    partition = write_hourly_partition(
        tmp_path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        points=[_point()],
    )

    manifest = build_partition_manifest(
        partition,
        policy=BTC_POLICY,
        covered_ranges=[
            CoveredRange(
                PARTITION_START_MS,
                PARTITION_START_MS + 60_000,
            )
        ],
    )

    assert manifest.to_bytes() == manifest.to_bytes()

    payload = json.loads(manifest.to_bytes())
    assert payload["schema"] == PARTITION_MANIFEST_SCHEMA
    assert payload["data_file"] == partition.data_file
    assert payload["data_sha256"] == partition.data_sha256
    assert payload["row_count"] == 1


def test_manifest_is_written_atomically(
    tmp_path: Path,
) -> None:
    partition = write_hourly_partition(
        tmp_path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        points=[_point()],
    )

    manifest = build_partition_manifest(
        partition,
        policy=BTC_POLICY,
        covered_ranges=[
            CoveredRange(
                PARTITION_START_MS,
                PARTITION_START_MS + 60_000,
            )
        ],
    )

    path = write_partition_manifest(
        tmp_path,
        manifest,
    )

    assert path.exists()
    assert path.read_bytes() == manifest.to_bytes()
    assert not list(path.parent.glob("*.tmp"))


def test_written_manifest_strictly_round_trips(
    tmp_path: Path,
) -> None:
    partition = write_hourly_partition(
        tmp_path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
        points=[_point()],
    )

    manifest = build_partition_manifest(
        partition,
        policy=BTC_POLICY,
        covered_ranges=[
            CoveredRange(
                PARTITION_START_MS,
                PARTITION_START_MS + 60_000,
            )
        ],
    )

    path = write_partition_manifest(
        tmp_path,
        manifest,
    )

    loaded = read_partition_manifest(
        path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
    )

    assert loaded == manifest

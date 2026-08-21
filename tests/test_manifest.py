from __future__ import annotations

import json
from pathlib import Path

import pytest

from collector.manifest import (
    MARKET_COMPLETENESS_UNPROVEN,
    PARTITION_MANIFEST_SCHEMA,
    PARTITION_MANIFEST_SCHEMA_V1,
    PARTITION_MANIFEST_SCHEMA_V2,
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

    assert PARTITION_MANIFEST_SCHEMA == PARTITION_MANIFEST_SCHEMA_V2
    assert manifest.schema == PARTITION_MANIFEST_SCHEMA_V2
    assert manifest.successful_query_ranges == (
        CoveredRange(
            PARTITION_START_MS,
            PARTITION_START_MS + 60_000,
        ),
    )
    assert manifest.covered_ranges == manifest.successful_query_ranges
    assert manifest.market_completeness == MARKET_COMPLETENESS_UNPROVEN
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

    assert payload["schema"] == PARTITION_MANIFEST_SCHEMA_V2
    assert payload["data_file"] == partition.data_file
    assert payload["data_sha256"] == partition.data_sha256
    assert payload["row_count"] == 1
    assert "covered_ranges" not in payload
    assert payload["successful_query_ranges"]
    assert payload["market_completeness"] == "unproven"


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


def test_reader_accepts_legacy_v1_query_provenance(
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

    payload = json.loads(path.read_text(encoding="utf-8"))
    query_ranges = payload.pop("successful_query_ranges")
    payload.pop("market_completeness")
    payload["schema"] = PARTITION_MANIFEST_SCHEMA_V1
    payload["covered_ranges"] = query_ranges

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded = read_partition_manifest(
        path,
        policy=BTC_POLICY,
        partition_start_ms=PARTITION_START_MS,
    )

    assert loaded.schema == PARTITION_MANIFEST_SCHEMA_V1
    assert loaded.market_completeness == MARKET_COMPLETENESS_UNPROVEN
    assert loaded.successful_query_ranges == (
        CoveredRange(
            PARTITION_START_MS,
            PARTITION_START_MS + 60_000,
        ),
    )


def test_v2_reader_rejects_legacy_covered_ranges_field(
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

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["covered_ranges"] = payload["successful_query_ranges"]

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Version-2 manifest.*legacy covered_ranges",
    ):
        read_partition_manifest(
            path,
            policy=BTC_POLICY,
            partition_start_ms=PARTITION_START_MS,
        )

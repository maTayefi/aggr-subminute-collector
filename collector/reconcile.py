# collector/reconcile.py
"""Remote hourly partition reconciliation.

This module loads an existing manifest/data pair only when the manifest exists
and both objects pass strict integrity validation. A missing manifest means the
hour has no trusted published state.

Reconciliation is local preparation only. It never uploads remote objects.
Publication ordering remains the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from collector.hf_store import HuggingFaceArchiveStore
from collector.manifest import (
    CoveredRange,
    PartitionManifest,
    build_partition_manifest,
    partition_manifest_path,
    read_partition_manifest,
    write_partition_manifest,
)
from collector.market_policy import MarketPolicy
from collector.models import CanonicalMarketPoint
from collector.partition import (
    HourlyPartition,
    merge_partition_points,
    partition_data_path,
    read_hourly_partition,
    write_hourly_partition,
)


@dataclass(frozen=True, slots=True)
class ReconciledHourlyPartition:
    """One fully prepared local data/manifest publication pair."""

    partition: HourlyPartition
    manifest: PartitionManifest
    manifest_local_path: Path
    existing_manifest_found: bool
    existing_row_count: int
    incoming_row_count: int
    exact_duplicates_dropped: int


def _load_existing_remote_partition(
    store: HuggingFaceArchiveStore,
    *,
    policy: MarketPolicy,
    partition_start_ms: int,
    remote_revision: str,
) -> tuple[
    tuple[CanonicalMarketPoint, ...],
    tuple[CoveredRange, ...],
    PartitionManifest | None,
]:
    manifest_file = partition_manifest_path(
        policy.base,
        partition_start_ms,
    )

    manifest_object = store.download_optional(
        manifest_file,
        force_download=True,
        revision=remote_revision,
    )

    if manifest_object is None:
        # An orphan Parquet file can remain after a prior data-first publication
        # failed before its manifest upload. Without a manifest it advertises no
        # trusted coverage and is deliberately not merged.
        return (), (), None

    manifest = read_partition_manifest(
        manifest_object.local_path,
        policy=policy,
        partition_start_ms=partition_start_ms,
    )

    data_object = store.require_download(
        manifest.data_file,
        force_download=True,
        revision=remote_revision,
    )

    if data_object.sha256 != manifest.data_sha256:
        raise RuntimeError(
            f"Existing remote SHA-256 mismatch for "
            f"{manifest.data_file!r}: manifest={manifest.data_sha256}, "
            f"remote={data_object.sha256}"
        )

    existing_points = read_hourly_partition(
        data_object.local_path,
        policy=policy,
        partition_start_ms=partition_start_ms,
    )

    if len(existing_points) != manifest.row_count:
        raise RuntimeError(
            f"Existing remote row_count mismatch for "
            f"{manifest.data_file!r}: manifest={manifest.row_count}, "
            f"parquet={len(existing_points)}"
        )

    return (
        existing_points,
        manifest.covered_ranges,
        manifest,
    )


def reconcile_hourly_partition(
    store: HuggingFaceArchiveStore,
    staging_root: Path,
    *,
    policy: MarketPolicy,
    partition_start_ms: int,
    incoming_points: Iterable[CanonicalMarketPoint],
    fetched_covered_ranges: Iterable[CoveredRange],
    remote_revision: str,
) -> ReconciledHourlyPartition:
    """Merge one incoming hourly slice with trusted remote archive state."""

    incoming = tuple(incoming_points)

    (
        existing_points,
        existing_coverage,
        existing_manifest,
    ) = _load_existing_remote_partition(
        store,
        policy=policy,
        partition_start_ms=partition_start_ms,
        remote_revision=remote_revision,
    )

    merged, duplicate_count = merge_partition_points(
        existing_points,
        incoming,
        policy=policy,
        partition_start_ms=partition_start_ms,
    )

    partition = write_hourly_partition(
        staging_root,
        policy=policy,
        partition_start_ms=partition_start_ms,
        points=merged,
    )

    manifest = build_partition_manifest(
        partition,
        policy=policy,
        covered_ranges=(
            *existing_coverage,
            *tuple(fetched_covered_ranges),
        ),
    )

    manifest_local_path = write_partition_manifest(
        staging_root,
        manifest,
    )

    expected_data_path = partition_data_path(
        policy.base,
        partition_start_ms,
    )
    expected_manifest_path = partition_manifest_path(
        policy.base,
        partition_start_ms,
    )

    if partition.data_file != expected_data_path:
        raise RuntimeError("Prepared partition data path is inconsistent")

    if (
        manifest_local_path.relative_to(staging_root).as_posix()
        != expected_manifest_path
    ):
        raise RuntimeError("Prepared partition manifest path is inconsistent")

    return ReconciledHourlyPartition(
        partition=partition,
        manifest=manifest,
        manifest_local_path=manifest_local_path,
        existing_manifest_found=existing_manifest is not None,
        existing_row_count=len(existing_points),
        incoming_row_count=len(incoming),
        exact_duplicates_dropped=duplicate_count,
    )

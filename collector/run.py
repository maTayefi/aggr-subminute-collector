# collector/run.py
"""Rolling BTC/ETH collection preparation and safe publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from collections.abc import Mapping

from collector.aggr_client import (
    AggrHistoricalClient,
    AggrHistoricalResponse,
)
from collector.config import CollectorSettings
from collector.hf_store import HuggingFaceArchiveStore
from collector.manifest import CoveredRange
from collector.retention import plan_retention_deletions
from collector.market_policy import (
    SOURCE_INTERVAL_MS,
    MarketPolicy,
    market_policy_segments,
)
from collector.models import CanonicalMarketPoint
from collector.partition import partition_start_ms_for_point
from collector.reconcile import (
    ReconciledHourlyPartition,
    reconcile_hourly_partition,
)

_HOUR_MS = 3_600_000
_HF_COMMIT_OPERATION_LIMIT = 500
_ARCHIVE_OBJECTS_PER_HOUR = 4  # BTC+ETH, each with Parquet+manifest


@dataclass(frozen=True, slots=True)
class CollectionWindow:
    from_ms: int
    to_ms: int

    @property
    def duration_ms(self) -> int:
        return self.to_ms - self.from_ms


@dataclass(frozen=True, slots=True)
class PreparedPolicyResponse:
    """One source response owned by one exact market-policy era."""

    policy: MarketPolicy
    window: CollectionWindow
    activation_start_ms: int
    response: AggrHistoricalResponse


@dataclass(frozen=True, slots=True)
class PreparedCollectionRun:
    window: CollectionWindow
    staging_root: Path
    base_revision: str
    responses: Mapping[str, tuple[PreparedPolicyResponse, ...]]
    partitions: tuple[ReconciledHourlyPartition, ...]


@dataclass(frozen=True, slots=True)
class PlannedCollectionPublication:
    """Exact additions and retention deletions for one prepared run.

    Creating a plan is read-only with respect to the Hugging Face repository.
    It may list the immutable base revision, but it never calls create_commit().
    """

    prepared: PreparedCollectionRun
    additions: Mapping[str, Path]
    data_files: tuple[str, ...]
    manifest_files: tuple[str, ...]
    deleted_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishedCollectionRun:
    prepared: PreparedCollectionRun
    committed_revision: str
    uploaded_data_files: tuple[str, ...]
    uploaded_manifest_files: tuple[str, ...]
    deleted_files: tuple[str, ...]


def rolling_collection_window(
    settings: CollectorSettings,
    *,
    now_utc: datetime | None = None,
) -> CollectionWindow:
    """Return the completed rolling source range under ``[from, to)``."""

    now = now_utc or datetime.now(tz=timezone.utc)

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")

    now = now.astimezone(timezone.utc)

    now_ms = int(now.timestamp() * 1000)
    lag_ms = settings.source_finalization_lag_seconds * 1000

    safe_now_ms = now_ms - lag_ms
    to_ms = safe_now_ms - (safe_now_ms % SOURCE_INTERVAL_MS)

    from_ms = to_ms - settings.rolling_refetch_hours * _HOUR_MS

    if (
        from_ms <= 0
        or to_ms <= from_ms
        or from_ms % SOURCE_INTERVAL_MS != 0
        or to_ms % SOURCE_INTERVAL_MS != 0
    ):
        raise RuntimeError("Could not construct a valid rolling collection window")

    return CollectionWindow(
        from_ms=from_ms,
        to_ms=to_ms,
    )


def _intersecting_hour_starts(
    window: CollectionWindow,
) -> tuple[int, ...]:
    first = window.from_ms - (window.from_ms % _HOUR_MS)
    last = window.to_ms - SOURCE_INTERVAL_MS
    last -= last % _HOUR_MS

    return tuple(
        range(
            first,
            last + _HOUR_MS,
            _HOUR_MS,
        )
    )


def _validate_publication_addition_bound(
    window: CollectionWindow,
) -> tuple[int, ...]:
    """Reject a collection window whose additions alone cannot fit one commit."""
    hour_starts = _intersecting_hour_starts(window)
    addition_count = len(hour_starts) * _ARCHIVE_OBJECTS_PER_HOUR

    if addition_count > _HF_COMMIT_OPERATION_LIMIT:
        raise ValueError(
            "Collection window intersects "
            f"{len(hour_starts)} UTC hour partition(s), requiring "
            f"{addition_count} Parquet/manifest additions for BTC+ETH. "
            f"One Hugging Face commit permits at most "
            f"{_HF_COMMIT_OPERATION_LIMIT} operations. Split the manual "
            "collection range before any AGGR or Hugging Face I/O is attempted."
        )

    return hour_starts


def _points_by_hour(
    points: tuple[CanonicalMarketPoint, ...],
) -> dict[int, tuple[CanonicalMarketPoint, ...]]:
    grouped: dict[int, list[CanonicalMarketPoint]] = {}

    for point in points:
        hour_start = partition_start_ms_for_point(point)
        grouped.setdefault(hour_start, []).append(point)

    return {
        hour_start: tuple(
            sorted(
                values,
                key=lambda point: point.key,
            )
        )
        for hour_start, values in grouped.items()
    }


def _coverage_for_hour(
    window: CollectionWindow,
    partition_start_ms: int,
) -> tuple[CoveredRange, ...]:
    start = max(window.from_ms, partition_start_ms)
    end = min(window.to_ms, partition_start_ms + _HOUR_MS)

    if end <= start:
        return ()

    return (CoveredRange(start, end),)


def prepare_collection_run(
    settings: CollectorSettings,
    *,
    client: AggrHistoricalClient,
    store: HuggingFaceArchiveStore,
    now_utc: datetime | None = None,
    window: CollectionWindow | None = None,
) -> PreparedCollectionRun:
    """Fetch both bases, then prepare every local object before publication.

    No remote upload occurs until:

    1. both BTC and ETH requests succeed;
    2. both responses pass strict parsing;
    3. all existing remote manifest/data pairs pass validation;
    4. all hourly merges complete without conflicts;
    5. every local Parquet and manifest is written successfully.

    ``window`` always uses the collector's internal half-open millisecond
    contract. CLI inclusive user ranges are translated once before this layer.
    """

    actual_window = window or rolling_collection_window(
        settings,
        now_utc=now_utc,
    )

    if not isinstance(actual_window, CollectionWindow):
        raise TypeError("window must be CollectionWindow or None")

    if (
        actual_window.from_ms <= 0
        or actual_window.to_ms <= actual_window.from_ms
        or actual_window.from_ms % SOURCE_INTERVAL_MS != 0
        or actual_window.to_ms % SOURCE_INTERVAL_MS != 0
    ):
        raise ValueError(
            "Collection window must be a positive, non-empty, five-second "
            "aligned internal [from, to) range"
        )

    bases = ("BTC", "ETH")

    # The number of additions is known entirely from the requested UTC-hour
    # intersections. Reject impossible manual windows before AGGR fetching,
    # Hugging Face revision discovery, downloads, reconciliation, or local
    # Parquet generation.
    hour_starts = _validate_publication_addition_bound(actual_window)
    _ = hour_starts

    # Split source requests at every deliberate UTC-hour policy boundary.
    # This prevents one response and one hourly partition from mixing rows
    # produced under different requested market universes.
    responses: dict[str, tuple[PreparedPolicyResponse, ...]] = {}

    for base in bases:
        prepared_responses: list[PreparedPolicyResponse] = []

        for segment in market_policy_segments(
            base,
            from_ms=actual_window.from_ms,
            to_ms=actual_window.to_ms,
        ):
            segment_window = CollectionWindow(
                from_ms=segment.start_ms,
                to_ms=segment.end_ms,
            )

            response = client.fetch(
                policy=segment.policy,
                from_ms=segment_window.from_ms,
                to_ms=segment_window.to_ms,
                timeframe_ms=SOURCE_INTERVAL_MS,
            )

            if not isinstance(response, AggrHistoricalResponse):
                raise TypeError(
                    "AGGR client returned an invalid response object for "
                    f"{base}: {type(response).__name__}"
                )

            if int(response.request_from_ms) != int(segment_window.from_ms) or int(
                response.request_to_ms
            ) != int(segment_window.to_ms):
                raise RuntimeError(
                    f"AGGR response ownership mismatch for {base}: "
                    f"requested=[{segment_window.from_ms}, "
                    f"{segment_window.to_ms}), response=["
                    f"{response.request_from_ms}, "
                    f"{response.request_to_ms})."
                )

            prepared_responses.append(
                PreparedPolicyResponse(
                    policy=segment.policy,
                    window=segment_window,
                    activation_start_ms=segment.activation_start_ms,
                    response=response,
                )
            )

        if not prepared_responses:
            raise RuntimeError(
                f"No market-policy request segment was produced for {base}"
            )

        responses[base] = tuple(prepared_responses)

    # Every remote object used for reconciliation must come from one immutable
    # repository snapshot. The same SHA becomes create_commit(parent_commit=...).
    #
    # Capture it only after both source fetches succeed, but before any remote
    # manifest/data object is downloaded.
    base_revision = store.current_revision()

    staging_root = settings.resolved_work_dir / "staging" / "current"

    shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    prepared: list[ReconciledHourlyPartition] = []

    try:
        for base in bases:
            for prepared_response in responses[base]:
                policy = prepared_response.policy
                segment_window = prepared_response.window
                response = prepared_response.response

                grouped = _points_by_hour(response.points)

                for partition_start_ms in _intersecting_hour_starts(segment_window):
                    prepared.append(
                        reconcile_hourly_partition(
                            store,
                            staging_root,
                            policy=policy,
                            partition_start_ms=partition_start_ms,
                            incoming_points=grouped.get(
                                partition_start_ms,
                                (),
                            ),
                            fetched_covered_ranges=_coverage_for_hour(
                                segment_window,
                                partition_start_ms,
                            ),
                            remote_revision=base_revision,
                            allow_policy_era_replacement=(
                                prepared_response.activation_start_ms > 0
                                and partition_start_ms
                                >= prepared_response.activation_start_ms
                            ),
                        )
                    )

    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    prepared.sort(
        key=lambda item: (
            item.partition.partition_start_ms,
            item.partition.base,
        )
    )

    return PreparedCollectionRun(
        window=actual_window,
        staging_root=staging_root,
        base_revision=base_revision,
        responses=responses,
        partitions=tuple(prepared),
    )


def plan_collection_publication(
    store: HuggingFaceArchiveStore,
    prepared: PreparedCollectionRun,
) -> PlannedCollectionPublication:
    """Build the exact atomic mutation plan without publishing it."""

    additions: dict[str, Path] = {}
    data_paths: list[str] = []
    manifest_paths: list[str] = []

    for item in prepared.partitions:
        data_path = item.partition.data_file
        manifest_path = item.manifest_local_path.relative_to(
            prepared.staging_root
        ).as_posix()

        if data_path in additions or manifest_path in additions:
            raise RuntimeError(
                "Prepared collection contains duplicate repository paths"
            )

        additions[data_path] = item.partition.local_path
        additions[manifest_path] = item.manifest_local_path

        data_paths.append(data_path)
        manifest_paths.append(manifest_path)

    repository_files = store.list_files(
        revision=prepared.base_revision,
    )

    retention_cutoff_ms = (
        prepared.window.to_ms - store.settings.retention_days * 86_400_000
    )

    addition_count = len(additions)

    if addition_count > _HF_COMMIT_OPERATION_LIMIT:
        raise ValueError(
            "Prepared collection additions alone exceed the "
            f"{_HF_COMMIT_OPERATION_LIMIT}-operation Hugging Face commit "
            "limit. Split the manual collection window."
        )

    deletion_budget = _HF_COMMIT_OPERATION_LIMIT - addition_count

    deletions = plan_retention_deletions(
        repository_files,
        cutoff_ms=retention_cutoff_ms,
        protected_paths=additions,
        max_deletions=deletion_budget,
    )

    operation_count = addition_count + len(deletions)
    if operation_count <= 0:
        raise RuntimeError("Prepared collection publication plan is empty")

    if operation_count > _HF_COMMIT_OPERATION_LIMIT:
        raise RuntimeError(
            "Prepared collection exceeded the "
            f"{_HF_COMMIT_OPERATION_LIMIT}-operation Hugging Face commit "
            "limit after bounded retention planning."
        )

    return PlannedCollectionPublication(
        prepared=prepared,
        additions=dict(additions),
        data_files=tuple(sorted(data_paths)),
        manifest_files=tuple(sorted(manifest_paths)),
        deleted_files=tuple(sorted(deletions)),
    )


def publish_collection_run(
    store: HuggingFaceArchiveStore,
    prepared: PreparedCollectionRun,
) -> PublishedCollectionRun:
    """Publish data, manifests, and retention deletions in one HF commit."""

    plan = plan_collection_publication(
        store,
        prepared,
    )

    committed = store.commit_atomic(
        dict(plan.additions),
        plan.deleted_files,
        parent_revision=prepared.base_revision,
        commit_message=(
            "collector: publish BTC/ETH five-second archive "
            f"[{prepared.window.from_ms}, {prepared.window.to_ms})"
        ),
    )

    return PublishedCollectionRun(
        prepared=prepared,
        committed_revision=committed.committed_revision,
        uploaded_data_files=plan.data_files,
        uploaded_manifest_files=plan.manifest_files,
        deleted_files=committed.deleted_paths,
    )


def run_collection(
    settings: CollectorSettings,
    *,
    client: AggrHistoricalClient | None = None,
    store: HuggingFaceArchiveStore | None = None,
    now_utc: datetime | None = None,
    window: CollectionWindow | None = None,
) -> PublishedCollectionRun:
    """Run one complete rolling or explicit collection publication."""

    actual_client = client or AggrHistoricalClient(
        timeout_seconds=settings.request_timeout_seconds,
        max_attempts=settings.request_max_attempts,
        initial_backoff_seconds=(settings.request_initial_backoff_seconds),
    )
    actual_store = store or HuggingFaceArchiveStore(settings)

    prepared = prepare_collection_run(
        settings,
        client=actual_client,
        store=actual_store,
        now_utc=now_utc,
        window=window,
    )

    return publish_collection_run(
        actual_store,
        prepared,
    )

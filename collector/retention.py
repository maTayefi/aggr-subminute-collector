# collector/retention.py
"""Bounded retention planning for collector-owned archive objects."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from collections.abc import Iterable

from collector.market_policy import SOURCE_INTERVAL_MS
from collector.hf_store import normalize_repo_path

_HOUR_MS = 3_600_000

_ARCHIVE_OBJECT_RE = re.compile(
    r"^(?P<family>market_points|coverage)/"
    r"(?P<base>BTC|ETH)/"
    rf"{SOURCE_INTERVAL_MS}/"
    r"(?P<year>\d{4})/"
    r"(?P<month>\d{2})/"
    r"(?P<day>\d{2})/"
    r"(?P<hour>\d{2})"
    r"(?P<suffix>\.parquet|\.json)$"
)


def archive_object_partition_start_ms(path_in_repo: str) -> int | None:
    """Return an owned object's UTC-hour start, or None for unrelated paths."""

    path = normalize_repo_path(path_in_repo)
    match = _ARCHIVE_OBJECT_RE.fullmatch(path)

    if match is None:
        return None

    family = match.group("family")
    suffix = match.group("suffix")

    if family == "market_points" and suffix != ".parquet":
        return None

    if family == "coverage" and suffix != ".json":
        return None

    try:
        timestamp = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None

    return int(timestamp.timestamp() * 1000)


def plan_retention_deletions(
    repository_files: Iterable[str],
    *,
    cutoff_ms: int,
    protected_paths: Iterable[str] = (),
    max_deletions: int = 500,
) -> tuple[str, ...]:
    """Plan a bounded oldest-first collector retention deletion set.

    ``max_deletions`` is the deletion share of the enclosing atomic commit
    operation budget. Additions and replacements consume the same Hugging Face
    commit-operation limit, so retention must not independently reserve all 500
    operations.
    """

    if isinstance(cutoff_ms, bool) or not isinstance(cutoff_ms, int):
        raise ValueError("cutoff_ms must be an integer")

    if cutoff_ms <= 0:
        raise ValueError("cutoff_ms must be positive")

    if isinstance(max_deletions, bool) or not isinstance(max_deletions, int):
        raise ValueError("max_deletions must be an integer")

    if max_deletions < 0 or max_deletions > 500:
        raise ValueError("max_deletions must be inside [0, 500]")

    protected = {
        normalize_repo_path(path) for path in protected_paths if str(path or "").strip()
    }

    expired_by_partition: dict[
        tuple[int, str],
        list[str],
    ] = {}

    for raw_path in repository_files:
        if not str(raw_path or "").strip():
            continue

        path = normalize_repo_path(raw_path)

        if path in protected:
            continue

        match = _ARCHIVE_OBJECT_RE.fullmatch(path)
        if match is None:
            continue

        partition_start_ms = archive_object_partition_start_ms(path)
        if partition_start_ms is None:
            continue

        # Retain any hour which still intersects the configured retention
        # window. Delete only when the complete hour ended at/before cutoff.
        if partition_start_ms + _HOUR_MS > cutoff_ms:
            continue

        key = (
            partition_start_ms,
            str(match.group("base")),
        )
        expired_by_partition.setdefault(key, []).append(path)

    planned: list[str] = []

    for key in sorted(expired_by_partition):
        partition_paths = sorted(set(expired_by_partition[key]))

        has_manifest = any(path.startswith("coverage/") for path in partition_paths)
        has_parquet = any(path.startswith("market_points/") for path in partition_paths)

        if has_manifest and has_parquet:
            # A complete partition is one ownership unit. If the remaining
            # budget cannot hold both objects, stop before splitting the pair.
            if len(planned) + len(partition_paths) > max_deletions:
                break

            planned.extend(partition_paths)
            continue

        # Existing orphan objects may be removed singly. They are already an
        # incomplete repository pair, so pairwise budgeting cannot restore
        # atomicity for them.
        remaining = max_deletions - len(planned)
        if remaining <= 0:
            break

        planned.extend(partition_paths[:remaining])

    return tuple(planned)

# collector/retention.py
"""Bounded retention planning for collector-owned archive objects."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Iterable

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
) -> tuple[str, ...]:
    """Plan explicit deletion of collector objects whose hour ended by cutoff."""

    if isinstance(cutoff_ms, bool) or not isinstance(cutoff_ms, int):
        raise ValueError("cutoff_ms must be an integer")

    if cutoff_ms <= 0:
        raise ValueError("cutoff_ms must be positive")

    protected = {
        normalize_repo_path(path) for path in protected_paths if str(path or "").strip()
    }

    expired: set[str] = set()

    for raw_path in repository_files:
        if not str(raw_path or "").strip():
            continue

        path = normalize_repo_path(raw_path)

        if path in protected:
            continue

        partition_start_ms = archive_object_partition_start_ms(path)

        if partition_start_ms is None:
            continue

        # Retain any hour which still intersects the configured retention
        # window. Delete only when the complete hour ended at/before cutoff.
        if partition_start_ms + _HOUR_MS <= cutoff_ms:
            expired.add(path)

    if len(expired) > 500:
        raise ValueError(
            "Retention planned more than 500 explicit deletions in one run. "
            "Refusing an unexpectedly broad archive mutation."
        )

    return tuple(sorted(expired))

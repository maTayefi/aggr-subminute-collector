from __future__ import annotations

from datetime import datetime, timezone

from collector.retention import (
    archive_object_partition_start_ms,
    plan_retention_deletions,
)


def _ms(year: int, month: int, day: int, hour: int) -> int:
    return int(
        datetime(
            year,
            month,
            day,
            hour,
            tzinfo=timezone.utc,
        ).timestamp()
        * 1000
    )


def test_retention_parses_only_collector_owned_paths() -> None:
    path = "market_points/BTC/5000/2026/08/12/18.parquet"

    assert archive_object_partition_start_ms(path) == _ms(
        2026,
        8,
        12,
        18,
    )

    assert archive_object_partition_start_ms("README.md") is None
    assert (
        archive_object_partition_start_ms(
            "market_points/BTC/60000/2026/08/12/18.parquet"
        )
        is None
    )
    assert (
        archive_object_partition_start_ms("coverage/BTC/5000/2026/08/12/18.parquet")
        is None
    )


def test_retention_deletes_only_complete_expired_hours() -> None:
    files = (
        "market_points/BTC/5000/2026/08/12/17.parquet",
        "coverage/BTC/5000/2026/08/12/17.json",
        "market_points/BTC/5000/2026/08/12/18.parquet",
        "coverage/BTC/5000/2026/08/12/18.json",
        "README.md",
    )

    # Hour 17 ends at 18:00 and expires. Hour 18 still intersects retention.
    planned = plan_retention_deletions(
        files,
        cutoff_ms=_ms(2026, 8, 12, 18),
    )

    assert planned == (
        "coverage/BTC/5000/2026/08/12/17.json",
        "market_points/BTC/5000/2026/08/12/17.parquet",
    )


def test_retention_never_deletes_protected_replacement_paths() -> None:
    old = "market_points/BTC/5000/2026/08/12/17.parquet"

    planned = plan_retention_deletions(
        [old],
        cutoff_ms=_ms(2026, 8, 13, 0),
        protected_paths=[old],
    )

    assert planned == ()

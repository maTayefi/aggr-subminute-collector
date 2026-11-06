from __future__ import annotations

import pandas as pd
import pytest

from collector.cli import inclusive_collection_window


def test_inclusive_manual_range_contains_both_endpoints() -> None:
    window = inclusive_collection_window(
        "2026-08-12 13:00:00",
        "2026-08-12 13:00:10",
        tz_name="Asia/Tehran",
    )

    assert window.from_ms == int(
        pd.Timestamp(
            "2026-08-12 09:30:00",
            tz="UTC",
        ).timestamp()
        * 1000
    )

    # User [13:00:00, 13:00:10] contains three buckets:
    # 00, 05, 10. Internal end is therefore 09:30:15 UTC.
    assert window.to_ms == int(
        pd.Timestamp(
            "2026-08-12 09:30:15",
            tz="UTC",
        ).timestamp()
        * 1000
    )

    assert window.duration_ms == 15_000


def test_equal_inclusive_endpoints_select_one_bucket() -> None:
    window = inclusive_collection_window(
        "2026-08-12 13:00:05",
        "2026-08-12 13:00:05",
        tz_name="Asia/Tehran",
    )

    assert window.duration_ms == 5_000


def test_manual_range_rejects_non_five_second_end() -> None:
    with pytest.raises(ValueError, match="five-second boundary"):
        inclusive_collection_window(
            "2026-08-12 13:00:00",
            "2026-08-12 13:00:03",
            tz_name="Asia/Tehran",
        )


def test_manual_range_rejects_reversed_inclusive_range() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        inclusive_collection_window(
            "2026-08-12 13:00:10",
            "2026-08-12 13:00:05",
            tz_name="Asia/Tehran",
        )


def test_manual_range_rejects_invalid_timezone() -> None:
    with pytest.raises(ValueError, match="Invalid IANA timezone"):
        inclusive_collection_window(
            "2026-08-12 13:00:00",
            "2026-08-12 13:00:05",
            tz_name="Invalid/Timezone",
        )


def test_manual_window_fully_expired_by_retention_is_rejected(tmp_path) -> None:
    import argparse

    from pydantic import SecretStr

    from collector.cli import _window_from_args
    from collector.config import CollectorSettings

    settings = CollectorSettings(
        hf_token=SecretStr("test-token"),
        work_dir=tmp_path / "work",
    )
    args = argparse.Namespace(
        start="2020-01-01 00:00:00",
        end="2020-01-01 00:00:10",
        timezone="UTC",
    )

    with pytest.raises(ValueError, match="retention would"):
        _window_from_args(args, settings)


def test_recent_manual_window_is_accepted(tmp_path) -> None:
    import argparse
    from datetime import datetime, timedelta, timezone

    from pydantic import SecretStr

    from collector.cli import _window_from_args
    from collector.config import CollectorSettings

    settings = CollectorSettings(
        hf_token=SecretStr("test-token"),
        work_dir=tmp_path / "work",
    )

    start = datetime.now(timezone.utc) - timedelta(hours=2)
    start = start.replace(microsecond=0, second=start.second - start.second % 5)
    end = start + timedelta(seconds=10)

    args = argparse.Namespace(
        start=start.strftime("%Y-%m-%d %H:%M:%S"),
        end=end.strftime("%Y-%m-%d %H:%M:%S"),
        timezone="UTC",
    )

    window = _window_from_args(args, settings)

    assert window is not None
    assert window.to_ms - window.from_ms == 15_000

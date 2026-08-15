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

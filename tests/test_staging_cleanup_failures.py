from __future__ import annotations

from pathlib import Path
from unittest import mock

import pyarrow as pa
import pytest

from collector.manifest import (
    PartitionManifest,
    partition_manifest_path,
    write_partition_manifest,
)
from collector.partition import _atomic_parquet_write


@pytest.mark.parametrize(
    "exception_type",
    [OSError, KeyboardInterrupt],
)
def test_parquet_cleanup_failure_preserves_primary_exception(
    tmp_path: Path,
    exception_type: type[BaseException],
) -> None:
    target = tmp_path / "hour.parquet"
    original_bytes = b"existing staging object"
    target.write_bytes(original_bytes)

    primary = exception_type("synthetic Parquet write failure")
    cleanup = PermissionError("synthetic Parquet cleanup failure")

    with (
        mock.patch(
            "collector.partition.pq.write_table",
            side_effect=primary,
        ),
        mock.patch.object(
            Path,
            "unlink",
            side_effect=cleanup,
        ) as unlink,
    ):
        with pytest.raises(exception_type) as caught:
            _atomic_parquet_write(
                target,
                pa.table({"value": [1]}),
            )

    assert caught.value is primary
    unlink.assert_called_once_with(missing_ok=True)
    assert target.read_bytes() == original_bytes
    assert any(
        "synthetic Parquet cleanup failure" in note
        for note in getattr(primary, "__notes__", ())
    )


@pytest.mark.parametrize(
    "exception_type",
    [OSError, KeyboardInterrupt],
)
def test_manifest_cleanup_failure_preserves_primary_exception(
    tmp_path: Path,
    exception_type: type[BaseException],
) -> None:
    hour_start_ms = (1_700_000_000 // 3_600 * 3_600) * 1000

    # This test isolates the atomic writer. Manifest schema validation
    # belongs to the existing manifest construction/reader tests.
    manifest = mock.Mock(spec=PartitionManifest)
    manifest.base = "BTC"
    manifest.partition_start_ms = hour_start_ms
    manifest.to_bytes.return_value = b'{"stage":"test"}\n'

    target = tmp_path / partition_manifest_path(
        "BTC",
        hour_start_ms,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    original_bytes = b"existing staging manifest"
    target.write_bytes(original_bytes)

    primary = exception_type("synthetic manifest replace failure")
    cleanup = PermissionError("synthetic manifest cleanup failure")

    with (
        mock.patch(
            "collector.manifest.os.replace",
            side_effect=primary,
        ),
        mock.patch.object(
            Path,
            "unlink",
            side_effect=cleanup,
        ) as unlink,
    ):
        with pytest.raises(exception_type) as caught:
            write_partition_manifest(
                tmp_path,
                manifest,
            )

    assert caught.value is primary
    unlink.assert_called_once_with(missing_ok=True)
    assert target.read_bytes() == original_bytes
    assert any(
        "synthetic manifest cleanup failure" in note
        for note in getattr(primary, "__notes__", ())
    )

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


@pytest.mark.parametrize(
    "cleanup_fails",
    [False, True],
)
def test_parquet_initial_close_failure_reaches_temporary_cleanup(
    tmp_path: Path,
    cleanup_fails: bool,
) -> None:
    import os
    import tempfile

    target = tmp_path / "hour.parquet"
    original_bytes = b"existing staging object"
    target.write_bytes(original_bytes)

    primary = OSError("Parquet initial close failed")
    cleanup_error = PermissionError("Parquet initial-close temporary cleanup failed")

    real_close = os.close
    real_mkstemp = tempfile.mkstemp
    real_unlink = Path.unlink

    created_paths: list[Path] = []
    close_calls: list[int] = []
    cleanup_calls: list[Path] = []

    def record_mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        created_paths.append(Path(name))
        return descriptor, name

    def close_then_report_failure(descriptor: int) -> None:
        close_calls.append(descriptor)

        # Release the real descriptor before reporting the synthetic
        # close error. Retrying this descriptor number would be unsafe.
        real_close(descriptor)
        raise primary

    def unlink(path: Path, *args, **kwargs):
        if path in created_paths and kwargs.get("missing_ok") is True:
            cleanup_calls.append(path)

            if cleanup_fails:
                raise cleanup_error

        return real_unlink(path, *args, **kwargs)

    with (
        mock.patch(
            "collector.partition.tempfile.mkstemp",
            side_effect=record_mkstemp,
        ),
        mock.patch(
            "collector.partition.os.close",
            side_effect=close_then_report_failure,
        ),
        mock.patch.object(
            Path,
            "unlink",
            new=unlink,
        ),
        mock.patch(
            "collector.partition.pq.write_table",
        ) as write_table,
        mock.patch(
            "collector.partition.os.replace",
        ) as replace_file,
    ):
        with pytest.raises(OSError) as caught:
            _atomic_parquet_write(
                target,
                pa.table({"value": [1]}),
            )

    assert caught.value is primary
    assert len(created_paths) == 1
    assert len(close_calls) == 1
    assert cleanup_calls == created_paths

    write_table.assert_not_called()
    replace_file.assert_not_called()

    assert target.read_bytes() == original_bytes

    if cleanup_fails:
        assert created_paths[0].is_file()
        assert any(
            str(cleanup_error) in note for note in getattr(primary, "__notes__", ())
        )
    else:
        assert not created_paths[0].exists()

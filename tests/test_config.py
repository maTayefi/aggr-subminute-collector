from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from collector.config import (
    DEFAULT_HF_REPO_ID,
    CollectorSettings,
)


def test_default_archive_identity_is_frozen() -> None:
    settings = CollectorSettings(
        hf_token=SecretStr("test-token"),
    )

    assert DEFAULT_HF_REPO_ID == "maTayefi/smp-aggr-subminute"
    assert settings.hf_repo_id == DEFAULT_HF_REPO_ID
    assert settings.hf_repo_type == "dataset"
    assert settings.hf_revision == "main"


def test_token_is_not_exposed_in_repr() -> None:
    settings = CollectorSettings(
        hf_token=SecretStr("sensitive-test-token"),
    )

    rendered = repr(settings)

    assert "sensitive-test-token" not in rendered


def test_retention_cannot_be_below_ninety_days() -> None:
    with pytest.raises(ValueError, match="at least 90"):
        CollectorSettings(
            hf_token=SecretStr("test-token"),
            retention_days=89,
        )


def test_work_directory_resolves_absolutely(
    tmp_path: Path,
) -> None:
    settings = CollectorSettings(
        hf_token=SecretStr("test-token"),
        work_dir=tmp_path / "collector-work",
    )

    assert settings.resolved_work_dir.is_absolute()
    assert settings.resolved_work_dir == (tmp_path / "collector-work").resolve()


def test_source_finalization_lag_cannot_be_below_one_minute() -> None:
    with pytest.raises(ValueError, match="at least 60"):
        CollectorSettings(
            hf_token=SecretStr("test-token"),
            source_finalization_lag_seconds=55,
        )


def test_source_finalization_lag_must_align_to_five_seconds() -> None:
    with pytest.raises(ValueError, match="align to five seconds"):
        CollectorSettings(
            hf_token=SecretStr("test-token"),
            source_finalization_lag_seconds=61,
        )

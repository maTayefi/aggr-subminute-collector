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


@pytest.mark.parametrize(
    "field",
    [
        "rolling_refetch_hours",
        "retention_days",
        "source_finalization_lag_seconds",
        "request_max_attempts",
        "hf_read_max_attempts",
    ],
)
def test_integer_configuration_rejects_boolean_and_fractional_values(
    field: str,
) -> None:
    for value in (True, False, 1.5, "1.5"):
        with pytest.raises(
            ValueError,
            match="positive integer",
        ):
            CollectorSettings(
                hf_token=SecretStr("test-token"),
                **{field: value},
            )


@pytest.mark.parametrize(
    "field",
    [
        "request_timeout_seconds",
        "request_initial_backoff_seconds",
        "hf_read_initial_backoff_seconds",
    ],
)
def test_float_configuration_rejects_nonfinite_and_boolean_values(
    field: str,
) -> None:
    for value in (True, False, float("nan"), float("inf"), -1.0, 0.0):
        with pytest.raises(
            ValueError,
            match="positive finite|finite and > 0",
        ):
            CollectorSettings(
                hf_token=SecretStr("test-token"),
                **{field: value},
            )


def test_hf_read_retry_configuration_accepts_environment_style_strings() -> None:
    settings = CollectorSettings(
        hf_token=SecretStr("test-token"),
        hf_read_max_attempts="3",
        hf_read_initial_backoff_seconds="0.25",
    )

    assert settings.hf_read_max_attempts == 3
    assert settings.hf_read_initial_backoff_seconds == pytest.approx(0.25)


def test_default_rolling_refetch_window_is_seven_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "AGGR_COLLECTOR_ROLLING_REFETCH_HOURS",
        raising=False,
    )

    settings = CollectorSettings(
        _env_file=None,
        hf_token=SecretStr("test-token"),
    )

    assert settings.rolling_refetch_hours == 7

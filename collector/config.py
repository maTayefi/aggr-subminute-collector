# collector/config.py
"""Environment-backed configuration for the autonomous collector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_HF_REPO_ID = "maTayefi/smp-aggr-subminute"


class CollectorSettings(BaseSettings):
    """Runtime configuration for GitHub Actions and local dry runs."""

    model_config = SettingsConfigDict(
        env_prefix="AGGR_COLLECTOR_",
        case_sensitive=False,
        env_file=".env",
        extra="ignore",
    )

    hf_repo_id: str = DEFAULT_HF_REPO_ID
    hf_repo_type: str = "dataset"
    hf_revision: str = "main"

    # Use a dedicated Hugging Face write token in GitHub Actions.
    hf_token: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        repr=False,
    )

    work_dir: Path = Path(".collector_work")

    rolling_refetch_hours: int = 7
    retention_days: int = 90

    # aggr-server can append in-memory pending bars when the request reaches
    # into its recent resampling window. Version 1 therefore collects only
    # boundaries at least one minute behind the current clock.
    source_finalization_lag_seconds: int = 60

    request_timeout_seconds: float = 30.0
    request_max_attempts: int = 4
    request_initial_backoff_seconds: float = 1.0

    # Safe, idempotent Hugging Face read operations:
    #   - repository revision discovery;
    #   - repository file listing;
    #   - immutable-revision object download.
    #
    # Write operations such as create_commit() are deliberately excluded from
    # automatic retry because a response timeout can leave commit success
    # uncertain.
    hf_read_max_attempts: int = 4
    hf_read_initial_backoff_seconds: float = 1.0

    @field_validator(
        "hf_repo_id",
        "hf_repo_type",
        "hf_revision",
    )
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        output = str(value or "").strip()
        if not output:
            raise ValueError("must be non-empty")
        return output

    @field_validator(
        "rolling_refetch_hours",
        "retention_days",
        "source_finalization_lag_seconds",
        "request_max_attempts",
        "hf_read_max_attempts",
        mode="before",
    )
    @classmethod
    def _positive_integer(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("must be a positive integer, not bool")

        if isinstance(value, int):
            output = value

        elif isinstance(value, str):
            text = value.strip()

            if not text or not text.isdigit():
                raise ValueError(f"must be a positive integer; got {value!r}")

            output = int(text)

        else:
            raise ValueError(f"must be a positive integer; got {value!r}")

        if output <= 0:
            raise ValueError("must be > 0")

        return output

    @field_validator(
        "request_timeout_seconds",
        "request_initial_backoff_seconds",
        "hf_read_initial_backoff_seconds",
        mode="before",
    )
    @classmethod
    def _positive_float(cls, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("must be a positive finite number, not bool")

        try:
            output = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"must be a positive finite number; got {value!r}"
            ) from exc

        if not math.isfinite(output) or output <= 0.0:
            raise ValueError("must be finite and > 0")

        return output

    @model_validator(mode="after")
    def _validate_collector_contract(self) -> CollectorSettings:
        if self.hf_repo_type != "dataset":
            raise ValueError("hf_repo_type must be 'dataset' for the private archive")

        if self.rolling_refetch_hours < 3:
            raise ValueError("rolling_refetch_hours must be at least 3 in version 1")

        if self.retention_days < 90:
            raise ValueError("retention_days must be at least 90")

        if self.source_finalization_lag_seconds < 60:
            raise ValueError(
                "source_finalization_lag_seconds must be at least 60 "
                "seconds so recent pending aggr bars are not archived as final"
            )

        if self.source_finalization_lag_seconds % 5 != 0:
            raise ValueError(
                "source_finalization_lag_seconds must align to five seconds"
            )

        return self

    @property
    def resolved_work_dir(self) -> Path:
        return self.work_dir.expanduser().resolve()

    def require_hf_token(self) -> str:
        token = self.hf_token.get_secret_value().strip()

        if not token:
            raise RuntimeError(
                "A private Hugging Face write token is required. Set "
                "AGGR_COLLECTOR_HF_TOKEN locally or the HF_TOKEN GitHub "
                "Actions secret mapped to that environment variable."
            )

        return token

# collector/hf_store.py
"""Verified private Hugging Face Dataset object storage.

This module intentionally knows nothing about aggr.trade collection windows.
It provides a small fail-closed remote-object boundary used by reconciliation
and collection orchestration.

Production publication is one atomic repository transition:

1. reconcile against one immutable parent revision;
2. prepare every Parquet file and manifest locally;
3. construct all additions/replacements and bounded retention deletions;
4. call ``HfApi.create_commit()`` once with ``parent_commit``;
5. verify every addition and deletion against the immutable returned revision.

Parquet objects, manifests, and retention deletions therefore become visible in
the same repository commit. The older single-file upload/delete methods remain
available only for isolated tooling and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import time
from typing import Callable, Iterable, TypeVar

import httpx
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

from collector.config import CollectorSettings

_T = TypeVar("_T")

_TRANSIENT_HF_HTTP_STATUS_CODES = {
    408,  # Request Timeout
    425,  # Too Early
    429,  # Too Many Requests
}


def _exception_chain(
    exc: BaseException,
) -> tuple[BaseException, ...]:
    """Return a bounded cause/context chain without cycles."""

    output: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and len(output) < 16:
        identity = id(current)
        if identity in seen:
            break

        seen.add(identity)
        output.append(current)

        cause = current.__cause__
        if cause is not None:
            current = cause
            continue

        current = current.__context__

    return tuple(output)


def _hf_read_error_is_transient(
    exc: BaseException,
) -> bool:
    """Classify retryable Hugging Face read failures.

    ``hf_hub_download`` can wrap an httpx timeout in ``ValueError`` such as:

        ValueError: Force download failed due to the above error.

    Therefore classification must inspect the complete exception chain rather
    than only the outer exception type.

    Confirmed remote object absence and repository/authentication failures are
    not transient read retries.
    """

    for item in _exception_chain(exc):
        if isinstance(item, LocalEntryNotFoundError):
            # This means the requested object could not be obtained and no
            # usable cache entry was available. It is infrastructure state,
            # not confirmed remote absence, and may recover on retry.
            return True

        if isinstance(item, RepositoryNotFoundError):
            return False

        if isinstance(item, EntryNotFoundError):
            # Confirmed missing remote object. download_optional() owns this
            # semantic result and must receive the original exception.
            return False

        if isinstance(
            item,
            (
                httpx.TimeoutException,
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ),
        ):
            return True

        if isinstance(item, HfHubHTTPError):
            response = getattr(item, "response", None)
            status_code = getattr(response, "status_code", None)

            try:
                status = int(status_code)
            except TypeError, ValueError:
                status = 0

            if status in _TRANSIENT_HF_HTTP_STATUS_CODES or 500 <= status <= 599:
                return True

            return False

    return False


@dataclass(frozen=True, slots=True)
class RemoteObject:
    """One verified remote archive object."""

    path_in_repo: str
    local_path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AtomicCommitResult:
    """One verified atomic Hugging Face Dataset repository transition."""

    parent_revision: str
    committed_revision: str
    added_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]


def normalize_repo_path(value: str) -> str:
    """Return one safe repository-relative POSIX path."""

    path = str(value or "").strip().replace("\\", "/")

    while path.startswith("./"):
        path = path[2:]

    path = path.lstrip("/")
    parts = tuple(part for part in path.split("/") if part)

    if (
        not parts
        or any(part in {".", ".."} for part in parts)
        or any("\x00" in part for part in parts)
    ):
        raise ValueError(f"Unsafe or empty Hugging Face repository path: {value!r}")

    return "/".join(parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


class HuggingFaceArchiveStore:
    """Small verified object-store facade for one private Dataset repository."""

    def __init__(
        self,
        settings: CollectorSettings,
        *,
        api: HfApi | None = None,
    ) -> None:
        if not isinstance(settings, CollectorSettings):
            raise TypeError("settings must be CollectorSettings")

        self.settings = settings
        self.token = settings.require_hf_token()
        self.api = api or HfApi(token=self.token)

        self.cache_dir = settings.resolved_work_dir / "hf_cache"
        self.download_dir = settings.resolved_work_dir / "downloads"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _retry_hf_read(
        self,
        operation_label: str,
        operation: Callable[[], _T],
    ) -> _T:
        """Retry one safe/idempotent Hugging Face read operation.

        This helper must not be used for create_commit(), upload, or deletion.
        A write can succeed remotely while its response times out, making blind
        replay unsafe.
        """

        label = str(operation_label or "").strip() or "Hugging Face read"
        attempts = int(self.settings.hf_read_max_attempts)
        initial_backoff = float(self.settings.hf_read_initial_backoff_seconds)

        last_error: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                return operation()

            except Exception as exc:  # noqa: BLE001
                last_error = exc

                if not _hf_read_error_is_transient(exc):
                    raise

                if attempt >= attempts:
                    break

                delay = initial_backoff * (2 ** (attempt - 1))

                time.sleep(delay)

        assert last_error is not None

        raise RuntimeError(
            f"{label} failed after {attempts} attempt(s) because of "
            "transient Hugging Face transport/cache-resolution errors."
        ) from last_error

    def _download_from_hub(
        self,
        path_in_repo: str,
        *,
        force_download: bool,
        revision: str | None = None,
    ) -> Path:
        remote_path = normalize_repo_path(path_in_repo)
        resolved_revision = str(revision or "").strip() or self.settings.hf_revision

        try:
            downloaded = self._retry_hf_read(
                f"Hugging Face download for {remote_path!r}",
                lambda: hf_hub_download(
                    repo_id=self.settings.hf_repo_id,
                    filename=remote_path,
                    repo_type=self.settings.hf_repo_type,
                    revision=resolved_revision,
                    token=self.token,
                    cache_dir=str(self.cache_dir),
                    force_download=bool(force_download),
                ),
            )
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                "The collector could not obtain the Hugging Face object "
                f"{remote_path!r}, and no usable local cached copy was "
                "available. This is an infrastructure/cache-resolution "
                "failure, not proof that the remote archive object is absent."
            ) from exc
        except EntryNotFoundError:
            # A confirmed remote object absence is handled by
            # download_optional(). Keep it distinct from local/cache failure.
            raise
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                "The private Hugging Face Dataset repository could not be "
                "opened. Verify AGGR_COLLECTOR_HF_REPO_ID and write-token "
                "permissions."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Hugging Face download failed for {remote_path!r}: {exc}"
            ) from exc

        path = Path(downloaded)

        if not path.is_file():
            raise RuntimeError(
                f"Hugging Face returned a non-file path for " f"{remote_path!r}: {path}"
            )

        return path

    def download_optional(
        self,
        path_in_repo: str,
        *,
        force_download: bool = True,
        revision: str | None = None,
    ) -> RemoteObject | None:
        """Download one object, returning None only when it does not exist."""

        remote_path = normalize_repo_path(path_in_repo)

        try:
            cached_path = self._download_from_hub(
                remote_path,
                force_download=force_download,
                revision=revision,
            )
        except EntryNotFoundError:
            return None

        # Copy out of the revision-addressed HF cache into a deterministic
        # collector work path. Later code can safely replace or inspect this
        # file without mutating the Hugging Face cache.
        local_path = self.download_dir / remote_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = local_path.with_name(f".{local_path.name}.download.tmp")
        temp_path.unlink(missing_ok=True)

        try:
            shutil.copyfile(cached_path, temp_path)
            temp_path.replace(local_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

        return RemoteObject(
            path_in_repo=remote_path,
            local_path=local_path,
            size_bytes=local_path.stat().st_size,
            sha256=sha256_file(local_path),
        )

    def require_download(
        self,
        path_in_repo: str,
        *,
        force_download: bool = True,
        revision: str | None = None,
    ) -> RemoteObject:
        downloaded = self.download_optional(
            path_in_repo,
            force_download=force_download,
            revision=revision,
        )

        if downloaded is None:
            raise FileNotFoundError(
                f"Required remote archive object does not exist: "
                f"{normalize_repo_path(path_in_repo)!r}"
            )

        return downloaded

    def upload_verified(
        self,
        local_path: Path,
        path_in_repo: str,
        *,
        commit_message: str,
    ) -> RemoteObject:
        """Upload one file and verify exact remote bytes by SHA-256."""

        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(f"Upload source file does not exist: {source}")

        remote_path = normalize_repo_path(path_in_repo)
        expected_size = source.stat().st_size
        expected_sha256 = sha256_file(source)

        message = str(commit_message or "").strip()
        if not message:
            raise ValueError("commit_message must be non-empty")

        try:
            self.api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=remote_path,
                repo_id=self.settings.hf_repo_id,
                repo_type=self.settings.hf_repo_type,
                revision=self.settings.hf_revision,
                token=self.token,
                commit_message=message,
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                "The private Hugging Face Dataset repository could not be "
                "opened for upload. Verify repository identity and write-token "
                "permissions."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Hugging Face upload failed for {remote_path!r}: {exc}"
            ) from exc

        remote = self.require_download(
            remote_path,
            force_download=True,
        )

        if remote.size_bytes != expected_size:
            raise RuntimeError(
                f"Remote size verification failed for {remote_path!r}: "
                f"local={expected_size}, remote={remote.size_bytes}"
            )

        if remote.sha256 != expected_sha256:
            raise RuntimeError(
                f"Remote SHA-256 verification failed for "
                f"{remote_path!r}: local={expected_sha256}, "
                f"remote={remote.sha256}"
            )

        return remote

    def list_files(
        self,
        *,
        prefix: str = "",
        revision: str | None = None,
    ) -> tuple[str, ...]:
        """Return deterministic repository paths, optionally under a prefix."""

        normalized_prefix = ""
        if str(prefix or "").strip():
            normalized_prefix = normalize_repo_path(prefix).rstrip("/") + "/"

        resolved_revision = str(revision or "").strip() or self.settings.hf_revision

        try:
            files = self._retry_hf_read(
                "Hugging Face repository file listing",
                lambda: self.api.list_repo_files(
                    repo_id=self.settings.hf_repo_id,
                    repo_type=self.settings.hf_repo_type,
                    revision=resolved_revision,
                    token=self.token,
                ),
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                "The private Hugging Face Dataset repository could not be "
                "listed. Verify repository identity and token permissions."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Hugging Face repository listing failed: {exc}"
            ) from exc

        output = {
            normalize_repo_path(path) for path in files if str(path or "").strip()
        }

        if normalized_prefix:
            output = {path for path in output if path.startswith(normalized_prefix)}

        return tuple(sorted(output))

    def current_revision(self) -> str:
        """Return the exact current repository commit SHA."""

        try:
            info = self._retry_hf_read(
                "Hugging Face repository revision discovery",
                lambda: self.api.repo_info(
                    repo_id=self.settings.hf_repo_id,
                    repo_type=self.settings.hf_repo_type,
                    revision=self.settings.hf_revision,
                    token=self.token,
                ),
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                "The private Hugging Face Dataset repository could not be "
                "opened for revision discovery."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Hugging Face repository revision lookup failed: {exc}"
            ) from exc

        revision = str(getattr(info, "sha", "") or "").strip()

        if not revision:
            raise RuntimeError(
                "Hugging Face repo_info returned no repository revision SHA."
            )

        return revision

    def commit_atomic(
        self,
        additions: dict[str, Path],
        deletions: Iterable[str],
        *,
        parent_revision: str,
        commit_message: str,
    ) -> AtomicCommitResult:
        """Atomically publish additions/replacements and explicit deletions.

        All Parquet files, manifests, and retention deletions become visible in
        one repository revision. ``parent_revision`` prevents publication
        against a repository state different from the one reconciled locally.
        """

        parent = str(parent_revision or "").strip()
        if not parent:
            raise ValueError("parent_revision must be non-empty")

        message = str(commit_message or "").strip()
        if not message:
            raise ValueError("commit_message must be non-empty")

        normalized_additions: dict[str, Path] = {}

        for raw_path, raw_source in additions.items():
            remote_path = normalize_repo_path(raw_path)
            source = Path(raw_source)

            if not source.is_file():
                raise FileNotFoundError(
                    f"Atomic commit source file does not exist: {source}"
                )

            if remote_path in normalized_additions:
                raise ValueError(
                    f"Atomic commit contains duplicate add path: {remote_path!r}"
                )

            normalized_additions[remote_path] = source

        normalized_deletions = {
            normalize_repo_path(path) for path in deletions if str(path or "").strip()
        }

        # An add/replace owns the final state for its path and must never also
        # appear as a delete operation in the same commit.
        normalized_deletions.difference_update(normalized_additions)

        operation_count = len(normalized_additions) + len(normalized_deletions)

        if operation_count <= 0:
            raise ValueError("Atomic commit contains no operations")

        if operation_count > 500:
            raise ValueError(
                "One collector atomic commit may contain at most 500 explicit "
                "add/delete operations"
            )

        expected: dict[str, tuple[int, str]] = {
            remote_path: (
                source.stat().st_size,
                sha256_file(source),
            )
            for remote_path, source in normalized_additions.items()
        }

        operations = [
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=str(normalized_additions[remote_path]),
            )
            for remote_path in sorted(normalized_additions)
        ]

        operations.extend(
            CommitOperationDelete(path_in_repo=remote_path)
            for remote_path in sorted(normalized_deletions)
        )

        # Do not pass create_commit() through _retry_hf_read(). If Hugging Face
        # commits the request but the response times out, publication success is
        # uncertain. Blind replay can create a duplicate/stale-parent attempt.
        # Fail closed and let the next scheduled run reconcile against the
        # repository's actual current revision.
        try:
            commit_info = self.api.create_commit(
                repo_id=self.settings.hf_repo_id,
                repo_type=self.settings.hf_repo_type,
                revision=self.settings.hf_revision,
                token=self.token,
                parent_commit=parent,
                operations=operations,
                commit_message=message,
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                "The private Hugging Face Dataset repository could not be "
                "opened for atomic publication."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(
                "Atomic Hugging Face publication failed. The repository may "
                "have changed after reconciliation; rerun the collector rather "
                f"than publishing against stale state: {exc}"
            ) from exc

        committed_revision = str(
            getattr(commit_info, "oid", "")
            or getattr(commit_info, "commit_id", "")
            or ""
        ).strip()

        if not committed_revision:
            raise RuntimeError(
                "Hugging Face create_commit returned no committed revision SHA"
            )

        # Verify additions against the immutable returned commit, not against a
        # branch name which could move again after publication.
        for remote_path in sorted(normalized_additions):
            remote = self.require_download(
                remote_path,
                force_download=True,
                revision=committed_revision,
            )
            expected_size, expected_sha256 = expected[remote_path]

            if remote.size_bytes != expected_size:
                raise RuntimeError(
                    f"Atomic commit size verification failed for "
                    f"{remote_path!r}: local={expected_size}, "
                    f"remote={remote.size_bytes}"
                )

            if remote.sha256 != expected_sha256:
                raise RuntimeError(
                    f"Atomic commit SHA-256 verification failed for "
                    f"{remote_path!r}: local={expected_sha256}, "
                    f"remote={remote.sha256}"
                )

        if normalized_deletions:
            remaining = set(self.list_files(revision=committed_revision))
            survivors = sorted(normalized_deletions & remaining)

            if survivors:
                raise RuntimeError(
                    "Atomic retention verification failed; deleted object(s) "
                    f"still exist in committed revision: {survivors[:20]}"
                )

        return AtomicCommitResult(
            parent_revision=parent,
            committed_revision=committed_revision,
            added_paths=tuple(sorted(normalized_additions)),
            deleted_paths=tuple(sorted(normalized_deletions)),
        )

    def delete_files(
        self,
        paths_in_repo: Iterable[str],
        *,
        commit_message: str,
    ) -> tuple[str, ...]:
        """Delete a bounded explicit object set and verify disappearance."""

        normalized = tuple(
            sorted(
                {
                    normalize_repo_path(path)
                    for path in paths_in_repo
                    if str(path or "").strip()
                }
            )
        )

        if not normalized:
            return ()

        # Prevent one bug from turning retention into an unbounded repository
        # deletion operation.
        if len(normalized) > 500:
            raise ValueError(
                "One Hugging Face deletion commit may contain at most "
                "500 explicit paths"
            )

        message = str(commit_message or "").strip()
        if not message:
            raise ValueError("commit_message must be non-empty")

        try:
            self.api.delete_files(
                delete_patterns=list(normalized),
                repo_id=self.settings.hf_repo_id,
                repo_type=self.settings.hf_repo_type,
                revision=self.settings.hf_revision,
                token=self.token,
                commit_message=message,
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                "The private Hugging Face Dataset repository could not be "
                "opened for deletion."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(f"Hugging Face deletion failed: {exc}") from exc

        remaining = set(self.list_files())

        survivors = [path for path in normalized if path in remaining]

        if survivors:
            raise RuntimeError(
                "Hugging Face deletion verification failed; object(s) still "
                f"exist: {survivors[:20]}"
            )

        return normalized

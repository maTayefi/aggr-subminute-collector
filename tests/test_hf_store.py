from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
)
from huggingface_hub.utils import EntryNotFoundError
from pydantic import SecretStr

from collector.config import CollectorSettings
from collector.hf_store import (
    HuggingFaceArchiveStore,
    normalize_repo_path,
    sha256_file,
)


class _FakeApi:
    def __init__(self, remote_root: Path) -> None:
        self.remote_root = remote_root
        self.upload_calls: list[str] = []
        self.delete_calls: list[tuple[str, ...]] = []
        self.create_commit_calls: list[dict] = []
        self.base_revision = "fake-base-revision"
        self.committed_revision = "fake-committed-revision"

    def repo_info(
        self,
        *,
        repo_id,
        repo_type,
        revision,
        token,
    ):
        _ = repo_id, repo_type, revision, token
        return SimpleNamespace(sha=self.base_revision)

    def create_commit(
        self,
        *,
        repo_id,
        repo_type,
        revision,
        token,
        parent_commit,
        operations,
        commit_message,
    ):
        _ = repo_id, repo_type, revision, token

        operations = list(operations)

        self.create_commit_calls.append(
            {
                "parent_commit": parent_commit,
                "commit_message": commit_message,
                "operations": operations,
            }
        )

        for operation in operations:
            path_in_repo = str(operation.path_in_repo)

            if isinstance(operation, CommitOperationAdd):
                target = self.remote_root / path_in_repo
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    operation.path_or_fileobj,
                    target,
                )

            elif isinstance(operation, CommitOperationDelete):
                (self.remote_root / path_in_repo).unlink(missing_ok=True)

            else:
                raise AssertionError(f"Unexpected fake commit operation: {operation!r}")

        return SimpleNamespace(oid=self.committed_revision)

    def upload_file(
        self,
        *,
        path_or_fileobj,
        path_in_repo,
        repo_id,
        repo_type,
        revision,
        token,
        commit_message,
    ):
        _ = (
            repo_id,
            repo_type,
            revision,
            token,
            commit_message,
        )

        target = self.remote_root / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path_or_fileobj, target)
        self.upload_calls.append(path_in_repo)
        return "fake-upload"

    def list_repo_files(
        self,
        *,
        repo_id,
        repo_type,
        revision,
        token,
    ):
        _ = repo_id, repo_type, revision, token

        return [
            path.relative_to(self.remote_root).as_posix()
            for path in self.remote_root.rglob("*")
            if path.is_file()
        ]

    def delete_files(
        self,
        *,
        delete_patterns,
        repo_id,
        repo_type,
        revision,
        token,
        commit_message,
    ):
        _ = (
            repo_id,
            repo_type,
            revision,
            token,
            commit_message,
        )

        normalized = tuple(sorted(delete_patterns))
        self.delete_calls.append(normalized)

        for relative in normalized:
            (self.remote_root / relative).unlink(missing_ok=True)

        return "fake-delete"


def _settings(tmp_path: Path) -> CollectorSettings:
    return CollectorSettings(
        hf_repo_id="maTayefi/smp-aggr-subminute",
        hf_repo_type="dataset",
        hf_revision="main",
        hf_token=SecretStr("test-write-token"),
        work_dir=tmp_path / "work",
    )


def _install_fake_downloader(
    monkeypatch,
    remote_root: Path,
) -> None:
    def _download(
        *,
        repo_id,
        filename,
        repo_type,
        revision,
        token,
        cache_dir,
        force_download,
    ):
        _ = (
            repo_id,
            repo_type,
            revision,
            token,
            cache_dir,
            force_download,
        )

        path = remote_root / filename

        if not path.is_file():
            raise EntryNotFoundError(f"missing fake remote object: {filename}")

        return str(path)

    monkeypatch.setattr(
        "collector.hf_store.hf_hub_download",
        _download,
    )


def test_normalize_repo_path_rejects_unsafe_values() -> None:
    assert (
        normalize_repo_path(r"\market_points\BTC\5000\2026\08\12\18.parquet")
        == "market_points/BTC/5000/2026/08/12/18.parquet"
    )

    with pytest.raises(ValueError, match="Unsafe"):
        normalize_repo_path("../secret.txt")

    with pytest.raises(ValueError, match="Unsafe"):
        normalize_repo_path("")


def test_private_token_is_required_before_store_creation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"hf_token": SecretStr("")},
    )

    with pytest.raises(RuntimeError, match="write token"):
        HuggingFaceArchiveStore(
            settings,
            api=_FakeApi(tmp_path / "remote"),
        )


def test_missing_remote_object_returns_none(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()

    _install_fake_downloader(
        monkeypatch,
        remote_root,
    )

    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=_FakeApi(remote_root),
    )

    assert store.download_optional("coverage/BTC/5000/2026/08/12/18.json") is None


def test_upload_is_verified_by_remote_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()

    _install_fake_downloader(
        monkeypatch,
        remote_root,
    )

    api = _FakeApi(remote_root)
    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=api,
    )

    source = tmp_path / "source.parquet"
    source.write_bytes(b"deterministic parquet fixture")

    remote_path = "market_points/BTC/5000/2026/08/12/18.parquet"

    uploaded = store.upload_verified(
        source,
        remote_path,
        commit_message="test: upload BTC data",
    )

    assert api.upload_calls == [remote_path]
    assert uploaded.path_in_repo == remote_path
    assert uploaded.size_bytes == source.stat().st_size
    assert uploaded.sha256 == sha256_file(source)
    assert uploaded.local_path.read_bytes() == source.read_bytes()


def test_listing_is_filtered_and_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"

    paths = [
        "coverage/BTC/5000/2026/08/12/18.json",
        "coverage/BTC/5000/2026/08/12/17.json",
        "coverage/ETH/5000/2026/08/12/18.json",
    ]

    for relative in paths:
        path = remote_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    _install_fake_downloader(
        monkeypatch,
        remote_root,
    )

    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=_FakeApi(remote_root),
    )

    assert store.list_files(prefix="coverage/BTC") == (
        "coverage/BTC/5000/2026/08/12/17.json",
        "coverage/BTC/5000/2026/08/12/18.json",
    )


def test_delete_is_explicit_bounded_and_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"

    delete_paths = (
        "coverage/BTC/5000/2026/08/12/18.json",
        "market_points/BTC/5000/2026/08/12/18.parquet",
    )

    for relative in delete_paths:
        path = remote_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    _install_fake_downloader(
        monkeypatch,
        remote_root,
    )

    api = _FakeApi(remote_root)
    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=api,
    )

    deleted = store.delete_files(
        delete_paths,
        commit_message="test: retention delete",
    )

    assert deleted == tuple(sorted(delete_paths))
    assert api.delete_calls == [tuple(sorted(delete_paths))]
    assert store.list_files() == ()


def test_current_revision_returns_repository_sha(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()

    _install_fake_downloader(monkeypatch, remote_root)

    api = _FakeApi(remote_root)
    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=api,
    )

    assert store.current_revision() == api.base_revision


def test_atomic_commit_adds_and_deletes_in_one_transition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()

    _install_fake_downloader(monkeypatch, remote_root)

    old_path = "coverage/BTC/5000/2026/05/01/00.json"
    old_file = remote_root / old_path
    old_file.parent.mkdir(parents=True, exist_ok=True)
    old_file.write_text("expired", encoding="utf-8")

    source = tmp_path / "new.parquet"
    source.write_bytes(b"new deterministic object")

    new_path = "market_points/BTC/5000/2026/08/12/18.parquet"

    api = _FakeApi(remote_root)
    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=api,
    )

    result = store.commit_atomic(
        {new_path: source},
        [old_path],
        parent_revision=api.base_revision,
        commit_message="test: atomic add and retention delete",
    )

    assert len(api.create_commit_calls) == 1

    call = api.create_commit_calls[0]
    assert call["parent_commit"] == api.base_revision

    assert any(
        isinstance(operation, CommitOperationAdd) for operation in call["operations"]
    )
    assert any(
        isinstance(operation, CommitOperationDelete) for operation in call["operations"]
    )

    assert result.parent_revision == api.base_revision
    assert result.committed_revision == api.committed_revision
    assert result.added_paths == (new_path,)
    assert result.deleted_paths == (old_path,)

    assert (remote_root / new_path).read_bytes() == source.read_bytes()
    assert not (remote_root / old_path).exists()


def test_atomic_commit_addition_wins_over_same_path_deletion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()

    _install_fake_downloader(monkeypatch, remote_root)

    source = tmp_path / "replacement.json"
    source.write_text("replacement", encoding="utf-8")

    path = "coverage/BTC/5000/2026/08/12/18.json"

    api = _FakeApi(remote_root)
    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=api,
    )

    result = store.commit_atomic(
        {path: source},
        [path],
        parent_revision=api.base_revision,
        commit_message="test: replacement owns final state",
    )

    assert result.added_paths == (path,)
    assert result.deleted_paths == ()
    assert (remote_root / path).read_text(encoding="utf-8") == "replacement"


def test_atomic_commit_rejects_empty_operation_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_root = tmp_path / "remote"
    remote_root.mkdir()

    _install_fake_downloader(monkeypatch, remote_root)

    api = _FakeApi(remote_root)
    store = HuggingFaceArchiveStore(
        _settings(tmp_path),
        api=api,
    )

    with pytest.raises(ValueError, match="no operations"):
        store.commit_atomic(
            {},
            [],
            parent_revision=api.base_revision,
            commit_message="test: invalid empty commit",
        )

    assert api.create_commit_calls == []

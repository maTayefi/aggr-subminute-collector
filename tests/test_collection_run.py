from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from collector.aggr_client import AggrHistoricalResponse
from collector.config import CollectorSettings
from collector.hf_store import (
    AtomicCommitResult,
    RemoteObject,
)
from collector.market_policy import (
    BTC_POLICY,
    COLLECTOR_SCHEMA_VERSION,
    ETH_POLICY,
    SOURCE_INTERVAL_MS,
)
from collector.models import CanonicalMarketPoint
from collector.run import (
    CollectionWindow,
    plan_collection_publication,
    prepare_collection_run,
    publish_collection_run,
    rolling_collection_window,
)


class _FakeClient:
    def __init__(
        self,
        responses: dict[str, AggrHistoricalResponse],
        *,
        fail_base: str = "",
    ) -> None:
        self.responses = responses
        self.fail_base = fail_base
        self.calls: list[str] = []

    def fetch(
        self,
        *,
        policy,
        from_ms,
        to_ms,
        timeframe_ms,
    ):
        assert timeframe_ms == SOURCE_INTERVAL_MS
        self.calls.append(policy.base)

        if policy.base == self.fail_base:
            raise RuntimeError(f"synthetic {policy.base} failure")

        response = self.responses[policy.base]
        assert response.request_from_ms == from_ms
        assert response.request_to_ms == to_ms
        return response


class _MemoryStore:
    """Revision-aware in-memory stand-in for atomic HF publication."""

    BASE_REVISION = "fake-base-revision"
    COMMITTED_REVISION = "fake-committed-revision"

    def __init__(self, tmp_path: Path) -> None:
        self.remote_root = tmp_path / "remote"
        self.download_root = tmp_path / "downloads"

        # publish_collection_run() reads retention_days from store.settings.
        self.settings = SimpleNamespace(retention_days=90)

        self.revision_calls = 0
        self.download_calls: list[tuple[str, str | None]] = []
        self.list_calls: list[str | None] = []
        self.atomic_calls: list[dict] = []

        # Retained only as a convenient "visible publication paths" diagnostic.
        # Atomic publication has no meaningful Parquet-before-manifest order
        # because all operations become visible in one repository revision.
        self.upload_order: list[str] = []

        self.remote_root.mkdir(parents=True)
        self.download_root.mkdir(parents=True)

    def _object(self, path_in_repo: str) -> RemoteObject:
        from collector.hf_store import sha256_file

        path = self.remote_root / path_in_repo
        return RemoteObject(
            path_in_repo=path_in_repo,
            local_path=path,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )

    def current_revision(self) -> str:
        self.revision_calls += 1
        return self.BASE_REVISION

    def download_optional(
        self,
        path_in_repo: str,
        *,
        force_download: bool = True,
        revision: str | None = None,
    ):
        _ = force_download
        self.download_calls.append((path_in_repo, revision))

        if revision is not None:
            assert revision in {
                self.BASE_REVISION,
                self.COMMITTED_REVISION,
            }

        path = self.remote_root / path_in_repo
        if not path.is_file():
            return None

        return self._object(path_in_repo)

    def require_download(
        self,
        path_in_repo: str,
        *,
        force_download: bool = True,
        revision: str | None = None,
    ):
        downloaded = self.download_optional(
            path_in_repo,
            force_download=force_download,
            revision=revision,
        )

        if downloaded is None:
            raise FileNotFoundError(path_in_repo)

        return downloaded

    def list_files(
        self,
        *,
        prefix: str = "",
        revision: str | None = None,
    ) -> tuple[str, ...]:
        self.list_calls.append(revision)

        if revision is not None:
            assert revision in {
                self.BASE_REVISION,
                self.COMMITTED_REVISION,
            }

        normalized_prefix = str(prefix or "").strip().replace("\\", "/")
        normalized_prefix = normalized_prefix.strip("/")

        files = [
            path.relative_to(self.remote_root).as_posix()
            for path in self.remote_root.rglob("*")
            if path.is_file()
        ]

        if normalized_prefix:
            prefix_with_separator = normalized_prefix + "/"
            files = [path for path in files if path.startswith(prefix_with_separator)]

        return tuple(sorted(files))

    def commit_atomic(
        self,
        additions: dict[str, Path],
        deletions,
        *,
        parent_revision: str,
        commit_message: str,
    ) -> AtomicCommitResult:
        assert parent_revision == self.BASE_REVISION
        assert commit_message

        normalized_additions = {
            str(path).replace("\\", "/"): Path(source)
            for path, source in additions.items()
        }
        normalized_deletions = {str(path).replace("\\", "/") for path in deletions}
        normalized_deletions.difference_update(normalized_additions)

        self.atomic_calls.append(
            {
                "parent_revision": parent_revision,
                "commit_message": commit_message,
                "additions": tuple(sorted(normalized_additions)),
                "deletions": tuple(sorted(normalized_deletions)),
            }
        )

        # Model the final state of one atomic repository transition. The copy
        # order below has no publication semantics; callers observe only the
        # completed revision.
        for path_in_repo, source in normalized_additions.items():
            assert source.is_file()

            target = self.remote_root / path_in_repo
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        for path_in_repo in normalized_deletions:
            (self.remote_root / path_in_repo).unlink(missing_ok=True)

        self.upload_order.extend(sorted(normalized_additions))

        return AtomicCommitResult(
            parent_revision=parent_revision,
            committed_revision=self.COMMITTED_REVISION,
            added_paths=tuple(sorted(normalized_additions)),
            deleted_paths=tuple(sorted(normalized_deletions)),
        )

    # Retain the old helper only for isolated compatibility tests. Production
    # publish_collection_run() must use commit_atomic(), not this method.
    def upload_verified(
        self,
        local_path: Path,
        path_in_repo: str,
        *,
        commit_message: str,
    ):
        assert commit_message

        target = self.remote_root / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)
        self.upload_order.append(path_in_repo)

        return self._object(path_in_repo)


def _settings(tmp_path: Path) -> CollectorSettings:
    return CollectorSettings(
        hf_token=SecretStr("test-token"),
        work_dir=tmp_path / "work",
        rolling_refetch_hours=3,
        source_finalization_lag_seconds=60,
    )


def _point(policy, timestamp: int) -> CanonicalMarketPoint:
    market = policy.price_market

    return CanonicalMarketPoint(
        ts_utc_seconds=timestamp,
        market=market,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        cbuy=2,
        csell=1,
        vbuy=1000.0,
        vsell=400.0,
        lbuy=None,
        lsell=None,
        source_interval_ms=SOURCE_INTERVAL_MS,
        collector_schema_version=COLLECTOR_SCHEMA_VERSION,
        market_policy_version=policy.version,
        market_tier=policy.tier_by_market[market],
    )


def _responses(window) -> dict[str, AggrHistoricalResponse]:
    timestamp = window.from_ms // 1000

    return {
        "BTC": AggrHistoricalResponse(
            points=(_point(BTC_POLICY, timestamp),),
            exact_duplicates_dropped=0,
            source_row_count=1,
            request_from_ms=window.from_ms,
            request_to_ms=window.to_ms,
        ),
        "ETH": AggrHistoricalResponse(
            points=(_point(ETH_POLICY, timestamp),),
            exact_duplicates_dropped=0,
            source_row_count=1,
            request_from_ms=window.from_ms,
            request_to_ms=window.to_ms,
        ),
    }


def test_rolling_window_stays_one_minute_behind_now(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    now = datetime(
        2026,
        8,
        13,
        12,
        34,
        47,
        tzinfo=timezone.utc,
    )

    window = rolling_collection_window(
        settings,
        now_utc=now,
    )

    # now - 60 seconds = 12:33:47; floor to five seconds = 12:33:45.
    assert window.to_ms == 1_786_624_425_000
    assert window.from_ms == window.to_ms - 3 * 3_600_000
    assert window.from_ms % 5000 == 0
    assert window.to_ms % 5000 == 0


def test_both_fetches_finish_before_any_remote_upload(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    now = datetime(
        2026,
        8,
        13,
        12,
        34,
        47,
        tzinfo=timezone.utc,
    )
    window = rolling_collection_window(
        settings,
        now_utc=now,
    )

    client = _FakeClient(_responses(window))
    store = _MemoryStore(tmp_path)

    prepared = prepare_collection_run(
        settings,
        client=client,
        store=store,
        now_utc=now,
    )

    assert client.calls == ["BTC", "ETH"]
    assert store.revision_calls == 1
    assert store.upload_order == []
    assert store.atomic_calls == []
    assert prepared.base_revision == store.BASE_REVISION
    assert prepared.partitions

    # Every attempted manifest/data read during reconciliation must be pinned to
    # the same immutable repository revision.
    assert store.download_calls
    assert all(
        revision == prepared.base_revision for _path, revision in store.download_calls
    )

    published = publish_collection_run(
        store,
        prepared,
    )

    assert published.committed_revision == store.COMMITTED_REVISION
    assert published.uploaded_data_files
    assert published.uploaded_manifest_files

    # Production publication is one repository transition. There must not be a
    # separate Parquet commit followed by a manifest commit.
    assert len(store.atomic_calls) == 1

    atomic_call = store.atomic_calls[0]
    assert atomic_call["parent_revision"] == prepared.base_revision

    added_paths = set(atomic_call["additions"])

    assert added_paths == {
        *published.uploaded_data_files,
        *published.uploaded_manifest_files,
    }

    assert all(
        path.startswith("market_points/") for path in published.uploaded_data_files
    )
    assert all(
        path.startswith("coverage/") for path in published.uploaded_manifest_files
    )

    # Both object families become visible in the same committed revision.
    assert set(store.upload_order) == added_paths


def test_second_base_fetch_failure_uploads_nothing(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    now = datetime(
        2026,
        8,
        13,
        12,
        34,
        47,
        tzinfo=timezone.utc,
    )
    window = rolling_collection_window(
        settings,
        now_utc=now,
    )

    client = _FakeClient(
        _responses(window),
        fail_base="ETH",
    )
    store = _MemoryStore(tmp_path)

    with pytest.raises(RuntimeError, match="synthetic ETH failure"):
        prepare_collection_run(
            settings,
            client=client,
            store=store,
            now_utc=now,
        )

    assert client.calls == ["BTC", "ETH"]

    # Source validation is all-or-fail before remote archive discovery.
    assert store.revision_calls == 0
    assert store.download_calls == []
    assert store.list_calls == []
    assert store.atomic_calls == []
    assert store.upload_order == []


def test_successful_empty_source_response_records_query_provenance(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    now = datetime(
        2026,
        8,
        13,
        12,
        34,
        47,
        tzinfo=timezone.utc,
    )
    window = rolling_collection_window(
        settings,
        now_utc=now,
    )

    responses = {
        base: AggrHistoricalResponse(
            points=(),
            exact_duplicates_dropped=0,
            source_row_count=0,
            request_from_ms=window.from_ms,
            request_to_ms=window.to_ms,
        )
        for base in ("BTC", "ETH")
    }

    prepared = prepare_collection_run(
        settings,
        client=_FakeClient(responses),
        store=_MemoryStore(tmp_path),
        now_utc=now,
    )

    assert prepared.partitions
    assert all(item.partition.row_count == 0 for item in prepared.partitions)
    assert all(item.manifest.successful_query_ranges for item in prepared.partitions)
    assert all(
        item.manifest.market_completeness == "unproven" for item in prepared.partitions
    )

    assert set(prepared.responses) == {"BTC", "ETH"}
    assert all(len(segments) == 1 for segments in prepared.responses.values())


def test_publication_plan_is_read_only_and_matches_atomic_commit(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    now = datetime(
        2026,
        8,
        13,
        12,
        34,
        47,
        tzinfo=timezone.utc,
    )
    window = rolling_collection_window(
        settings,
        now_utc=now,
    )

    store = _MemoryStore(tmp_path)
    prepared = prepare_collection_run(
        settings,
        client=_FakeClient(_responses(window)),
        store=store,
        now_utc=now,
    )

    plan = plan_collection_publication(
        store,
        prepared,
    )

    assert plan.additions
    assert plan.data_files
    assert plan.manifest_files
    assert store.atomic_calls == []

    published = publish_collection_run(
        store,
        prepared,
    )

    assert len(store.atomic_calls) == 1
    assert set(store.atomic_calls[0]["additions"]) == set(plan.additions)
    assert published.uploaded_data_files == plan.data_files
    assert published.uploaded_manifest_files == plan.manifest_files
    assert published.deleted_files == plan.deleted_files


def test_prepare_accepts_explicit_internal_window(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    window = CollectionWindow(
        from_ms=1_786_560_000_000,
        to_ms=1_786_560_015_000,
    )

    responses = {
        base: AggrHistoricalResponse(
            points=(),
            exact_duplicates_dropped=0,
            source_row_count=0,
            request_from_ms=window.from_ms,
            request_to_ms=window.to_ms,
        )
        for base in ("BTC", "ETH")
    }

    prepared = prepare_collection_run(
        settings,
        client=_FakeClient(responses),
        store=_MemoryStore(tmp_path),
        window=window,
    )

    assert prepared.window == window
    assert prepared.partitions


def test_oversized_manual_window_is_rejected_before_any_io(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    # 126 complete UTC hours require:
    #   126 hours × 2 bases × 2 objects = 504 additions.
    window = CollectionWindow(
        from_ms=1_786_560_000_000,
        to_ms=1_786_560_000_000 + 126 * 3_600_000,
    )

    client = _FakeClient({})
    store = _MemoryStore(tmp_path)

    with pytest.raises(ValueError, match="before any AGGR or Hugging Face I/O"):
        prepare_collection_run(
            settings,
            client=client,
            store=store,
            window=window,
        )

    assert client.calls == []
    assert store.revision_calls == 0
    assert store.download_calls == []
    assert store.list_calls == []
    assert store.atomic_calls == []

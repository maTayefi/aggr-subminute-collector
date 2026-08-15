# collector\__init__.py

"""Autonomous aggr.trade five-second archive collector.

This package is independent from the local SMP NiceGUI/PostgreSQL application.
It fetches and validates aggr.trade historical data, merges fixed hourly
Parquet partitions, writes coverage manifests, and uploads them to a private
Hugging Face Dataset repository.
"""

from collector.aggr_client import (
    AGGR_HISTORICAL_BASE_URL,
    AggrHistoricalClient,
    AggrHistoricalResponse,
    build_historical_url,
    parse_aggr_historical_payload,
)
from collector.config import (
    DEFAULT_HF_REPO_ID,
    CollectorSettings,
)
from collector.hf_store import (
    AtomicCommitResult,
    HuggingFaceArchiveStore,
    RemoteObject,
    normalize_repo_path,
)
from collector.manifest import (
    PARTITION_MANIFEST_SCHEMA,
    CoveredRange,
    PartitionManifest,
    build_partition_manifest,
    partition_manifest_path,
    read_partition_manifest,
    write_partition_manifest,
)
from collector.market_policy import (
    BTC_POLICY,
    COLLECTOR_SCHEMA_VERSION,
    ETH_POLICY,
    MARKET_POLICY_SCHEMA_VERSION,
    SOURCE_INTERVAL_MS,
    MarketPolicy,
    MarketSpec,
    get_market_policy,
)
from collector.partition import (
    PARQUET_SCHEMA_VERSION,
    HourlyPartition,
    merge_partition_points,
    partition_data_path,
    partition_start_ms_for_point,
    read_hourly_partition,
    write_hourly_partition,
)
from collector.reconcile import (
    ReconciledHourlyPartition,
    reconcile_hourly_partition,
)
from collector.run import (
    CollectionWindow,
    PlannedCollectionPublication,
    PreparedCollectionRun,
    PublishedCollectionRun,
    plan_collection_publication,
    prepare_collection_run,
    publish_collection_run,
    rolling_collection_window,
    run_collection,
)
from collector.retention import (
    archive_object_partition_start_ms,
    plan_retention_deletions,
)

__all__ = [
    "AGGR_HISTORICAL_BASE_URL",
    "BTC_POLICY",
    "COLLECTOR_SCHEMA_VERSION",
    "DEFAULT_HF_REPO_ID",
    "ETH_POLICY",
    "MARKET_POLICY_SCHEMA_VERSION",
    "PARQUET_SCHEMA_VERSION",
    "PARTITION_MANIFEST_SCHEMA",
    "SOURCE_INTERVAL_MS",
    "AggrHistoricalClient",
    "AggrHistoricalResponse",
    "CollectorSettings",
    "CoveredRange",
    "HourlyPartition",
    "HuggingFaceArchiveStore",
    "MarketPolicy",
    "MarketSpec",
    "PartitionManifest",
    "RemoteObject",
    "CollectionWindow",
    "PlannedCollectionPublication",
    "PreparedCollectionRun",
    "PublishedCollectionRun",
    "ReconciledHourlyPartition",
    "AtomicCommitResult",
    "archive_object_partition_start_ms",
    "plan_retention_deletions",
    "build_historical_url",
    "build_partition_manifest",
    "get_market_policy",
    "merge_partition_points",
    "normalize_repo_path",
    "parse_aggr_historical_payload",
    "partition_data_path",
    "partition_manifest_path",
    "partition_start_ms_for_point",
    "read_hourly_partition",
    "write_hourly_partition",
    "write_partition_manifest",
    "plan_collection_publication",
    "prepare_collection_run",
    "publish_collection_run",
    "read_partition_manifest",
    "reconcile_hourly_partition",
    "rolling_collection_window",
    "run_collection",
]

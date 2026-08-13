# collector\__init__.py

"""Autonomous five-second aggr.trade archive collector.

This project is independent from the local SMP NiceGUI application. It fetches
validated historical market points and, in later modules, writes deterministic
hourly Parquet partitions to a private Hugging Face Dataset repository.
"""

from collector.aggr_client import (
    AGGR_HISTORICAL_BASE_URL,
    AggrHistoricalClient,
    AggrHistoricalResponse,
    build_historical_url,
    parse_aggr_historical_payload,
)
from collector.market_policy import (
    COLLECTOR_SCHEMA_VERSION,
    SOURCE_INTERVAL_MS,
    MarketPolicy,
    MarketSpec,
    get_market_policy,
)

__all__ = [
    "AGGR_HISTORICAL_BASE_URL",
    "COLLECTOR_SCHEMA_VERSION",
    "SOURCE_INTERVAL_MS",
    "AggrHistoricalClient",
    "AggrHistoricalResponse",
    "MarketPolicy",
    "MarketSpec",
    "build_historical_url",
    "get_market_policy",
    "parse_aggr_historical_payload",
]
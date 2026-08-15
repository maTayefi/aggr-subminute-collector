# collector\manifest.py
"""Deterministic sidecar manifests for hourly sub-minute partitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from collector.market_policy import (
    COLLECTOR_SCHEMA_VERSION,
    SOURCE_INTERVAL_MS,
    MarketPolicy,
)
from collector.partition import (
    HourlyPartition,
    partition_data_path,
)

PARTITION_MANIFEST_SCHEMA = "smp_aggr_subminute_partition.v1"


@dataclass(frozen=True, order=True, slots=True)
class CoveredRange:
    """One certified internal half-open millisecond range."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if isinstance(self.start_ms, bool) or not isinstance(self.start_ms, int):
            raise ValueError("CoveredRange.start_ms must be an integer")

        if isinstance(self.end_ms, bool) or not isinstance(self.end_ms, int):
            raise ValueError("CoveredRange.end_ms must be an integer")

        if self.start_ms <= 0 or self.end_ms <= self.start_ms:
            raise ValueError("CoveredRange must be a non-empty positive range")

        if (
            self.start_ms % SOURCE_INTERVAL_MS != 0
            or self.end_ms % SOURCE_INTERVAL_MS != 0
        ):
            raise ValueError("CoveredRange boundaries must align to five seconds")


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    schema: str
    base: str
    source_interval_ms: int
    partition_start_ms: int
    partition_end_ms: int
    data_file: str
    data_sha256: str
    row_count: int
    collector_schema_version: str
    market_policy_version: str
    market_list_hash: str
    selected_markets: tuple[str, ...]
    price_market: str
    covered_ranges: tuple[CoveredRange, ...]

    @property
    def partition_start_utc(self) -> datetime:
        return datetime.fromtimestamp(
            self.partition_start_ms / 1000,
            tz=timezone.utc,
        )

    @property
    def partition_end_utc(self) -> datetime:
        return datetime.fromtimestamp(
            self.partition_end_ms / 1000,
            tz=timezone.utc,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base": self.base,
            "source_interval_ms": self.source_interval_ms,
            "partition_start_utc": _iso_utc(self.partition_start_ms),
            "partition_end_utc": _iso_utc(self.partition_end_ms),
            "data_file": self.data_file,
            "data_sha256": self.data_sha256,
            "row_count": self.row_count,
            "collector_schema_version": self.collector_schema_version,
            "market_policy_version": self.market_policy_version,
            "market_list_hash": self.market_list_hash,
            "selected_markets": list(self.selected_markets),
            "price_market": self.price_market,
            "covered_ranges": [
                [
                    _iso_utc(item.start_ms),
                    _iso_utc(item.end_ms),
                ]
                for item in self.covered_ranges
            ],
        }

    def to_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")


def _iso_utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).isoformat(timespec="seconds")


def partition_manifest_path(
    base: str,
    partition_start_ms: int,
) -> str:
    base_u = str(base or "").strip().upper()
    if base_u not in {"BTC", "ETH"}:
        raise ValueError("Manifest base must be BTC or ETH")

    if (
        isinstance(partition_start_ms, bool)
        or not isinstance(partition_start_ms, int)
        or partition_start_ms <= 0
        or partition_start_ms % 3_600_000 != 0
    ):
        raise ValueError("partition_start_ms must be a positive UTC-hour integer")

    timestamp = datetime.fromtimestamp(
        partition_start_ms / 1000,
        tz=timezone.utc,
    )

    return f"coverage/{base_u}/{SOURCE_INTERVAL_MS}/" f"{timestamp:%Y/%m/%d/%H}.json"


def normalize_covered_ranges(
    ranges: Iterable[CoveredRange],
    *,
    partition_start_ms: int,
    partition_end_ms: int,
) -> tuple[CoveredRange, ...]:
    """Clip, sort, and combine overlapping or adjacent coverage."""

    clipped: list[CoveredRange] = []

    for item in ranges:
        if not isinstance(item, CoveredRange):
            raise TypeError("covered ranges must contain CoveredRange values")

        start = max(item.start_ms, partition_start_ms)
        end = min(item.end_ms, partition_end_ms)

        if end <= start:
            continue

        clipped.append(CoveredRange(start, end))

    clipped.sort()

    merged: list[CoveredRange] = []

    for item in clipped:
        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]

        if item.start_ms <= previous.end_ms:
            merged[-1] = CoveredRange(
                previous.start_ms,
                max(previous.end_ms, item.end_ms),
            )
        else:
            merged.append(item)

    return tuple(merged)


def build_partition_manifest(
    partition: HourlyPartition,
    *,
    policy: MarketPolicy,
    covered_ranges: Iterable[CoveredRange],
) -> PartitionManifest:
    """Construct a manifest only after its Parquet file was written."""

    if partition.base != policy.base:
        raise ValueError("Partition and policy base do not match")

    if partition.row_count != len(partition.points):
        raise ValueError("Partition row_count does not match its points")

    if len(partition.data_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in partition.data_sha256
    ):
        raise ValueError("Partition SHA-256 is invalid")

    normalized_ranges = normalize_covered_ranges(
        covered_ranges,
        partition_start_ms=partition.partition_start_ms,
        partition_end_ms=partition.partition_end_ms,
    )

    return PartitionManifest(
        schema=PARTITION_MANIFEST_SCHEMA,
        base=policy.base,
        source_interval_ms=SOURCE_INTERVAL_MS,
        partition_start_ms=partition.partition_start_ms,
        partition_end_ms=partition.partition_end_ms,
        data_file=partition.data_file,
        data_sha256=partition.data_sha256,
        row_count=partition.row_count,
        collector_schema_version=COLLECTOR_SCHEMA_VERSION,
        market_policy_version=policy.version,
        market_list_hash=policy.market_list_hash,
        selected_markets=policy.selected_markets,
        price_market=policy.price_market,
        covered_ranges=normalized_ranges,
    )


def _strict_manifest_integer(
    name: str,
    value: Any,
    *,
    nonnegative: bool = False,
) -> int:
    """Parse an exact manifest integer without truncating floats."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")

    if isinstance(value, int):
        output = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdigit():
            raise ValueError(f"{name} must be an integer; got {value!r}")
        output = int(text)
    else:
        raise ValueError(f"{name} must be an integer; got {value!r}")

    if nonnegative and output < 0:
        raise ValueError(f"{name} must be non-negative; got {output}")

    return output


def _manifest_timestamp_ms(
    name: str,
    value: Any,
) -> int:
    """Parse one timezone-aware UTC manifest timestamp to milliseconds."""

    try:
        timestamp = datetime.fromisoformat(
            str(value or "").strip().replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp") from exc

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")

    timestamp = timestamp.astimezone(timezone.utc)

    if timestamp.microsecond != 0:
        raise ValueError(f"{name} must be whole-second aligned")

    output = int(timestamp.timestamp()) * 1000

    if output <= 0 or output % SOURCE_INTERVAL_MS != 0:
        raise ValueError(f"{name} must align to a positive five-second boundary")

    return output


def _strict_sha256(name: str, value: Any) -> str:
    output = str(value or "").strip().lower()

    if len(output) != 64 or any(
        character not in "0123456789abcdef" for character in output
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")

    return output


def read_partition_manifest(
    path: Path,
    *,
    policy: MarketPolicy,
    partition_start_ms: int,
) -> PartitionManifest:
    """Read one collector-owned manifest and verify its complete identity."""

    source = Path(path)

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read partition manifest {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Partition manifest is invalid JSON: {source}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Partition manifest must contain one JSON object")

    schema = str(payload.get("schema") or "").strip()
    if schema != PARTITION_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported partition manifest schema {schema!r}")

    base = str(payload.get("base") or "").strip().upper()
    if base != policy.base:
        raise ValueError(
            f"Partition manifest base mismatch: expected {policy.base!r}, "
            f"got {base!r}"
        )

    start_ms = _strict_manifest_integer(
        "partition_start_ms",
        partition_start_ms,
    )
    if start_ms <= 0 or start_ms % 3_600_000 != 0:
        raise ValueError("partition_start_ms must be a positive UTC-hour boundary")

    end_ms = start_ms + 3_600_000

    observed_start = _manifest_timestamp_ms(
        "partition_start_utc",
        payload.get("partition_start_utc"),
    )
    observed_end = _manifest_timestamp_ms(
        "partition_end_utc",
        payload.get("partition_end_utc"),
    )

    if observed_start != start_ms or observed_end != end_ms:
        raise ValueError("Partition manifest hourly identity does not match its path")

    interval = _strict_manifest_integer(
        "source_interval_ms",
        payload.get("source_interval_ms"),
    )
    if interval != SOURCE_INTERVAL_MS:
        raise ValueError("Partition manifest source_interval_ms is inconsistent")

    expected_data_file = partition_data_path(
        policy.base,
        start_ms,
    )
    data_file = (
        str(payload.get("data_file") or "")
        .strip()
        .replace(
            "\\",
            "/",
        )
    )
    if data_file != expected_data_file:
        raise ValueError(
            f"Partition manifest data_file mismatch: "
            f"expected {expected_data_file!r}, got {data_file!r}"
        )

    data_sha256 = _strict_sha256(
        "data_sha256",
        payload.get("data_sha256"),
    )
    row_count = _strict_manifest_integer(
        "row_count",
        payload.get("row_count"),
        nonnegative=True,
    )

    collector_version = str(payload.get("collector_schema_version") or "").strip()
    if collector_version != COLLECTOR_SCHEMA_VERSION:
        raise ValueError("Partition manifest collector_schema_version is inconsistent")

    policy_version = str(payload.get("market_policy_version") or "").strip()
    if policy_version != policy.version:
        raise ValueError("Partition manifest market_policy_version is inconsistent")

    market_list_hash = _strict_sha256(
        "market_list_hash",
        payload.get("market_list_hash"),
    )
    if market_list_hash != policy.market_list_hash:
        raise ValueError("Partition manifest market_list_hash is inconsistent")

    raw_markets = payload.get("selected_markets")
    if not isinstance(raw_markets, list):
        raise ValueError("selected_markets must be a JSON list")

    selected_markets = tuple(str(value or "").strip() for value in raw_markets)
    if (
        any(not market for market in selected_markets)
        or len(selected_markets) != len(set(selected_markets))
        or selected_markets != policy.selected_markets
    ):
        raise ValueError("Partition manifest selected_markets is inconsistent")

    price_market = str(payload.get("price_market") or "").strip()
    if price_market != policy.price_market:
        raise ValueError("Partition manifest price_market is inconsistent")

    raw_ranges = payload.get("covered_ranges")
    if not isinstance(raw_ranges, list):
        raise ValueError("covered_ranges must be a JSON list")

    parsed_ranges: list[CoveredRange] = []

    for index, raw_range in enumerate(raw_ranges):
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError(f"covered_ranges[{index}] must be [start, end]")

        range_start = _manifest_timestamp_ms(
            f"covered_ranges[{index}][0]",
            raw_range[0],
        )
        range_end = _manifest_timestamp_ms(
            f"covered_ranges[{index}][1]",
            raw_range[1],
        )

        if range_start < start_ms or range_end > end_ms or range_end <= range_start:
            raise ValueError(f"covered_ranges[{index}] is outside its partition")

        parsed_ranges.append(
            CoveredRange(
                range_start,
                range_end,
            )
        )

    covered_ranges = tuple(parsed_ranges)

    canonical_ranges = normalize_covered_ranges(
        covered_ranges,
        partition_start_ms=start_ms,
        partition_end_ms=end_ms,
    )

    # Collector-produced manifests always contain already-sorted, merged,
    # non-overlapping, non-adjacent ranges.
    if covered_ranges != canonical_ranges:
        raise ValueError("Partition manifest covered_ranges is not canonical")

    return PartitionManifest(
        schema=schema,
        base=base,
        source_interval_ms=interval,
        partition_start_ms=start_ms,
        partition_end_ms=end_ms,
        data_file=data_file,
        data_sha256=data_sha256,
        row_count=row_count,
        collector_schema_version=collector_version,
        market_policy_version=policy_version,
        market_list_hash=market_list_hash,
        selected_markets=selected_markets,
        price_market=price_market,
        covered_ranges=covered_ranges,
    )


def write_partition_manifest(
    root: Path,
    manifest: PartitionManifest,
) -> Path:
    """Atomically write deterministic JSON after Parquet persistence."""

    relative_path = partition_manifest_path(
        manifest.base,
        manifest.partition_start_ms,
    )
    target = Path(root) / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(manifest.to_bytes())
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_name, target)

    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)
        raise

    return target

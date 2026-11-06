# collector/cli.py
"""Command-line entry point for scheduled and operator collection runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from collector.aggr_client import AggrHistoricalClient
from collector.config import CollectorSettings
from collector.hf_store import HuggingFaceArchiveStore
from collector.market_policy import SOURCE_INTERVAL_MS
from collector.run import (
    CollectionWindow,
    plan_collection_publication,
    prepare_collection_run,
    publish_collection_run,
    rolling_collection_window,
)

log = logging.getLogger(__name__)

_MANUAL_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _strict_localize(
    naive: datetime,
    tz_name: str,
    *,
    original_text: str,
) -> datetime:
    """Attach an IANA timezone while rejecting ambiguous/nonexistent times."""

    try:
        local_zone = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Invalid IANA timezone: {tz_name!r}") from exc

    candidates: dict[datetime, datetime] = {}

    for fold in (0, 1):
        aware = naive.replace(tzinfo=local_zone, fold=fold)
        utc_value = aware.astimezone(timezone.utc)
        roundtrip = utc_value.astimezone(local_zone).replace(
            tzinfo=None,
            fold=0,
        )

        if roundtrip == naive.replace(fold=0):
            candidates[utc_value] = aware

    if not candidates:
        raise ValueError(
            f"Local time {original_text!r} does not exist in timezone "
            f"{tz_name!r} because of a DST transition"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Local time {original_text!r} is ambiguous in timezone "
            f"{tz_name!r} because of a DST transition"
        )

    return next(iter(candidates.values()))


def _parse_manual_boundary(
    name: str,
    value: str,
    *,
    tz_name: str,
) -> datetime:
    text = str(value or "").strip()

    if not text:
        raise ValueError(f"{name} must be non-empty")

    try:
        naive = datetime.strptime(text, _MANUAL_DATETIME_FORMAT)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD HH:MM:SS; got {value!r}") from exc

    if naive.microsecond != 0 or naive.second % 5 != 0:
        raise ValueError(
            f"{name} must align to a five-second boundary: "
            "seconds must be 00, 05, 10, ..., 55"
        )

    localized = _strict_localize(
        naive,
        tz_name,
        original_text=text,
    )

    return localized.astimezone(timezone.utc)


def inclusive_collection_window(
    start_text: str,
    end_text: str,
    *,
    tz_name: str = "Asia/Tehran",
) -> CollectionWindow:
    """Translate user-visible inclusive ``[A, B]`` to internal ``[A, B+5s)``."""

    start = _parse_manual_boundary(
        "start",
        start_text,
        tz_name=tz_name,
    )
    end_inclusive = _parse_manual_boundary(
        "end",
        end_text,
        tz_name=tz_name,
    )

    if end_inclusive < start:
        raise ValueError(
            "Inclusive end must be greater than or equal to inclusive start"
        )

    from_ms = int(start.timestamp()) * 1000
    to_ms = int(end_inclusive.timestamp()) * 1000 + SOURCE_INTERVAL_MS

    if (
        from_ms <= 0
        or to_ms <= from_ms
        or from_ms % SOURCE_INTERVAL_MS != 0
        or to_ms % SOURCE_INTERVAL_MS != 0
    ):
        raise ValueError("Could not construct a valid five-second collection range")

    return CollectionWindow(
        from_ms=from_ms,
        to_ms=to_ms,
    )


def _window_from_args(
    args: argparse.Namespace,
    settings: CollectorSettings,
) -> CollectionWindow | None:
    start = str(getattr(args, "start", "") or "").strip()
    end = str(getattr(args, "end", "") or "").strip()

    if bool(start) != bool(end):
        raise ValueError("--start and --end must be supplied together")

    if not start:
        return None

    window = inclusive_collection_window(
        start,
        end,
        tz_name=str(args.timezone),
    )

    safe_frontier = rolling_collection_window(settings).to_ms

    # Retention must be evaluated against the current rolling frontier, not
    # against the manual window end. A historical manual range that already
    # lies outside the retention window would be deleted by the next scheduled
    # run, so reject it before doing any collection work.
    retention_cutoff_ms = safe_frontier - settings.retention_days * 86_400_000
    first_hour_end_ms = window.from_ms - (window.from_ms % 3_600_000) + 3_600_000
    if first_hour_end_ms <= retention_cutoff_ms:
        cutoff_text = datetime.fromtimestamp(
            retention_cutoff_ms / 1000,
            tz=timezone.utc,
        ).isoformat(timespec="seconds")
        raise ValueError(
            "The manual range starts in a UTC hour that retention would "
            "delete on the next scheduled run. Start at or after the hour "
            f"containing {cutoff_text}."
        )

    if window.to_ms > safe_frontier:
        safe_text = datetime.fromtimestamp(
            (safe_frontier - SOURCE_INTERVAL_MS) / 1000,
            tz=timezone.utc,
        ).isoformat(timespec="seconds")
        raise ValueError(
            "Manual ranges must not reach into the source-finalization lag. "
            f"End at or before {safe_text} UTC."
        )

    return window


def _build_client(settings: CollectorSettings) -> AggrHistoricalClient:
    return AggrHistoricalClient(
        timeout_seconds=settings.request_timeout_seconds,
        max_attempts=settings.request_max_attempts,
        initial_backoff_seconds=settings.request_initial_backoff_seconds,
    )


def _summary(
    prepared,
    *,
    mode: str,
    plan=None,
    committed_revision: str = "",
) -> dict[str, Any]:
    partitions = []

    for item in prepared.partitions:
        partitions.append(
            {
                "base": item.partition.base,
                "partition_start_ms": item.partition.partition_start_ms,
                "row_count": item.partition.row_count,
                "incoming_row_count": item.incoming_row_count,
                "existing_row_count": item.existing_row_count,
                "exact_duplicates_dropped": item.exact_duplicates_dropped,
                "replaced_incompatible_policy_era": (
                    item.replaced_incompatible_policy_era
                ),
                "data_file": item.partition.data_file,
                "manifest_file": item.manifest_local_path.relative_to(
                    prepared.staging_root
                ).as_posix(),
                "successful_query_ranges": [
                    [value.start_ms, value.end_ms]
                    for value in item.manifest.successful_query_ranges
                ],
                "market_completeness": (item.manifest.market_completeness),
            }
        )

    output: dict[str, Any] = {
        "mode": mode,
        "window": {
            "from_ms": prepared.window.from_ms,
            "to_ms": prepared.window.to_ms,
            "semantics": "[from, to)",
        },
        "base_revision": prepared.base_revision,
        "committed_revision": committed_revision or None,
        "source": {
            base: {
                "request_count": len(segments),
                "source_row_count": sum(
                    item.response.source_row_count for item in segments
                ),
                "unique_row_count": sum(
                    item.response.unique_row_count for item in segments
                ),
                "exact_duplicates_dropped": sum(
                    item.response.exact_duplicates_dropped for item in segments
                ),
                "policy_segments": [
                    {
                        "from_ms": item.window.from_ms,
                        "to_ms": item.window.to_ms,
                        "activation_start_ms": (item.activation_start_ms),
                        "market_policy_version": (item.policy.version),
                        "market_list_hash": (item.policy.market_list_hash),
                    }
                    for item in segments
                ],
            }
            for base, segments in prepared.responses.items()
        },
        "partitions": partitions,
    }

    if plan is not None:
        output["publication"] = {
            "addition_count": len(plan.additions),
            "data_files": list(plan.data_files),
            "manifest_files": list(plan.manifest_files),
            "deletion_count": len(plan.deleted_files),
            "deleted_files": list(plan.deleted_files),
        }

    return output


def _add_range_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--start",
        default="",
        help=(
            "Optional inclusive local start: YYYY-MM-DD HH:MM:SS. "
            "Must be supplied together with --end."
        ),
    )
    parser.add_argument(
        "--end",
        default="",
        help=(
            "Optional inclusive local end: YYYY-MM-DD HH:MM:SS. The internal "
            "request boundary is translated to end + 5 seconds."
        ),
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Tehran",
        help="IANA timezone used to interpret --start/--end.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggr-subminute-collector",
        description=(
            "Collect reviewed BTC/ETH five-second AGGR futures points into "
            "the private Hugging Face Dataset archive."
        ),
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    collect_parser = subparsers.add_parser(
        "collect",
        help="Prepare and atomically publish one collection run.",
    )
    _add_range_arguments(collect_parser)

    dry_run_parser = subparsers.add_parser(
        "dry-run",
        help=(
            "Fetch, reconcile, validate, and plan publication without calling "
            "Hugging Face create_commit()."
        ),
    )
    _add_range_arguments(dry_run_parser)

    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        settings = CollectorSettings()
        window = _window_from_args(args, settings)

        store = HuggingFaceArchiveStore(settings)
        client = _build_client(settings)

        prepared = prepare_collection_run(
            settings,
            client=client,
            store=store,
            window=window,
        )

        plan = plan_collection_publication(
            store,
            prepared,
        )

        if args.command == "dry-run":
            print(
                json.dumps(
                    _summary(
                        prepared,
                        mode="dry-run",
                        plan=plan,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        published = publish_collection_run(
            store,
            prepared,
        )

        print(
            json.dumps(
                _summary(
                    prepared,
                    mode="collect",
                    plan=plan,
                    committed_revision=published.committed_revision,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    except KeyboardInterrupt:
        log.warning("Collector interrupted by operator")
        return 130

    except Exception as exc:  # noqa: BLE001
        log.exception("Collector failed")
        print(
            f"collector failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

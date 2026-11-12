"""Local watchdog: dispatch the GitHub Actions collector only when needed.

Runs via Windows Task Scheduler. Performs at most two HTTPS API calls per
check. All collection work stays on GitHub-hosted runners.

Decision logic (fail closed):
1. Any run queued/in_progress/waiting  -> skip (it will refresh the archive).
2. Newest successful completion (ANY event) within STALE_AFTER_MINUTES -> skip.
3. We dispatched within DISPATCH_COOLDOWN_MINUTES -> skip.
4. >= MAX_DISPATCHES_PER_24H dispatches in the last 24 h -> skip.
5. Otherwise POST one workflow_dispatch with inputs.mode=collect.

API read failure means "unknown", never "missing": no dispatch.
Exit codes: 0 = skip or dispatched, 1 = transient failure, 2 = config/auth error.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "maTayefi/aggr-subminute-collector"
WORKFLOW_FILE = "collect.yml"
REF = "main"
DISPATCH_INPUTS = {"mode": "collect"}  # workflow default is dry-run; must override

STALE_AFTER_MINUTES = 120
DISPATCH_COOLDOWN_MINUTES = 60
MAX_DISPATCHES_PER_24H = 6

API_BASE = "https://api.github.com"
API_VERSION = "2026-03-10"
HTTP_TIMEOUT_SECONDS = 30
READ_MAX_ATTEMPTS = 3
READ_BACKOFF_SECONDS = 2.0

TOKEN_ENV_VAR = "AGGR_WATCHDOG_GH_TOKEN"
TOKEN_FILE = Path.home() / ".aggr_watchdog" / "github_token.txt"

ACTIVE_STATUSES = frozenset(
    {"queued", "in_progress", "waiting", "requested", "pending"}
)

log = logging.getLogger("aggr_watchdog")


class ConfigError(Exception):
    """Bad token, permissions, repo/workflow identity. Do not retry."""


class TransientError(Exception):
    """API unreadable. Fail closed: never dispatch on unknown state."""


def _data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home()
    path = root / "AggrCollectorWatchdog"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _setup_logging(data_dir: Path) -> None:
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        data_dir / "watchdog.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(console)


def _resolve_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if token:
        return token
    try:
        token = TOKEN_FILE.read_text(encoding="utf-8-sig").strip()
    except OSError:
        token = ""
    if not token:
        raise ConfigError(
            f"No GitHub token. Set {TOKEN_ENV_VAR} or write the token to {TOKEN_FILE}"
        )
    return token


def _request(
    method: str,
    url: str,
    token: str,
    body: dict | None = None,
) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", API_VERSION)
    request.add_header("User-Agent", "aggr-collector-watchdog/1.0")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.status, response.read()


def _list_runs(token: str) -> list[dict]:
    url = (
        f"{API_BASE}/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?branch={REF}&per_page=30"
    )
    last_error: BaseException | None = None
    for attempt in range(1, READ_MAX_ATTEMPTS + 1):
        try:
            status, payload = _request("GET", url, token)
            if status != 200:
                raise RuntimeError(f"Unexpected list status {status}")
            runs = json.loads(payload).get("workflow_runs", [])
            if not isinstance(runs, list):
                raise RuntimeError("Malformed workflow_runs payload")
            return runs
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 404):
                raise ConfigError(
                    f"HTTP {exc.code} listing runs. Check token validity, "
                    f"Actions read permission, and that {WORKFLOW_FILE} "
                    f"exists on branch {REF}."
                ) from exc
            if exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0":
                last_error = exc  # rate limited: transient
            elif exc.code == 403:
                raise ConfigError(
                    "HTTP 403 listing runs. Token likely lacks Actions read access."
                ) from exc
            else:
                last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
        if attempt < READ_MAX_ATTEMPTS:
            time.sleep(READ_BACKOFF_SECONDS * attempt)
    raise TransientError(
        f"Could not read workflow runs after {READ_MAX_ATTEMPTS} attempts; "
        "state unknown, not dispatching."
    ) from last_error


def _dispatch(token: str) -> dict | None:
    url = f"{API_BASE}/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    status, payload = _request(
        "POST", url, token, {"ref": REF, "inputs": DISPATCH_INPUTS}
    )
    if status == 204:  # older API versions: accepted, no body
        return None
    if status == 200:  # 2026-03-10: body contains workflow_run_id and URLs
        try:
            result = json.loads(payload)
            return result if isinstance(result, dict) else None
        except ValueError:
            return None
    raise RuntimeError(f"Unexpected dispatch status {status}")


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def _load_history(path: Path, now: datetime) -> list[datetime]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [_parse_utc(item) for item in raw.get("dispatch_history", [])]
    except (OSError, ValueError, TypeError):
        return []
    cutoff = now - timedelta(hours=24)
    return sorted(item for item in entries if item > cutoff)


def _save_history(path: Path, history: list[datetime]) -> None:
    payload = {"dispatch_history": [item.isoformat() for item in history]}
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AGGR collector dispatch watchdog")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate and log the decision, never dispatch.")
    parser.add_argument("--force", action="store_true",
                        help="Dispatch unconditionally (one-off operator test).")
    parser.add_argument("--stale-minutes", type=int, default=STALE_AFTER_MINUTES)
    parser.add_argument("--cooldown-minutes", type=int, default=DISPATCH_COOLDOWN_MINUTES)
    parser.add_argument("--max-per-day", type=int, default=MAX_DISPATCHES_PER_24H)
    args = parser.parse_args(argv)

    data_dir = _data_dir()
    _setup_logging(data_dir)
    state_path = data_dir / "state.json"
    now = datetime.now(timezone.utc)

    try:
        if min(args.stale_minutes, args.cooldown_minutes, args.max_per_day) <= 0:
            raise ConfigError("Thresholds must be positive integers.")
        token = _resolve_token()
        history = _load_history(state_path, now)

        if not args.force:
            runs = _list_runs(token)
            if runs:
                newest = runs[0]
                log.info(
                    "Newest run: id=%s event=%s status=%s conclusion=%s created=%s",
                    newest.get("id"), newest.get("event"), newest.get("status"),
                    newest.get("conclusion"), newest.get("created_at"),
                )
            active = [r for r in runs if r.get("status") in ACTIVE_STATUSES]
            if active:
                log.info("SKIP: run %s is %s; it will refresh the archive.",
                         active[0].get("id"), active[0].get("status"))
                return 0
            successes = [
                _parse_utc(r["updated_at"])
                for r in runs
                if r.get("status") == "completed" and r.get("conclusion") == "success"
            ]
            if successes:
                age_min = (now - max(successes)).total_seconds() / 60.0
                if age_min <= args.stale_minutes:
                    log.info("SKIP: last successful run finished %.1f min ago "
                             "(threshold %d min).", age_min, args.stale_minutes)
                    return 0
                log.warning("Last successful completion was %.1f min ago.", age_min)
            else:
                log.warning("No successful run found in the recent run list.")
            if history and (now - history[-1]) < timedelta(minutes=args.cooldown_minutes):
                log.info("SKIP: dispatch cooldown active (last dispatch %s).",
                         history[-1].isoformat())
                return 0
            if len(history) >= args.max_per_day:
                log.warning("SKIP: 24h dispatch cap reached (%d).", args.max_per_day)
                return 0
        else:
            log.warning("FORCE mode: bypassing staleness/cooldown/cap checks.")

        if args.dry_run:
            log.info("DRY-RUN: would dispatch %s ref=%s inputs=%s",
                     WORKFLOW_FILE, REF, DISPATCH_INPUTS)
            return 0

        try:
            result = _dispatch(token)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404, 422):
                raise ConfigError(
                    f"HTTP {exc.code} on dispatch. Check Actions write permission, "
                    f"that workflow_dispatch exists on {REF}, and ref/inputs."
                ) from exc
            history.append(now)
            _save_history(state_path, history)
            log.warning("Dispatch failed (HTTP %s); cooldown recorded; "
                        "next check re-evaluates.", exc.code)
            return 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # The commit may have succeeded remotely; never blindly replay.
            history.append(now)
            _save_history(state_path, history)
            log.warning("Dispatch outcome uncertain (%s); cooldown recorded; "
                        "next check re-evaluates.", exc)
            return 1

        history.append(now)
        _save_history(state_path, history)
        if result and result.get("workflow_run_id"):
            log.info("DISPATCHED: run id=%s url=%s",
                     result.get("workflow_run_id"), result.get("html_url"))
        else:
            log.info("DISPATCHED: GitHub accepted the workflow_dispatch event.")
        return 0

    except ConfigError as exc:
        log.error("CONFIG/AUTH ERROR: %s", exc)
        return 2
    except TransientError as exc:
        log.warning("TRANSIENT: %s", exc)
        return 1
    except Exception:
        log.exception("Watchdog failed unexpectedly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
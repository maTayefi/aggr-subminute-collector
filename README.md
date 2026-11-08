# SMP aggr.trade Sub-minute Collector

Autonomous five-second BTC/ETH derivatives-data collector for the local Smart
Money Profile application.

## Architecture

```text
GitHub Actions
    |
    | GET requests every 30 minutes, subject to scheduler delays
    v
api.aggr.trade/historical
    |
    | strict parsing and reviewed futures-only policy
    v
hourly UTC Parquet partitions + coverage manifests
    |
    | one atomic repository commit
    v
private Hugging Face Dataset:
maTayefi/smp-aggr-subminute
```

The collector GitHub repository is:

```text
maTayefi/aggr-subminute-collector
```

The collector is independent from SMP. It does not import or start:

- NiceGUI;
- PostgreSQL;
- SQLAlchemy;
- Coinalyze FetchService;
- EngineResult;
- OCFI;
- SMP profile engines.

The local SMP application is an on-demand archive consumer.

## Source contract

Historical request:

```text
GET /historical/{from_ms}/{to_ms}/5000/{encoded_markets}
```

Semantics:

- URL boundaries use UNIX milliseconds;
- response `time` values use UNIX seconds;
- source requests use `[from, to)` boundaries;
- rows are per-market aggregate five-second buckets;
- `vbuy` and `vsell` are USD-like buy/sell notionals;
- `cbuy` and `csell` are trade counts;
- `lbuy` and `lsell` are liquidation notionals;
- exact duplicate `(time, market)` rows inside one source response may be
  collapsed;
- conflicting duplicate rows inside one source response fail closed;
- during rolling reconciliation, a newly fetched canonical row is authoritative
  over an older archived row with the same `(time, market)` key. This allows the
  configured rolling refetch window to publish upstream corrections instead of
  becoming permanently blocked by the first archived value;
- rows absent from the new response are retained from the older trusted
  partition; version 1 does not infer source deletion merely from sparse
  response absence;
- successful empty responses may contain `columns={}` and `results=[]`.

The normal rolling collector remains one minute behind the current clock so
aggr-server's still-changing in-memory pending bars are not archived as final.
The rolling refetch window still reconciles later upstream corrections to
already archived keys.

This preserves existing rows absent from the new response. That is safer than treating a successful sparse/empty refetch as an instruction to delete every prior row in that range.

## Market policy

Version 1 supports BTC and ETH only.

The collector uses explicit reviewed derivatives-only allowlists. It does not
automatically construct the production universe from `/products`.

Tiers:

- USDT derivatives;
- USDC derivatives;
- USD/inverse/other USD-like derivatives.

The only execution-price source used by the local SMP chart is:

```text
BTC: BINANCE_FUTURES:btcusdt
ETH: BINANCE_FUTURES:ethusdt
```

Spot and options are not included in the reviewed collector policy.

## Policy eras

Market policies are versioned by exact UTC-hour activation boundaries.

The production registry currently contains only the existing BTC and ETH
version-1 policies. No synthetic or future policy is active.

When a future reviewed policy is deployed:

- its activation must be an exact UTC-hour boundary;
- source requests crossing that boundary are split;
- every hourly partition belongs to exactly one policy;
- old-policy rows are never merged into a new-policy partition;
- a partition written under an obsolete policy after the new activation may be
  replaced from the new-policy query rather than blocking rolling collection;
- historical pre-boundary partitions remain validated under their original
  policy identity.

A real future policy must be added independently to both the collector and SMP
consumer registries. Synthetic policies are test fixtures only.

## Archive layout

```text
market_points/BTC/5000/YYYY/MM/DD/HH.parquet
market_points/ETH/5000/YYYY/MM/DD/HH.parquet

coverage/BTC/5000/YYYY/MM/DD/HH.json
coverage/ETH/5000/YYYY/MM/DD/HH.json
```

Each version-2 manifest records:

- hourly partition identity;
- Parquet path;
- SHA-256;
- row count;
- collector schema version;
- market policy version;
- market-list hash;
- exact requested markets;
- Binance execution-price market;
- `successful_query_ranges`;
- `market_completeness="unproven"`.

The reviewed AGGR server does not provide market-level completeness proof.
Successful query ranges prove only that the storage query completed. They do
not prove that every requested market was historically configured, collected,
retained, or returned.

Legacy version-1 manifests remain readable. Their field named
`covered_ranges` is interpreted conservatively as successful storage-query
provenance with unproven market completeness. A touched v1 partition is
rewritten as v2 during ordinary reconciliation.

A missing manifest advertises no successful query provenance.

## Publication transaction

Each run:

1. fetches and validates BTC;
2. fetches and validates ETH;
3. captures one Hugging Face repository revision;
4. reads all existing objects from that immutable revision;
5. reconciles hourly partitions locally;
6. writes and validates every Parquet file;
7. writes every manifest;
8. plans bounded 90-day retention deletions;
9. publishes all additions, replacements, and deletions in one
   `HfApi.create_commit()` call;
10. supplies the reconciled revision as `parent_commit`;
11. verifies additions and deletions against the returned immutable commit.

If the repository changed after reconciliation, publication fails. Rerun the
collector instead of applying the stale plan.

Transient Hugging Face read failures are retried with bounded exponential
backoff for:

- repository revision discovery;
- repository file listing;
- immutable-revision manifest/Parquet downloads;
- post-commit verification downloads.

Atomic write operations are not blindly retried. A remote commit can succeed
while its HTTP response times out, so automatic replay could create an
uncertain duplicate or stale-parent attempt. In that case the run fails closed
and the next scheduled run reconciles against the repository's actual current
revision.

## User-visible inclusive ranges

Normal scheduled collection is internally half-open.

Manual CLI ranges use inclusive mathematical `[A, B]` semantics:

```text
--start "2026-08-12 13:00:00"
--end   "2026-08-12 13:00:10"
```

includes buckets:

```text
13:00:00
13:00:05
13:00:10
```

and translates internally to:

```text
[A, B + 5 seconds)
```

## Development

```powershell
cd C:\aggr-subminute-collector

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

python -m pytest -q
```

## Local configuration

Copy:

```powershell
Copy-Item .env.example .env
```

Set a dedicated Hugging Face write token in `.env`:

```dotenv
AGGR_COLLECTOR_HF_TOKEN=hf_your_write_token
```

Do not commit `.env`.

The SMP application should use a separate read-only token. Do not reuse the
collector write token in SMP.

## Dry run

A dry run performs source fetching, remote snapshot reads, reconciliation,
local Parquet/manifest creation, and retention planning, but does not call
`create_commit()`:

```powershell
python -m collector dry-run
```

Manual inclusive local range:

```powershell
python -m collector dry-run `
  --start "2026-08-12 13:00:00" `
  --end "2026-08-12 13:05:00" `
  --timezone "Asia/Tehran"
```

## Real collection

Normal rolling publication:

```powershell
python -m collector collect
```

Manual inclusive publication:

```powershell
python -m collector collect `
  --start "2026-08-12 13:00:00" `
  --end "2026-08-12 13:05:00" `
  --timezone "Asia/Tehran"
```

Manual ranges must not reach into the configured source-finalization lag.

## GitHub Actions setup

Repository:

```text
maTayefi/aggr-subminute-collector
```

Dataset:

```text
maTayefi/smp-aggr-subminute
```

Create a dedicated Hugging Face write token and add it to the GitHub collector
repository as this Actions secret:

```text
HF_TOKEN
```

Do not place the token in:

- workflow YAML;
- repository variables;
- README;
- logs;
- test fixtures;
- screenshots.

The workflow is scheduled every 30 minutes at minutes 7 and 37 UTC and
supports manual dispatch. Actual start times can be delayed.

Each normal run re-reads the configured 7-hour rolling source window,
ending at the finalized five-second frontier. This provides overlap for
recovery and upstream reconciliation; it is not a guarantee that every
requested observation remains available upstream.

Scheduled and manual runs share one concurrency group. A new run does not
cancel the active publisher. Pending runs are subject to GitHub Actions
concurrency and scheduling behavior; do not treat the schedule as a durable
queue of every requested execution.

## First production run

1. Verify the private Dataset exists.
2. Verify `HF_TOKEN` has write access only where required.
3. Run the full tests.
4. Trigger `workflow_dispatch` with `mode=dry-run`.
5. Inspect the JSON plan and workflow logs.
6. Trigger `workflow_dispatch` with `mode=collect`.
7. Verify one Hugging Face commit contains both:
   - `market_points/.../*.parquet`;
   - `coverage/.../*.json`.
8. Start SMP and load a small fully covered historical range.
9. Confirm:
   - manifest validation passes;
   - Binance price candles appear;
   - synthetic CVD appears;
   - buy/sell bars appear;
   - no token is printed.

## Retention

The latest repository revision retains at least 90 days.

A partition is eligible for deletion only when its complete UTC hour ends at or
before the retention cutoff.

Only recognized collector-owned paths are deleted. Unrelated repository files
are ignored.

Retention planning uses the remaining capacity in the 500-operation atomic
commit after additions/replacements are counted. When more expired objects
exist, the oldest recognized collector-owned objects are deleted first and
later scheduled runs continue the bounded cleanup.

## Limitations

- GitHub Actions scheduling is not real-time. Scheduled runs can start late
  or be dropped. The workflow is scheduled every 30 minutes, and each normal
  successful run re-reads a 7-hour source window. This reduces exposure to
  missed executions but does not guarantee complete five-second history.
- Recovery requires a successful fetch before the needed source observations
  expire. The 7-hour window, source-finalization lag, request execution time,
  upstream outages, and changing AGGR retention all affect recoverability.
  Already-expired observations cannot be reconstructed by a later collector run.
- A source outage longer than aggr.trade's five-second history window can create
  unrecoverable gaps.
- Free GitHub/Hugging Face policies and quotas can change.
- The archive has no commercial SLA.
- True intrabucket CVD high/low cannot be reconstructed from five-second
  buy/sell bucket totals; SMP shows endpoint-derived synthetic CVD candles.
- Missing Binance price buckets remain gaps.
- A policy-version boundary cannot be silently merged into one homogeneous CVD
  plot.
- AGGR HTTP success does not prove complete selected-market observation.
  Empty/no-row buckets are not certified zero activity. Returned rows are
  observed-row evidence only until a future source/manifest contract can
  provide market-level completeness proof.
```

---

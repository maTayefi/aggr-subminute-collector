# SMP aggr.trade Sub-minute Collector

Autonomous five-second BTC/ETH derivatives-data collector for the local
Smart Money Profile application.

## Architecture

```text
GitHub Actions
    |
    | hourly HTTP GET
    v
api.aggr.trade/historical
    |
    | strict validation and reviewed futures-only policy
    v
hourly UTC Parquet partitions + coverage manifests
    |
    v
private Hugging Face Dataset:
maTayefi/smp-aggr-subminute
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

Historical URL:

```text
GET /historical/{from_ms}/{to_ms}/5000/{encoded_markets}
```

Semantics:

- URL boundaries use UNIX milliseconds;
- response `time` values use UNIX seconds;
- requests use `[from, to)` boundaries;
- rows are per-market aggregate five-second buckets;
- `vbuy` and `vsell` are USD-like buy/sell notionals;
- `cbuy` and `csell` are trade counts;
- `lbuy` and `lsell` are liquidation notionals;
- exact duplicate `(time, market)` rows may be collapsed;
- conflicting duplicate rows fail closed.

## Market policy

Version 1 supports BTC and ETH only.

The production URLs originally supplied by the project owner mixed spot and
derivative markets. The collector uses reviewed derivatives-only allowlists.

Tiers:

- USDT;
- USDC;
- USD/inverse/other USD-like perpetual derivatives.

The exact Binance USDT perpetual is also the only execution-price source used
by the local chart:

```text
BTC: BINANCE_FUTURES:btcusdt
ETH: BINANCE_FUTURES:ethusdt
```

## Development

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

No credentials are required for parser and market-policy tests.

Hugging Face upload support, hourly Parquet merging, manifests, retention, CLI,
and GitHub Actions are added in the next implementation batch.
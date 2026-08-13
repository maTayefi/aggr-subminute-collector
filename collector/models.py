# collector\models.py
"""Canonical collector records and strict numeric helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def finite_float(
    value: Any,
    *,
    name: str,
    allow_none: bool = False,
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be null")

    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")

    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}") from exc

    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite; got {output!r}")

    return output


def nonnegative_float(
    value: Any,
    *,
    name: str,
    allow_none: bool = False,
) -> float | None:
    output = finite_float(
        value,
        name=name,
        allow_none=allow_none,
    )

    if output is not None and output < 0.0:
        raise ValueError(f"{name} must be non-negative; got {output}")

    return output


def nonnegative_count(
    value: Any,
    *,
    name: str,
    allow_none: bool = False,
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be null")

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc

    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer; got {value!r}")

    output = int(numeric)

    if output < 0:
        raise ValueError(f"{name} must be non-negative; got {output}")

    return output


def validate_nullable_ohlc(
    *,
    open_: Any,
    high: Any,
    low: Any,
    close: Any,
    context: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    raw = (open_, high, low, close)
    null_count = sum(value is None for value in raw)

    if null_count == 4:
        return None, None, None, None

    if null_count:
        raise ValueError(
            f"{context}: OHLC must be either fully present or fully null"
        )

    o = finite_float(open_, name=f"{context}.open")
    h = finite_float(high, name=f"{context}.high")
    l_ = finite_float(low, name=f"{context}.low")
    c = finite_float(close, name=f"{context}.close")

    assert o is not None
    assert h is not None
    assert l_ is not None
    assert c is not None

    if min(o, h, l_, c) <= 0.0:
        raise ValueError(
            f"{context}: OHLC values must be positive; "
            f"open={o}, high={h}, low={l_}, close={c}"
        )

    scale = max(abs(o), abs(h), abs(l_), abs(c), 1.0)
    tolerance = max(1e-9, scale * 1e-10)

    if h + tolerance < max(o, c, l_):
        raise ValueError(
            f"{context}: high is below another OHLC value; "
            f"open={o}, high={h}, low={l_}, close={c}"
        )

    if l_ - tolerance > min(o, c, h):
        raise ValueError(
            f"{context}: low is above another OHLC value; "
            f"open={o}, high={h}, low={l_}, close={c}"
        )

    if h + tolerance < l_:
        raise ValueError(
            f"{context}: high is below low; high={h}, low={l_}"
        )

    return o, h, l_, c


@dataclass(frozen=True, slots=True)
class CanonicalMarketPoint:
    """One market's aggregate five-second bucket."""

    ts_utc_seconds: int
    market: str

    open: float | None
    high: float | None
    low: float | None
    close: float | None

    cbuy: int
    csell: int
    vbuy: float
    vsell: float

    lbuy: float | None
    lsell: float | None

    source_interval_ms: int
    collector_schema_version: str
    market_policy_version: str
    market_tier: str

    @property
    def key(self) -> tuple[int, str]:
        return self.ts_utc_seconds, self.market

    def comparable_values(self) -> tuple[Any, ...]:
        return (
            self.open,
            self.high,
            self.low,
            self.close,
            self.cbuy,
            self.csell,
            self.vbuy,
            self.vsell,
            self.lbuy,
            self.lsell,
            self.source_interval_ms,
            self.collector_schema_version,
            self.market_policy_version,
            self.market_tier,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "ts_utc": self.ts_utc_seconds,
            "market": self.market,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "cbuy": self.cbuy,
            "csell": self.csell,
            "vbuy": self.vbuy,
            "vsell": self.vsell,
            "lbuy": self.lbuy,
            "lsell": self.lsell,
            "source_interval_ms": self.source_interval_ms,
            "collector_schema_version": self.collector_schema_version,
            "market_policy_version": self.market_policy_version,
            "market_tier": self.market_tier,
        }


def points_identical(
    left: CanonicalMarketPoint,
    right: CanonicalMarketPoint,
    *,
    absolute_tolerance: float = 1e-9,
) -> bool:
    if left.key != right.key:
        return False

    for a, b in zip(
        left.comparable_values(),
        right.comparable_values(),
    ):
        if a is None and b is None:
            continue

        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(
                float(a),
                float(b),
                rel_tol=0.0,
                abs_tol=absolute_tolerance,
            ):
                return False
            continue

        if a != b:
            return False

    return True
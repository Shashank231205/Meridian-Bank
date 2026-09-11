"""Foreign exchange reference rates (Frankfurter, backed by ECB).

Meridian reports in INR. FX enters the project in two places: converting the
USD-denominated FDIC peer benchmarks into comparable INR terms, and providing a
real exogenous time series for the seasonality work.

Frankfurter is the source that motivated the custom User-Agent in
``common/http`` -- it returns 403 to the default ``Python-urllib`` agent. No key
is required.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..common.exceptions import SchemaError
from ..common.http import get_bytes
from ..common.logging import get_logger
from .base import SourceResult

log = get_logger(__name__)

BASE = "https://api.frankfurter.app"


def fetch_latest(
    *, base_ccy: str = "EUR", symbols: str = "INR,USD,GBP",
    cache_dir: Path | None = None, ttl_days: int = 1,
) -> SourceResult:
    """Fetch the most recent published reference rates."""
    resp = get_bytes(f"{BASE}/latest", params={"from": base_ccy, "to": symbols},
                     cache_dir=cache_dir, ttl_days=ttl_days)
    payload = resp.json()
    rates = payload.get("rates")
    if not isinstance(rates, dict) or not rates:
        raise SchemaError(f"Frankfurter returned no rates for {base_ccy}->{symbols}")

    df = pd.DataFrame(
        [{"date": pd.to_datetime(payload.get("date")), "base": payload.get("base", base_ccy),
          "quote": k, "rate": float(v)} for k, v in rates.items()]
    )
    log.info("fx latest %s: %s", base_ccy, {k: round(float(v), 4) for k, v in rates.items()})

    return SourceResult(
        name="fx_latest", df=df, source_url=resp.url, fetched_at=resp.fetched_at,
        sha256=resp.sha256, from_cache=resp.from_cache,
        notes={"base": base_ccy, "as_of": str(payload.get("date"))},
    )


def fetch_series(
    *, start: date | str, end: date | str | None = None,
    base_ccy: str = "USD", symbols: str = "INR",
    cache_dir: Path | None = None, ttl_days: int = 7,
) -> SourceResult:
    """Fetch a daily rate time series over a date range."""
    start_s = start.isoformat() if isinstance(start, date) else str(start)
    end_s = end.isoformat() if isinstance(end, date) else (str(end) if end else "")
    path = f"{BASE}/{start_s}..{end_s}" if end_s else f"{BASE}/{start_s}.."

    resp = get_bytes(path, params={"from": base_ccy, "to": symbols},
                     cache_dir=cache_dir, ttl_days=ttl_days)
    payload = resp.json()
    rates = payload.get("rates")
    if not isinstance(rates, dict):
        raise SchemaError(f"Frankfurter series payload has no 'rates' object: {resp.url}")

    records = [
        {"date": pd.to_datetime(day), "base": base_ccy, "quote": ccy, "rate": float(val)}
        for day, quotes in rates.items()
        for ccy, val in (quotes or {}).items()
    ]
    df = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    log.info("fx series %s->%s: %d observations", base_ccy, symbols, len(df))

    return SourceResult(
        name=f"fx_{base_ccy.lower()}_{symbols.lower().replace(',', '_')}",
        df=df, source_url=resp.url, fetched_at=resp.fetched_at,
        sha256=resp.sha256, from_cache=resp.from_cache,
        notes={"base": base_ccy, "symbols": symbols, "start": start_s, "end": end_s},
    )


def usd_inr_rate(df: pd.DataFrame) -> float:
    """Latest USD->INR rate from a fetched frame, for converting FDIC figures."""
    inr = df[df["quote"] == "INR"].dropna(subset=["rate"])
    if inr.empty:
        raise SchemaError("no INR rate present in the FX frame")
    return float(inr.loc[inr["date"].idxmax(), "rate"])

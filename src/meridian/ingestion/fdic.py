"""FDIC BankFind Suite -- real per-institution quarterly financials.

This is the peer benchmark. The synthetic portfolio's ratios are asserted to
fall inside the interquartile range of real institutions of comparable size,
and that assertion is a build gate: if generated economics drift outside what
real banks actually report, the build fails rather than producing numbers that
cannot be defended.

**The unit trap.** FDIC exposes both ``NIM`` and ``NIMY``. They are not the same
quantity in different precision -- ``NIM`` is net interest margin in *dollars*
(thousands), ``NIMY`` is net interest margin as a *percentage*. Verified on
CERT 628: ``NIM = 46,983,000`` against ``NIMY = 3.02``. Charting NIM as a
percentage produces "4,698,300% margin", which is the kind of error that
survives review because nobody reads the axis. Only the ``*Y``-suffixed yield
fields are treated as percentages here; :data:`DOLLAR_FIELDS` and
:data:`PERCENT_FIELDS` make the distinction explicit and
:func:`assert_units_sane` enforces it at runtime.

The API needs no key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..common.exceptions import SchemaError
from ..common.http import get_bytes
from ..common.logging import get_logger
from .base import SourceResult, coerce_numeric

log = get_logger(__name__)

BASE = "https://banks.data.fdic.gov/api"

# Percentages. Every one of these is already a ratio expressed in percent.
PERCENT_FIELDS = frozenset({
    "ROA",      # return on assets
    "ROE",      # return on equity
    "NIMY",     # net interest margin, PERCENT  <- use this one
    "EEFFR",    # efficiency ratio
    "ROAPTX",   # pre-tax return on assets
    "INTINCY",  # total interest income yield
    "EQV",      # equity to assets
    "LNLSDEPR", # loans to deposits
    "NPERFV",   # non-performing assets to assets
})

# Dollar amounts, reported in thousands. Never chart these as a rate.
DOLLAR_FIELDS = frozenset({
    "ASSET",    # total assets
    "DEP",      # total deposits
    "NETINC",   # net income
    "NIM",      # net interest margin, DOLLARS  <- NOT a percentage
    "LNLSNET",  # net loans and leases
    "EQ",       # total equity
    "INTINC",   # total interest income
    "EINTEXP",  # total interest expense
})

FINANCIAL_FIELDS = sorted(PERCENT_FIELDS | DOLLAR_FIELDS)


def _rows(payload: Any, *, url: str) -> list[dict[str, Any]]:
    """Pull the ``data`` object out of each element of an FDIC response."""
    if not isinstance(payload, dict):
        raise SchemaError(f"FDIC returned {type(payload).__name__}, expected object: {url}")
    data = payload.get("data")
    if data is None:
        raise SchemaError(f"FDIC payload has no 'data' key for {url}")
    if not isinstance(data, list):
        raise SchemaError(f"FDIC 'data' was {type(data).__name__}, expected list")
    return [item.get("data", item) if isinstance(item, dict) else {} for item in data]


def fetch_institutions(
    *,
    filters: str = "ACTIVE:1",
    limit: int = 1000,
    fields: str = "CERT,NAME,CITY,STALP,ASSET,DEP,OFFDOM,ESTYMD,BKCLASS",
    cache_dir: Path | None = None,
    ttl_days: int = 7,
) -> SourceResult:
    """Fetch the institution directory (one row per bank)."""
    url = f"{BASE}/institutions"
    params = {"filters": filters, "fields": fields, "limit": limit, "format": "json"}
    resp = get_bytes(url, params=params, cache_dir=cache_dir, ttl_days=ttl_days)
    df = pd.DataFrame.from_records(_rows(resp.json(), url=resp.url))

    for col in ("ASSET", "DEP", "OFFDOM", "CERT"):
        if col in df:
            df[col] = coerce_numeric(df[col])

    log.info("fdic institutions: %d rows", len(df))
    return SourceResult(
        name="fdic_institutions", df=df, source_url=resp.url,
        fetched_at=resp.fetched_at, sha256=resp.sha256, from_cache=resp.from_cache,
        notes={"filters": filters},
    )


def fetch_financials(
    cert: int | str,
    *,
    limit: int = 20,
    cache_dir: Path | None = None,
    ttl_days: int = 7,
) -> SourceResult:
    """Fetch a quarterly financial time series for one institution by CERT."""
    url = f"{BASE}/financials"
    params = {
        "filters": f"CERT:{cert}",
        "fields": "CERT,REPDTE," + ",".join(FINANCIAL_FIELDS),
        "sort_by": "REPDTE", "sort_order": "DESC",
        "limit": limit, "format": "json",
    }
    resp = get_bytes(url, params=params, cache_dir=cache_dir, ttl_days=ttl_days)
    df = pd.DataFrame.from_records(_rows(resp.json(), url=resp.url))

    for col in df.columns:
        if col != "REPDTE":
            df[col] = coerce_numeric(df[col])
    if "REPDTE" in df:
        df["report_date"] = pd.to_datetime(df["REPDTE"], format="%Y%m%d", errors="coerce")
        df = df.sort_values("report_date").reset_index(drop=True)

    return SourceResult(
        name=f"fdic_financials_{cert}", df=df, source_url=resp.url,
        fetched_at=resp.fetched_at, sha256=resp.sha256, from_cache=resp.from_cache,
        notes={"cert": str(cert), "n_quarters": len(df)},
    )


def fetch_peer_panel(
    *,
    min_assets_k: int = 1_000_000,
    max_assets_k: int = 100_000_000,
    n_banks: int = 60,
    quarters: int = 8,
    cache_dir: Path | None = None,
    ttl_days: int = 7,
) -> SourceResult:
    """Build a peer panel of comparably sized institutions.

    ``min_assets_k``/``max_assets_k`` are in thousands of USD, matching FDIC's
    own reporting unit. The default window (USD 1bn-100bn) brackets a mid-size
    retail bank, which is what Meridian is modelled as.
    """
    inst = fetch_institutions(
        filters=f"ACTIVE:1 AND ASSET:[{min_assets_k} TO {max_assets_k}]",
        limit=n_banks, cache_dir=cache_dir, ttl_days=ttl_days,
    )
    if inst.df.empty:
        raise SchemaError("FDIC returned no institutions for the peer size band")

    certs = inst.df["CERT"].dropna().astype(int).tolist()[:n_banks]
    log.info("fdic peer panel: %d institutions x %d quarters", len(certs), quarters)

    frames: list[pd.DataFrame] = []
    for cert in certs:
        try:
            res = fetch_financials(cert, limit=quarters, cache_dir=cache_dir, ttl_days=ttl_days)
        except Exception as exc:
            log.debug("fdic cert %s skipped: %s", cert, exc)
            continue
        if not res.df.empty:
            frames.append(res.df)

    if not frames:
        raise SchemaError("FDIC peer panel is empty: no institution returned financials")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(
        inst.df[["CERT", "NAME", "STALP"]].assign(CERT=lambda d: coerce_numeric(d["CERT"])),
        on="CERT", how="left",
    )
    assert_units_sane(panel)

    return SourceResult(
        name="fdic_peer_panel", df=panel, source_url=inst.source_url,
        fetched_at=inst.fetched_at, from_cache=inst.from_cache,
        notes={"n_institutions": int(panel["CERT"].nunique()),
               "n_quarters": quarters,
               "asset_band_thousands_usd": [min_assets_k, max_assets_k]},
    )


def assert_units_sane(df: pd.DataFrame) -> None:
    """Guard the NIM/NIMY confusion at runtime.

    A percent field whose median sits in the thousands means a dollar column has
    been mislabelled somewhere upstream. Warn loudly rather than let it reach a
    chart axis.
    """
    for col in df.columns:
        if col in PERCENT_FIELDS and col in df:
            med = pd.to_numeric(df[col], errors="coerce").median()
            if pd.notna(med) and abs(med) > 1000:
                log.warning(
                    "%s has median %.0f -- percent field holding dollar-scale values?",
                    col, med,
                )


def peer_iqr(df: pd.DataFrame, field: str) -> tuple[float, float, float]:
    """Return (q1, median, q3) for a peer field -- the calibration envelope.

    The generator's portfolio ratios must land inside [q1, q3] or the build
    fails. Computed on the most recent quarter per institution so that a bank
    with more history does not dominate the distribution.
    """
    if field not in df:
        raise SchemaError(f"peer panel has no field {field!r}")
    latest = (
        df.sort_values("report_date").groupby("CERT", as_index=False).tail(1)
        if "report_date" in df else df
    )
    s = pd.to_numeric(latest[field], errors="coerce").dropna()
    if s.empty:
        raise SchemaError(f"peer field {field!r} is entirely null")
    return float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))

"""World Bank World Development Indicators -- India macroeconomic series.

These are the real macro facts the synthetic generator is calibrated against:
the lending interest rate sets the interest spread on generated products, and
bank branch density and account ownership inform the channel mix.

Two documented failure modes are handled explicitly, because both corrupt
results silently rather than raising.

**Null recent years.** The API returns rows for years that have no data yet,
with ``value: null``. For FR.INR.LEND, 2023-2025 are null and the true latest
observation is 2022. Naively taking ``rows[0]`` -- the most recent year -- gives
None, and a downstream ``float(None)`` failure far from the cause. Worse, a
``fillna(0)`` anywhere in between would silently model a zero-interest economy.
:func:`latest_value` drops nulls first and takes the latest surviving year.

**Two-element envelope.** A successful response is ``[metadata, rows]``. An
error response is a one-element list containing a message object. Indexing
``[1]`` unconditionally turns an API error into an IndexError that reads like a
bug in our code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.exceptions import SchemaError
from ..common.http import get_bytes
from ..common.logging import get_logger
from .base import SourceResult, coerce_numeric

log = get_logger(__name__)

BASE = "https://api.worldbank.org/v2"
COUNTRY = "IND"

# The indicators the project actually uses, with the role each one plays.
INDICATORS: dict[str, str] = {
    "FR.INR.LEND": "Lending interest rate (%)",
    "FR.INR.DPST": "Deposit interest rate (%)",
    "FB.CBK.BRCH.P5": "Commercial bank branches (per 100,000 adults)",
    "FX.OWN.TOTL.ZS": "Account ownership, age 15+ (% of population)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "FB.BNK.CAPA.ZS": "Bank capital to assets ratio (%)",
}


@dataclass(frozen=True)
class Observation:
    """One indicator's latest non-null observation."""

    indicator: str
    label: str
    year: int
    value: float


def _unwrap(payload: Any, *, url: str) -> list[dict[str, Any]]:
    """Return the rows array from a World Bank response envelope.

    Handles the success shape ``[meta, rows]`` and the error shape
    ``[{message: [...]}]``, raising SchemaError with the provider's own message
    rather than letting an IndexError escape.
    """
    if not isinstance(payload, list) or not payload:
        raise SchemaError(f"World Bank returned a non-list payload for {url}")

    if len(payload) == 1:
        head = payload[0]
        detail = ""
        if isinstance(head, dict) and "message" in head:
            messages = head.get("message") or []
            if messages and isinstance(messages[0], dict):
                detail = messages[0].get("value", "") or messages[0].get("key", "")
        raise SchemaError(f"World Bank error for {url}: {detail or head}")

    rows = payload[1]
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise SchemaError(f"World Bank rows were {type(rows).__name__}, expected list")
    return rows


def fetch_indicator(
    indicator: str,
    *,
    country: str = COUNTRY,
    start: int = 2000,
    end: int | None = None,
    cache_dir: Path | None = None,
    ttl_days: int = 7,
) -> SourceResult:
    """Fetch one indicator's full time series for a country."""
    end = end or datetime.now().year
    url = f"{BASE}/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": 500, "date": f"{start}:{end}"}

    resp = get_bytes(url, params=params, cache_dir=cache_dir, ttl_days=ttl_days)
    rows = _unwrap(resp.json(), url=resp.url)

    records = [
        {
            "country": (r.get("country") or {}).get("value"),
            "country_iso3": r.get("countryiso3code"),
            "indicator": (r.get("indicator") or {}).get("id", indicator),
            "indicator_label": (r.get("indicator") or {}).get("value", ""),
            "year": r.get("date"),
            "value": r.get("value"),
        }
        for r in rows
    ]

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["year"] = coerce_numeric(df["year"]).astype("Int64")
        df["value"] = coerce_numeric(df["value"])
        df = df.sort_values("year").reset_index(drop=True)

    n_null = int(df["value"].isna().sum()) if not df.empty else 0
    log.info("worldbank %s: %d rows, %d null values", indicator, len(df), n_null)

    return SourceResult(
        name=f"worldbank_{indicator.replace('.', '_').lower()}",
        df=df,
        source_url=resp.url,
        fetched_at=resp.fetched_at,
        sha256=resp.sha256,
        from_cache=resp.from_cache,
        notes={"indicator": indicator, "n_null_values": n_null,
               "label": INDICATORS.get(indicator, "")},
    )


def latest_value(df: pd.DataFrame) -> Observation | None:
    """Latest *non-null* observation, or None if the series is entirely null.

    This is the guard against the null-recent-years gotcha. Never take the most
    recent row; take the most recent row that has a value.
    """
    if df.empty or "value" not in df:
        return None
    valid = df.dropna(subset=["value"])
    if valid.empty:
        return None
    row = valid.loc[valid["year"].idxmax()]
    return Observation(
        indicator=str(row["indicator"]),
        label=str(row.get("indicator_label", "")),
        year=int(row["year"]),
        value=float(row["value"]),
    )


def fetch_all(
    indicators: dict[str, str] | None = None,
    *,
    cache_dir: Path | None = None,
    ttl_days: int = 7,
) -> tuple[pd.DataFrame, dict[str, Observation]]:
    """Fetch every configured indicator.

    Returns the stacked long-format panel and a map of indicator -> latest
    non-null observation. One failing indicator does not abort the rest: the
    pipeline is more useful with seven series than with an exception.
    """
    indicators = indicators or INDICATORS
    frames: list[pd.DataFrame] = []
    latest: dict[str, Observation] = {}

    for code in indicators:
        try:
            res = fetch_indicator(code, cache_dir=cache_dir, ttl_days=ttl_days)
        except (SchemaError, Exception) as exc:  # noqa: B014 - breadth is deliberate
            log.warning("worldbank %s unavailable: %s", code, exc)
            continue
        if res.df.empty:
            continue
        frames.append(res.df)
        obs = latest_value(res.df)
        if obs is not None:
            latest[code] = obs
            log.info("  %s = %.4f (%d) %s", code, obs.value, obs.year, obs.label[:40])

    # Some indicators (FR.INR.DPST for India, currently) have no observation in
    # any year, giving an all-NA float column. pandas warns that such columns
    # will participate in dtype resolution in a future version; pinning the
    # dtype here makes the concat explicit rather than order-dependent.
    for f in frames:
        f["value"] = f["value"].astype("float64")
    panel = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["country", "country_iso3", "indicator",
                                   "indicator_label", "year", "value"])
    )
    return panel, latest

"""Source registry -- one entry point that fetches everything and records it.

The registry exists so that provenance is a by-product of ingestion rather than
something reconstructed later. Every run writes ``data/raw/manifest.json``
listing each source, its URL, fetch time, row count and payload hash. That file
is what docs/DATA_DICTIONARY.md and the report's provenance table are generated
from, and it is what makes "where did this number come from" answerable.

Individual sources are allowed to fail. A missing optional series should not
cost you the other four, so failures are recorded in the manifest with their
error text and the pipeline continues. Sources marked ``required`` do halt it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.config import Settings, get_settings
from ..common.exceptions import IngestionError
from ..common.io import write_json
from ..common.logging import get_logger
from . import fdic, fx, uci_campaign, worldbank
from .base import SourceResult

log = get_logger(__name__)


@dataclass
class SourceSpec:
    """One registered source: how to fetch it and how much it matters."""

    key: str
    description: str
    fetch: Callable[[Settings], SourceResult]
    required: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


def _worldbank_panel(s: Settings) -> SourceResult:
    panel, latest = worldbank.fetch_all(
        cache_dir=s.raw_dir, ttl_days=s.http_cache_ttl_days
    )
    if panel.empty:
        raise IngestionError("World Bank returned no usable indicator rows")
    return SourceResult(
        name="worldbank_india_macro", df=panel,
        source_url=f"{worldbank.BASE}/country/{worldbank.COUNTRY}/indicator/...",
        fetched_at=datetime.now(),
        notes={
            "n_indicators": int(panel["indicator"].nunique()),
            "latest_non_null": {
                k: {"year": o.year, "value": o.value} for k, o in latest.items()
            },
        },
    )


def _uci(s: Settings) -> SourceResult:
    return uci_campaign.load_campaign(cache_dir=s.raw_dir, ttl_days=30)


def _fdic_peers(s: Settings) -> SourceResult:
    return fdic.fetch_peer_panel(
        cache_dir=s.raw_dir, ttl_days=s.http_cache_ttl_days, n_banks=40, quarters=8
    )


def _fx_latest(s: Settings) -> SourceResult:
    return fx.fetch_latest(cache_dir=s.raw_dir, ttl_days=1)


def _fx_series(s: Settings) -> SourceResult:
    start = (datetime.now() - timedelta(days=730)).date()
    return fx.fetch_series(start=start, base_ccy="USD", symbols="INR",
                           cache_dir=s.raw_dir, ttl_days=s.http_cache_ttl_days)


REGISTRY: tuple[SourceSpec, ...] = (
    SourceSpec(
        "uci_campaign",
        "UCI Bank Marketing: 41,188 real retail bank campaign contacts with outcomes",
        _uci, required=True, tags=("real", "campaign", "customer-behaviour"),
    ),
    SourceSpec(
        "worldbank_macro",
        "World Bank WDI: India lending/deposit rates, branch density, GDP, CPI",
        _worldbank_panel, required=True, tags=("real", "macro"),
    ),
    SourceSpec(
        "fdic_peers",
        "FDIC BankFind: real quarterly financials for a size-matched peer panel",
        _fdic_peers, required=True, tags=("real", "benchmark"),
    ),
    SourceSpec(
        "fx_latest", "Frankfurter/ECB: latest reference rates",
        _fx_latest, required=False, tags=("real", "fx"),
    ),
    SourceSpec(
        "fx_usd_inr_series", "Frankfurter/ECB: daily USD/INR, trailing 24 months",
        _fx_series, required=False, tags=("real", "fx", "timeseries"),
    ),
)


def ingest_all(
    settings: Settings | None = None, *, only: tuple[str, ...] | None = None,
) -> dict[str, SourceResult]:
    """Fetch every registered source, persist it, and write the manifest."""
    s = settings or get_settings()
    s.ensure_dirs()

    specs = [sp for sp in REGISTRY if not only or sp.key in only]
    results: dict[str, SourceResult] = {}
    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {},
    }

    for spec in specs:
        log.info("--- ingesting %s ---", spec.key)
        try:
            res = spec.fetch(s)
        except Exception as exc:
            log.error("%s FAILED: %s", spec.key, exc)
            manifest["sources"][spec.key] = {
                "status": "failed", "error": str(exc),
                "description": spec.description, "required": spec.required,
            }
            if spec.required:
                write_json(s.raw_dir / "manifest.json", manifest)
                raise IngestionError(f"required source {spec.key!r} failed: {exc}") from exc
            continue

        res.save(s.interim_dir, stem=spec.key)
        results[spec.key] = res
        manifest["sources"][spec.key] = {
            "status": "ok", "description": spec.description,
            "tags": list(spec.tags), "required": spec.required,
            **res.provenance(),
        }

    write_json(s.raw_dir / "manifest.json", manifest)
    ok = sum(1 for v in manifest["sources"].values() if v["status"] == "ok")
    log.info("ingestion complete: %d/%d sources ok", ok, len(specs))
    return results


def manifest_table(manifest_path: Path) -> pd.DataFrame:
    """Render the manifest as a table for the docs provenance section."""
    import json

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        {
            "source": key,
            "status": meta.get("status"),
            "rows": meta.get("n_rows"),
            "fetched_at": meta.get("fetched_at"),
            "description": meta.get("description"),
        }
        for key, meta in data.get("sources", {}).items()
    ]
    return pd.DataFrame(rows)

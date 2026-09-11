"""Shared scaffolding for ingestion modules.

Every source in this package returns a :class:`SourceResult`: a DataFrame plus
the provenance needed to defend the number in a report -- where it came from,
when it was fetched, and the hash of the bytes it was parsed from. A figure in
the final deck should be traceable back to a URL and a timestamp without
archaeology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.io import write_json
from ..common.logging import get_logger

log = get_logger(__name__)


@dataclass
class SourceResult:
    """A parsed dataset plus its provenance record."""

    name: str
    df: pd.DataFrame
    source_url: str
    fetched_at: datetime
    sha256: str = ""
    from_cache: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    def provenance(self) -> dict[str, Any]:
        """A JSON-serialisable record for docs/DATA_DICTIONARY.md and the report."""
        return {
            "name": self.name,
            "source_url": self.source_url,
            "fetched_at": self.fetched_at.isoformat(timespec="seconds"),
            "from_cache": self.from_cache,
            "sha256": self.sha256,
            "n_rows": int(self.n_rows),
            "n_cols": int(self.df.shape[1]),
            "columns": list(self.df.columns),
            **self.notes,
        }

    def save(self, out_dir: Path, *, stem: str | None = None) -> Path:
        """Write the frame as CSV alongside a sidecar provenance JSON."""
        stem = stem or self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{stem}.csv"
        self.df.to_csv(csv_path, index=False)
        write_json(out_dir / f"{stem}.provenance.json", self.provenance())
        log.info("saved %s: %d rows -> %s", self.name, self.n_rows, csv_path.name)
        return csv_path


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Parse a possibly-dirty numeric column without raising.

    Public APIs return numbers as strings, with thousands separators, or as
    empty strings for "no data". Anything unparseable becomes NaN, which the
    validation layer then reports on rather than silently absorbing.
    """
    if s.dtype.kind in "if":
        return s
    cleaned = (
        s.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({"": None, "-": None, "N/A": None, "NA": None, "null": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")

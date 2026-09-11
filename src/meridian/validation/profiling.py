"""Column profiling: the shape of a table before any analysis touches it.

Profiling answers the questions you should ask before trusting a column --
how much is missing, how many distinct values, what the tails look like -- and
does it uniformly so the answers are comparable across tables and across runs.

The profile feeds three consumers: the validation report's overview section,
docs/DATA_DICTIONARY.md, and the dashboard's Data Quality page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..common.logging import get_logger

log = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Summary statistics for a single column."""

    name: str
    dtype: str
    n_rows: int
    n_missing: int
    pct_missing: float
    n_distinct: int
    is_numeric: bool
    is_constant: bool
    # Numeric only
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    p01: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    p99: float | None = None
    maximum: float | None = None
    skew: float | None = None
    n_zero: int | None = None
    n_negative: int | None = None
    n_outliers_iqr: int | None = None
    # Categorical only
    top_values: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_column(s: pd.Series, name: str | None = None) -> ColumnProfile:
    """Profile one column."""
    name = name or str(s.name)
    n = len(s)
    n_missing = int(s.isna().sum())
    is_numeric = pd.api.types.is_numeric_dtype(s)
    non_null = s.dropna()

    prof = ColumnProfile(
        name=name, dtype=str(s.dtype), n_rows=n, n_missing=n_missing,
        pct_missing=round(n_missing / n, 6) if n else 0.0,
        n_distinct=int(non_null.nunique()), is_numeric=is_numeric,
        is_constant=bool(non_null.nunique() <= 1 and len(non_null) > 0),
    )

    if is_numeric and not non_null.empty:
        q = non_null.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        iqr = float(q[0.75] - q[0.25])
        lo, hi = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
        prof.mean = float(non_null.mean())
        prof.std = float(non_null.std()) if len(non_null) > 1 else 0.0
        prof.minimum = float(non_null.min())
        prof.p01, prof.p25 = float(q[0.01]), float(q[0.25])
        prof.median, prof.p75, prof.p99 = float(q[0.5]), float(q[0.75]), float(q[0.99])
        prof.maximum = float(non_null.max())
        prof.skew = float(non_null.skew()) if len(non_null) > 2 else 0.0
        prof.n_zero = int((non_null == 0).sum())
        prof.n_negative = int((non_null < 0).sum())
        prof.n_outliers_iqr = int(((non_null < lo) | (non_null > hi)).sum())
    elif not non_null.empty:
        prof.top_values = {
            str(k): int(v) for k, v in non_null.value_counts().head(10).items()
        }

    return prof


def profile_table(df: pd.DataFrame, *, table: str = "") -> pd.DataFrame:
    """Profile every column, returned as a tidy frame."""
    rows = [profile_column(df[c], c).to_dict() for c in df.columns]
    out = pd.DataFrame(rows)
    if table:
        out.insert(0, "table", table)
    log.info("profiled %s: %d columns x %d rows", table or "table", df.shape[1], len(df))
    return out


def missingness_matrix(df: pd.DataFrame, *, max_cols: int = 40) -> pd.DataFrame:
    """Pairwise co-missingness between columns.

    Columns that go missing together usually share an upstream cause -- one
    failed join, one optional API block -- so the pattern is more diagnostic
    than the per-column rate alone.
    """
    cols = [c for c in df.columns if df[c].isna().any()][:max_cols]
    if not cols:
        return pd.DataFrame()
    miss = df[cols].isna()
    n = len(df)
    return pd.DataFrame(
        {a: {b: round(float((miss[a] & miss[b]).sum()) / n, 6) for b in cols} for a in cols}
    )


def correlation_pairs(df: pd.DataFrame, *, threshold: float = 0.8,
                      method: str = "pearson") -> pd.DataFrame:
    """Numeric column pairs correlated above a threshold.

    Near-perfect correlation between two columns usually means one is derived
    from the other -- which matters before either is fed to a model.
    """
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["column_a", "column_b", "correlation"])
    corr = num.corr(method=method, numeric_only=True)
    pairs = [
        {"column_a": a, "column_b": b, "correlation": round(float(corr.loc[a, b]), 6)}
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1:]
        if pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) >= threshold
    ]
    return (
        pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)
        if pairs else pd.DataFrame(columns=["column_a", "column_b", "correlation"])
    )

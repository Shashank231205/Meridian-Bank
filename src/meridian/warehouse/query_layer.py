"""Run the analytical SQL layer and persist the results.

Named query_layer rather than queries so it does not collide with the
``queries/`` directory holding the .sql files themselves.

The queries live as .sql files rather than as strings inside Python. That is
deliberate: they can be opened in any SQL client, run against the database
directly, and reviewed by someone who does not read Python -- which is most of
the audience for a banking analytics project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..common.config import Settings, get_settings
from ..common.exceptions import WarehouseError
from ..common.logging import get_logger
from .db import query_conn, session

log = get_logger(__name__)

QUERY_DIR = Path(__file__).parent / "queries"


@dataclass(frozen=True)
class QuerySpec:
    """One analytical query and what it is for."""

    key: str
    filename: str
    title: str
    grain: str
    description: str


QUERIES: tuple[QuerySpec, ...] = (
    QuerySpec("q01", "q01_customer_360.sql", "Customer 360",
              "one row per customer",
              "Joined view of demographics, holdings, engagement and outcome."),
    QuerySpec("q02", "q02_rfm_segmentation.sql", "RFM Segmentation",
              "one row per customer",
              "Recency/Frequency/Monetary quintiles and named segments."),
    QuerySpec("q03", "q03_revenue_trend.sql", "Revenue Trend",
              "one row per month",
              "Monthly revenue with MoM, YoY and moving averages."),
    QuerySpec("q04", "q04_product_pareto.sql", "Product Pareto",
              "one row per product",
              "Revenue concentration by product with running totals."),
    QuerySpec("q05", "q05_cohort_retention.sql", "Cohort Retention",
              "one row per acquisition cohort",
              "Retention triangle by months since acquisition."),
    QuerySpec("q06", "q06_churn_risk.sql", "Churn Risk Flags",
              "one row per active customer",
              "Transparent rule-based risk score and revenue at risk."),
    QuerySpec("q07", "q07_campaign_funnel.sql", "Campaign Funnel and ROI",
              "one row per campaign",
              "Contacts, conversion, cost per acquisition and ROI."),
    QuerySpec("q08", "q08_channel_rollup.sql", "Channel Rollup",
              "one row per acquisition x transaction channel",
              "Channel migration and digital share, with subtotals."),
    QuerySpec("q09", "q09_clv_deciles.sql", "CLV Deciles",
              "one row per customer",
              "Lifetime value with deciles and cumulative share."),
    QuerySpec("q10", "q10_executive_kpis.sql", "Executive KPIs",
              "one row per KPI",
              "The eight headline dashboard metrics with prior comparisons."),
)

QUERIES_BY_KEY = {q.key: q for q in QUERIES}


def load_sql(key: str) -> str:
    """Read one query's SQL text."""
    spec = QUERIES_BY_KEY.get(key)
    if spec is None:
        raise WarehouseError(f"unknown query {key!r}; known: {list(QUERIES_BY_KEY)}")
    path = QUERY_DIR / spec.filename
    if not path.is_file():
        raise WarehouseError(f"query file missing: {path}")
    return path.read_text(encoding="utf-8")


def run_query(
    key: str, settings: Settings | None = None, *, limit: int | None = None,
) -> pd.DataFrame:
    """Execute one query against the warehouse."""
    s = settings or get_settings()
    sql = load_sql(key)
    if limit:
        sql = f"SELECT * FROM (\n{sql.rstrip().rstrip(';')}\n) LIMIT {int(limit)}"
    with session(s.db_path, read_only=True) as conn:
        df = query_conn(conn, sql)
    log.info("%s %s: %s rows", key, QUERIES_BY_KEY[key].title, f"{len(df):,}")
    return df


def run_all(
    settings: Settings | None = None, *, save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute every query, optionally writing each to outputs/tables/."""
    s = settings or get_settings()
    if not s.db_path.is_file():
        raise WarehouseError(
            f"warehouse not found at {s.db_path}. Build it before running queries."
        )

    results: dict[str, pd.DataFrame] = {}
    for spec in QUERIES:
        df = run_query(spec.key, s)
        results[spec.key] = df
        if save:
            out = s.tables_dir / f"{spec.key}_{spec.filename[4:-4]}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
    return results


def catalogue() -> pd.DataFrame:
    """The query catalogue, for the docs and the report."""
    return pd.DataFrame([
        {"key": q.key, "title": q.title, "grain": q.grain,
         "description": q.description, "file": q.filename}
        for q in QUERIES
    ])

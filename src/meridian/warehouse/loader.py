"""Build the warehouse: create the schema, load dimensions, then facts.

Load order is dictated by the foreign keys. Dimensions first, because a fact row
cannot reference a dimension row that does not exist yet; and within the facts,
anything depending on a surrogate key waits for the dimension that mints it.

Natural keys (``customer_id``, ``product_code``) are mapped to surrogate keys at
load time. Surrogates are used rather than natural keys throughout the facts
because they are compact integers, and because a dimension that later needs
history can grow a second row for the same natural key without every fact having
to change.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..common.config import Settings, get_settings
from ..common.exceptions import WarehouseError
from ..common.logging import get_logger
from .db import execute_script, foreign_key_violations, session, table_counts

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

CHANNELS: tuple[tuple[str, str, int, int], ...] = (
    ("mobile", "Mobile App", 1, 0),
    ("internet", "Internet Banking", 1, 0),
    ("upi", "UPI", 1, 0),
    ("atm", "ATM", 0, 0),
    ("pos", "Point of Sale", 0, 0),
    ("branch", "Branch Counter", 0, 1),
    ("telesales", "Telesales", 0, 1),
    ("dsa", "Direct Sales Agent", 0, 1),
    ("partner", "Partner", 0, 1),
    ("digital", "Digital Acquisition", 1, 0),
)

REGIONS: dict[str, str] = {
    "Maharashtra": "West", "Gujarat": "West", "Rajasthan": "West",
    "Delhi": "North", "Punjab": "North", "Uttar Pradesh": "North",
    "Karnataka": "South", "Tamil Nadu": "South", "Telangana": "South",
    "Kerala": "South",
    "West Bengal": "East", "Madhya Pradesh": "Central",
}


def build_warehouse(
    settings: Settings | None = None, *, rebuild: bool = True,
) -> Path:
    """Create and populate the warehouse from the generated and real data."""
    s = settings or get_settings()
    db_path = s.db_path

    if rebuild and db_path.exists():
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            sidecar.unlink(missing_ok=True)
        log.info("dropped existing warehouse at %s", db_path.name)

    syn = s.synthetic_dir
    required = ["dim_customer", "dim_product", "dim_campaign", "holdings",
                "fact_monthly_account", "fact_transaction", "fact_campaign_contact",
                "churn_outcomes"]
    missing = [t for t in required if not (syn / f"{t}.csv").is_file()]
    if missing:
        raise WarehouseError(
            f"generated data missing: {missing}. Run generation before loading."
        )

    with session(db_path) as conn:
        log.info("creating schema")
        execute_script(conn, SCHEMA_PATH.read_text(encoding="utf-8"))

        customers = pd.read_csv(syn / "dim_customer.csv")
        products = pd.read_csv(syn / "dim_product.csv")
        campaigns = pd.read_csv(syn / "dim_campaign.csv")

        _load_dim_date(conn, customers)
        cust_keys = _load_dim_customer(conn, customers)
        prod_keys = _load_dim_product(conn, products)
        chan_keys = _load_dim_channel(conn)
        _load_dim_branch(conn, customers)
        camp_keys = _load_dim_campaign(conn, campaigns)

        _load_fact_monthly_account(conn, syn, cust_keys, prod_keys)
        _load_fact_transaction(conn, syn, cust_keys, chan_keys)
        _load_fact_campaign_contact(conn, syn, cust_keys, camp_keys)
        _load_fact_customer_snapshot(conn, syn, cust_keys)

        _load_reference(conn, s)

    counts = table_counts(db_path)
    log.info("warehouse built: %s", {k: f"{v:,}" for k, v in counts.items() if v})

    violations = foreign_key_violations(db_path)
    if not violations.empty:
        raise WarehouseError(
            f"{len(violations)} foreign key violation(s) after load:\n{violations.head()}"
        )
    log.info("foreign key check: clean")
    return db_path


# --- dimensions ------------------------------------------------------------

def _date_key(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s)
    return d.dt.year * 10000 + d.dt.month * 100 + d.dt.day


def _load_dim_date(conn: sqlite3.Connection, customers: pd.DataFrame) -> None:
    """Populate the date dimension across the full span of the data."""
    start = pd.to_datetime(customers["acquired_date"]).min().normalize()
    # Extend a year past the data so late-arriving facts still join.
    end = pd.Timestamp.now().normalize() + pd.DateOffset(years=1)
    dates = pd.date_range(start.replace(day=1), end, freq="D")

    df = pd.DataFrame({"full_date": dates})
    df["date_key"] = df["full_date"].dt.year * 10000 + \
        df["full_date"].dt.month * 100 + df["full_date"].dt.day
    df["year"] = df["full_date"].dt.year
    df["quarter"] = df["full_date"].dt.quarter
    df["month"] = df["full_date"].dt.month
    df["month_name"] = df["full_date"].dt.strftime("%B")
    df["day"] = df["full_date"].dt.day
    df["day_of_week"] = df["full_date"].dt.dayofweek
    df["day_name"] = df["full_date"].dt.strftime("%A")
    df["week_of_year"] = df["full_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["full_date"].dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = df["full_date"].dt.is_month_end.astype(int)
    df["month_start_date"] = df["full_date"].dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")

    # Indian fiscal year: April to March. FY2025-26 starts 2025-04-01.
    df["fiscal_year"] = df["year"].where(df["month"] >= 4, df["year"] - 1)
    df["fiscal_quarter"] = ((df["month"] - 4) % 12) // 3 + 1

    # Navratri through Diwali: the festive retail peak.
    df["is_festive_season"] = df["month"].isin([10, 11]).astype(int)

    df["full_date"] = df["full_date"].dt.strftime("%Y-%m-%d")
    df.to_sql("dim_date", conn, if_exists="append", index=False)
    log.info("dim_date: %s rows", f"{len(df):,}")


def _load_dim_customer(
    conn: sqlite3.Connection, customers: pd.DataFrame
) -> dict[str, int]:
    df = customers.copy()
    df["acquired_date"] = pd.to_datetime(df["acquired_date"]).dt.strftime("%Y-%m-%d")
    df["acquired_date_key"] = _date_key(customers["acquired_date"])

    cols = ["customer_id", "age", "age_band", "gender", "marital", "education",
            "job", "is_salaried", "city", "state", "segment", "annual_income_inr",
            "credit_score", "digital_engagement", "acquisition_channel",
            "acquired_date", "acquired_date_key", "tenure_months"]
    df[cols].to_sql("dim_customer", conn, if_exists="append", index=False)

    keys = {
        r["customer_id"]: r["customer_key"]
        for r in conn.execute("SELECT customer_id, customer_key FROM dim_customer")
    }
    log.info("dim_customer: %s rows", f"{len(keys):,}")
    return keys


def _load_dim_product(
    conn: sqlite3.Connection, products: pd.DataFrame
) -> dict[str, int]:
    cols = ["product_code", "product_name", "category", "interest_rate_pct",
            "rate_offset_pp", "annual_fee_inr", "min_income_inr", "is_anchor",
            "calibrated_from_lending_rate_pct"]
    products[cols].to_sql("dim_product", conn, if_exists="append", index=False)
    keys = {
        r["product_code"]: r["product_key"]
        for r in conn.execute("SELECT product_code, product_key FROM dim_product")
    }
    log.info("dim_product: %d rows", len(keys))
    return keys


def _load_dim_channel(conn: sqlite3.Connection) -> dict[str, int]:
    df = pd.DataFrame(
        CHANNELS, columns=["channel_code", "channel_name", "is_digital", "is_assisted"]
    )
    df.to_sql("dim_channel", conn, if_exists="append", index=False)
    keys = {
        r["channel_code"]: r["channel_key"]
        for r in conn.execute("SELECT channel_code, channel_key FROM dim_channel")
    }
    log.info("dim_channel: %d rows", len(keys))
    return keys


def _load_dim_branch(conn: sqlite3.Connection, customers: pd.DataFrame) -> None:
    """One branch per city in the customer base."""
    cities = customers[["city", "state"]].drop_duplicates().reset_index(drop=True)
    cities["branch_code"] = [f"BR{i + 1:03d}" for i in range(len(cities))]
    cities["branch_name"] = cities["city"] + " Main Branch"
    cities["region"] = cities["state"].map(REGIONS).fillna("Other")
    cities[["branch_code", "branch_name", "city", "state", "region"]].to_sql(
        "dim_branch", conn, if_exists="append", index=False
    )
    log.info("dim_branch: %d rows", len(cities))


def _load_dim_campaign(
    conn: sqlite3.Connection, campaigns: pd.DataFrame
) -> dict[str, int]:
    df = campaigns.copy()
    for col in ("start_date", "end_date"):
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    cols = ["campaign_id", "campaign_name", "product_code", "channel",
            "target_segment", "start_date", "end_date", "cost_per_contact_inr"]
    df[cols].to_sql("dim_campaign", conn, if_exists="append", index=False)
    keys = {
        r["campaign_id"]: r["campaign_key"]
        for r in conn.execute("SELECT campaign_id, campaign_key FROM dim_campaign")
    }
    log.info("dim_campaign: %d rows", len(keys))
    return keys


# --- facts -----------------------------------------------------------------

def _load_fact_monthly_account(
    conn: sqlite3.Connection, syn: Path,
    cust_keys: dict[str, int], prod_keys: dict[str, int],
) -> None:
    df = pd.read_csv(syn / "fact_monthly_account.csv")
    df["customer_key"] = df["customer_id"].map(cust_keys)
    df["product_key"] = df["product_code"].map(prod_keys)
    _assert_mapped(df, ["customer_key", "product_key"], "fact_monthly_account")

    df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m-%d")
    df["date_key"] = _date_key(df["month"])

    cols = ["customer_key", "product_key", "date_key", "month", "balance_inr",
            "interest_rate_pct", "interest_revenue_inr", "fee_revenue_inr",
            "total_revenue_inr", "account_age_months"]
    df[cols].to_sql("fact_monthly_account", conn, if_exists="append",
                    index=False, chunksize=50_000)
    log.info("fact_monthly_account: %s rows", f"{len(df):,}")


def _load_fact_transaction(
    conn: sqlite3.Connection, syn: Path,
    cust_keys: dict[str, int], chan_keys: dict[str, int],
) -> None:
    df = pd.read_csv(syn / "fact_transaction.csv")
    df["customer_key"] = df["customer_id"].map(cust_keys)
    df["channel_key"] = df["channel"].map(chan_keys)
    _assert_mapped(df, ["customer_key"], "fact_transaction")

    df["txn_date"] = pd.to_datetime(df["txn_date"]).dt.strftime("%Y-%m-%d")
    df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m-%d")
    df["date_key"] = _date_key(df["txn_date"])
    df["merchant_category"] = df["merchant_category"].fillna("")

    cols = ["txn_id", "customer_key", "date_key", "channel_key", "txn_date",
            "month", "txn_type", "direction", "amount_inr", "merchant_category"]
    df[cols].to_sql("fact_transaction", conn, if_exists="append",
                    index=False, chunksize=100_000)
    log.info("fact_transaction: %s rows", f"{len(df):,}")


def _load_fact_campaign_contact(
    conn: sqlite3.Connection, syn: Path,
    cust_keys: dict[str, int], camp_keys: dict[str, int],
) -> None:
    df = pd.read_csv(syn / "fact_campaign_contact.csv")
    df["customer_key"] = df["customer_id"].map(cust_keys)
    df["campaign_key"] = df["campaign_id"].map(camp_keys)
    _assert_mapped(df, ["customer_key", "campaign_key"], "fact_campaign_contact")

    df["contact_date"] = pd.to_datetime(df["contact_date"]).dt.strftime("%Y-%m-%d")
    df["date_key"] = _date_key(df["contact_date"])

    cols = ["contact_id", "campaign_key", "customer_key", "date_key",
            "contact_sequence", "contact_date", "channel", "prior_outcome",
            "n_prior_contacts", "converted", "cost_inr"]
    df[cols].to_sql("fact_campaign_contact", conn, if_exists="append", index=False)
    log.info("fact_campaign_contact: %s rows", f"{len(df):,}")


def _load_fact_customer_snapshot(
    conn: sqlite3.Connection, syn: Path, cust_keys: dict[str, int]
) -> None:
    """Load churn outcomes enriched with per-customer aggregates."""
    df = pd.read_csv(syn / "churn_outcomes.csv")
    df["customer_key"] = df["customer_id"].map(cust_keys)
    _assert_mapped(df, ["customer_key"], "fact_customer_snapshot")

    for col in ("acquired_date", "churn_date"):
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    df["churn_date"] = df["churn_date"].where(df["churned"] == 1, None)

    # Aggregates computed in SQL from the facts already loaded, so the snapshot
    # cannot disagree with the tables it summarises.
    agg = pd.read_sql_query("""
        SELECT f.customer_key,
               COUNT(DISTINCT f.product_key) AS n_products,
               SUM(f.balance_inr)            AS total_balance_inr,
               SUM(f.total_revenue_inr)      AS total_revenue_inr
        FROM fact_monthly_account f
        WHERE f.month = (SELECT MAX(month) FROM fact_monthly_account)
        GROUP BY f.customer_key
    """, conn)
    txn = pd.read_sql_query("""
        SELECT customer_key, COUNT(*) AS n_transactions
        FROM fact_transaction GROUP BY customer_key
    """, conn)

    df = df.merge(agg, on="customer_key", how="left").merge(
        txn, on="customer_key", how="left"
    )
    for col in ("n_products", "total_balance_inr", "total_revenue_inr",
                "n_transactions"):
        df[col] = df[col].fillna(0)

    cols = ["customer_key", "acquired_date", "churn_date", "churned", "censored",
            "observed_months", "n_products", "total_balance_inr",
            "total_revenue_inr", "n_transactions"]
    df[cols].to_sql("fact_customer_snapshot", conn, if_exists="append", index=False)
    log.info("fact_customer_snapshot: %s rows", f"{len(df):,}")


def _load_reference(conn: sqlite3.Connection, s: Settings) -> None:
    """Load the real source data, preserving provenance inside the warehouse."""
    wb_path = s.interim_dir / "worldbank_macro.csv"
    if wb_path.is_file():
        wb = pd.read_csv(wb_path)
        wb = wb.rename(columns={"country_iso3": "country_iso3"})
        cols = ["indicator", "indicator_label", "country_iso3", "year", "value"]
        wb = wb[[c for c in cols if c in wb.columns]].drop_duplicates(
            subset=["indicator", "country_iso3", "year"]
        )
        wb.to_sql("ref_macro_indicator", conn, if_exists="append", index=False)
        log.info("ref_macro_indicator: %s rows", f"{len(wb):,}")

    fdic_path = s.interim_dir / "fdic_peers.csv"
    if fdic_path.is_file():
        fd = pd.read_csv(fdic_path)
        out = pd.DataFrame({
            "cert": fd.get("CERT"),
            "institution_name": fd.get("NAME"),
            "state": fd.get("STALP"),
            "report_date": pd.to_datetime(fd.get("report_date")).dt.strftime("%Y-%m-%d"),
            "asset_thousands_usd": fd.get("ASSET"),
            "deposits_thousands_usd": fd.get("DEP"),
            "net_income_thousands_usd": fd.get("NETINC"),
            "nim_thousands_usd": fd.get("NIM"),
            "nimy_pct": fd.get("NIMY"),
            "roa_pct": fd.get("ROA"),
            "roe_pct": fd.get("ROE"),
            "efficiency_ratio_pct": fd.get("EEFFR"),
        }).dropna(subset=["cert", "report_date"]).drop_duplicates(
            subset=["cert", "report_date"]
        )
        out.to_sql("ref_peer_financials", conn, if_exists="append", index=False)
        log.info("ref_peer_financials: %s rows", f"{len(out):,}")

    uci_path = s.interim_dir / "uci_campaign.csv"
    if uci_path.is_file():
        uci = pd.read_csv(uci_path).rename(columns={"default": "default_credit"})
        cols = ["age", "job", "marital", "education", "default_credit", "housing",
                "loan", "contact", "month", "day_of_week", "campaign", "pdays",
                "previous", "poutcome", "emp_var_rate", "cons_price_idx",
                "cons_conf_idx", "euribor3m", "nr_employed", "subscribed"]
        present = [c for c in cols if c in uci.columns]
        uci[present].to_sql("ref_uci_campaign", conn, if_exists="append",
                            index=False, chunksize=20_000)
        log.info("ref_uci_campaign: %s rows (real)", f"{len(uci):,}")


def _assert_mapped(df: pd.DataFrame, key_cols: list[str], table: str) -> None:
    """Fail loudly when a natural key did not resolve to a surrogate.

    An unmapped key becomes NaN, and a NaN foreign key silently drops the row
    from every subsequent join instead of raising. Catching it here means the
    loader reports "3 rows could not be mapped" rather than the analysis quietly
    reporting revenue that is short by those rows.
    """
    for col in key_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            raise WarehouseError(
                f"{table}: {n_missing:,} row(s) have an unmapped {col} -- "
                "a natural key in the fact has no matching dimension row"
            )

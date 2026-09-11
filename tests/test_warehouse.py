"""Tests for the warehouse: schema, load integrity, grain, and query semantics.

Grain is tested explicitly for every fact table. "What is one row here" is the
first thing anyone should ask of a fact table, and a grain violation -- a
duplicate that should be impossible -- corrupts every aggregate built on top of
it while looking like nothing is wrong.

These tests build a small warehouse from a generated fixture rather than
depending on the full 2.6M-row build, so they run in seconds and pass in CI
where no generated data exists.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from meridian.common.config import Settings
from meridian.common.exceptions import WarehouseError
from meridian.generation.calibration import (
    CalibrationProfile,
    JointDemographics,
    LogNormalFit,
    PeerEnvelope,
)
from meridian.generation.customers import GenerationWindow
from meridian.generation.generator import generate_all
from meridian.warehouse import db, loader
from meridian.warehouse.query_layer import QUERIES, load_sql


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory) -> Settings:
    """A small but complete warehouse, built end to end."""
    root = tmp_path_factory.mktemp("wh")
    s = Settings(root=root, seed=20260911, n_customers=400)
    s.ensure_dirs()

    profile = CalibrationProfile(seed=20260911, generated_at="test", n_customers=400)
    profile.wealth = LogNormalFit(mu=11.36, sigma=1.25, n=1000, source="test")
    profile.demographics = JointDemographics(
        columns=("age_band", "job", "marital", "education"),
        combinations=[
            ("25-34", "admin.", "single", "university.degree"),
            ("35-44", "technician", "married", "high.school"),
            ("45-54", "management", "married", "university.degree"),
            ("55-64", "retired", "married", "basic.4y"),
        ],
        weights=[0.3, 0.3, 0.25, 0.15], n_source_rows=41_188, source="test",
    )
    profile.lending_rate_pct = 8.567143
    profile.deposit_rate_pct = 4.54
    profile.base_conversion_rate = 0.1127
    profile.peer_envelopes = {
        "ROA": PeerEnvelope("ROA", 0.1, 1.3, 9.9, 40),
        "NIMY": PeerEnvelope("NIMY", 0.1, 4.0, 20.0, 40),
    }
    profile.peer_equity_ratio = 0.1034

    import meridian.generation.generator as gen
    monkey_profile = profile

    original = gen.build_profile
    gen.build_profile = lambda _s: monkey_profile
    try:
        data = generate_all(
            s, window=GenerationWindow(date(2024, 1, 1), date(2026, 6, 30)),
            n_customers=400, strict_calibration=False,
        )
    finally:
        gen.build_profile = original

    data.save(s.synthetic_dir)
    loader.build_warehouse(s)
    return s


class TestSchema:
    def test_foreign_keys_are_enforced_on_every_connection(self, warehouse):
        # SQLite disables them by default; a star schema without them is just
        # a set of loosely associated tables.
        with db.session(warehouse.db_path, read_only=True) as conn:
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_no_foreign_key_violations_after_load(self, warehouse):
        assert db.foreign_key_violations(warehouse.db_path).empty

    def test_integrity_check_passes(self, warehouse):
        assert db.integrity_check(warehouse.db_path) == "ok"

    def test_every_expected_table_exists(self, warehouse):
        counts = db.table_counts(warehouse.db_path)
        for t in ["dim_date", "dim_customer", "dim_product", "dim_channel",
                  "dim_branch", "dim_campaign", "fact_monthly_account",
                  "fact_transaction", "fact_campaign_contact",
                  "fact_customer_snapshot"]:
            assert t in counts, f"{t} missing"

    def test_dimensions_are_populated(self, warehouse):
        counts = db.table_counts(warehouse.db_path)
        assert counts["dim_customer"] == 400
        assert counts["dim_product"] > 0
        assert counts["dim_date"] > 365


class TestGrain:
    """Each fact table's grain, asserted directly."""

    def test_monthly_account_is_one_row_per_customer_product_month(self, warehouse):
        dupes = db.query(warehouse.db_path, """
            SELECT customer_key, product_key, month, COUNT(*) AS n
            FROM fact_monthly_account
            GROUP BY customer_key, product_key, month
            HAVING COUNT(*) > 1
        """)
        assert dupes.empty, f"{len(dupes)} grain violations in fact_monthly_account"

    def test_transaction_is_one_row_per_transaction(self, warehouse):
        dupes = db.query(warehouse.db_path, """
            SELECT txn_id, COUNT(*) AS n FROM fact_transaction
            GROUP BY txn_id HAVING COUNT(*) > 1
        """)
        assert dupes.empty

    def test_campaign_contact_grain_includes_sequence(self, warehouse):
        # A customer contacted three times has three rows; collapsing them
        # would make contact-fatigue analysis impossible.
        dupes = db.query(warehouse.db_path, """
            SELECT campaign_key, customer_key, contact_sequence, COUNT(*) AS n
            FROM fact_campaign_contact
            GROUP BY campaign_key, customer_key, contact_sequence
            HAVING COUNT(*) > 1
        """)
        assert dupes.empty

    def test_customer_snapshot_is_one_row_per_customer(self, warehouse):
        dupes = db.query(warehouse.db_path, """
            SELECT customer_key, COUNT(*) AS n FROM fact_customer_snapshot
            GROUP BY customer_key HAVING COUNT(*) > 1
        """)
        assert dupes.empty

    def test_dimension_natural_keys_are_unique(self, warehouse):
        for table, key in [("dim_customer", "customer_id"),
                           ("dim_product", "product_code"),
                           ("dim_campaign", "campaign_id"),
                           ("dim_date", "full_date")]:
            dupes = db.query(warehouse.db_path, f"""
                SELECT {key}, COUNT(*) AS n FROM {table}
                GROUP BY {key} HAVING COUNT(*) > 1
            """)
            assert dupes.empty, f"{table}.{key} is not unique"

    def test_grain_is_enforced_by_a_constraint_not_just_convention(self, warehouse):
        # Re-inserting an existing row must be rejected by the database itself.
        with db.session(warehouse.db_path) as conn:
            row = conn.execute(
                "SELECT customer_key, product_key, date_key, month FROM "
                "fact_monthly_account LIMIT 1"
            ).fetchone()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO fact_monthly_account (customer_key, product_key, "
                    "date_key, month, balance_inr, interest_rate_pct, "
                    "interest_revenue_inr, fee_revenue_inr, total_revenue_inr, "
                    "account_age_months) VALUES (?,?,?,?,0,0,0,0,0,0)",
                    (row["customer_key"], row["product_key"], row["date_key"],
                     row["month"]),
                )
            conn.rollback()


class TestReferentialIntegrity:
    def test_no_orphan_facts(self, warehouse):
        for fact, key, dim in [
            ("fact_monthly_account", "customer_key", "dim_customer"),
            ("fact_monthly_account", "product_key", "dim_product"),
            ("fact_transaction", "customer_key", "dim_customer"),
            ("fact_campaign_contact", "campaign_key", "dim_campaign"),
        ]:
            orphans = db.query(warehouse.db_path, f"""
                SELECT COUNT(*) AS n FROM {fact} f
                LEFT JOIN {dim} d ON d.{key} = f.{key}
                WHERE d.{key} IS NULL
            """)
            assert orphans.iloc[0]["n"] == 0, f"{fact}.{key} has orphans"

    def test_churned_and_censored_are_complements(self, warehouse):
        bad = db.query(warehouse.db_path, """
            SELECT COUNT(*) AS n FROM fact_customer_snapshot
            WHERE churned + censored <> 1
        """)
        assert bad.iloc[0]["n"] == 0

    def test_churn_date_exists_exactly_when_churned(self, warehouse):
        bad = db.query(warehouse.db_path, """
            SELECT COUNT(*) AS n FROM fact_customer_snapshot
            WHERE (churned = 1 AND churn_date IS NULL)
               OR (churned = 0 AND churn_date IS NOT NULL)
        """)
        assert bad.iloc[0]["n"] == 0

    def test_unmapped_natural_key_is_caught_not_silently_dropped(self):
        # A NaN foreign key vanishes from joins instead of raising, which is why
        # the loader checks explicitly.
        df = pd.DataFrame({"customer_key": [1.0, np.nan, 3.0]})
        with pytest.raises(WarehouseError, match="unmapped"):
            loader._assert_mapped(df, ["customer_key"], "test_table")


class TestQueryLayer:
    def test_ten_queries_are_registered(self):
        assert len(QUERIES) == 10
        assert [q.key for q in QUERIES] == [f"q{i:02d}" for i in range(1, 11)]

    def test_every_query_file_exists_and_is_readable(self):
        for spec in QUERIES:
            sql = load_sql(spec.key)
            assert sql.strip()
            # Each query documents its own grain in a header comment.
            assert "Grain:" in sql, f"{spec.key} does not state its grain"

    def test_unknown_query_key_raises(self):
        with pytest.raises(WarehouseError, match="unknown query"):
            load_sql("q99")

    @pytest.mark.parametrize("key", [f"q{i:02d}" for i in range(1, 11)])
    def test_query_executes_and_returns_rows(self, warehouse, key):
        from meridian.warehouse.query_layer import run_query
        df = run_query(key, warehouse)
        assert not df.empty, f"{key} returned no rows"

    def test_q10_returns_exactly_eight_kpis(self, warehouse):
        from meridian.warehouse.query_layer import run_query
        df = run_query("q10", warehouse)
        assert len(df) == 8
        assert df["kpi_key"].is_unique

    def test_q10_marks_churn_as_lower_is_better(self, warehouse):
        # A tile that colours rising churn green would be actively misleading.
        from meridian.warehouse.query_layer import run_query
        df = run_query("q10", warehouse)
        churn = df[df["kpi_key"] == "churn_rate_pct"].iloc[0]
        assert churn["good_direction"] == "down"

    def test_q02_recency_score_is_not_inverted(self, warehouse):
        # The most common RFM bug: scoring recency ascending, so the customer
        # who last transacted years ago scores 5. Inverts every segment name.
        from meridian.warehouse.query_layer import run_query
        df = run_query("q02", warehouse)
        recent = df.nsmallest(50, "recency_days")["r_score"].mean()
        stale = df.nlargest(50, "recency_days")["r_score"].mean()
        assert recent > stale, "recency scoring is inverted"

    def test_q04_pareto_cumulative_share_is_monotonic(self, warehouse):
        from meridian.warehouse.query_layer import run_query
        df = run_query("q04", warehouse)
        assert df["cumulative_share_pct"].is_monotonic_increasing
        assert df["cumulative_share_pct"].iloc[-1] == pytest.approx(100.0, abs=0.5)

    def test_q05_retention_never_exceeds_one_hundred_percent(self, warehouse):
        from meridian.warehouse.query_layer import run_query
        df = run_query("q05", warehouse)
        for col in ["m01_pct", "m03_pct", "m06_pct"]:
            assert (df[col].dropna() <= 100.0).all()

    def test_q05_retention_is_non_increasing_with_horizon(self, warehouse):
        # Survival cannot rise: a customer retained at month 6 was retained at 3.
        from meridian.warehouse.query_layer import run_query
        df = run_query("q05", warehouse).dropna(subset=["m01_pct", "m03_pct", "m06_pct"])
        assert (df["m01_pct"] >= df["m03_pct"] - 1e-9).all()
        assert (df["m03_pct"] >= df["m06_pct"] - 1e-9).all()

    def test_q06_excludes_already_churned_customers(self, warehouse):
        # Scoring the risk of an event that already happened is how a model
        # ends up reporting an AUC of 1.0.
        from meridian.warehouse.query_layer import run_query
        risk = run_query("q06", warehouse)
        churned = db.query(warehouse.db_path, """
            SELECT c.customer_id FROM dim_customer c
            JOIN fact_customer_snapshot s ON s.customer_key = c.customer_key
            WHERE s.churned = 1
        """)
        assert not set(risk["customer_id"]) & set(churned["customer_id"])

    def test_q07_survives_a_campaign_with_no_conversions(self, warehouse):
        # NULLIF guards: a zero denominator must give NULL, not an exception.
        from meridian.warehouse.query_layer import run_query
        df = run_query("q07", warehouse)
        assert "cost_per_acquisition_inr" in df.columns

    def test_q09_reports_active_only_concentration_separately(self, warehouse):
        # Churned customers have zero future value, making the blended figure
        # bimodal rather than a smooth Pareto tail.
        from meridian.warehouse.query_layer import run_query
        df = run_query("q09", warehouse)
        assert "is_active" in df.columns
        assert "active_clv_decile" in df.columns
        assert df.loc[df["is_active"] == 1, "active_clv_decile"].notna().all()

    def test_q09_uses_the_real_lending_rate_as_the_discount_rate(self):
        sql = load_sql("q09")
        assert "FR.INR.LEND" in sql
        # And it must take the latest NON-NULL, not the latest row.
        assert "value IS NOT NULL" in sql


class TestDateDimension:
    def test_indian_fiscal_year_starts_in_april(self, warehouse):
        df = db.query(warehouse.db_path, """
            SELECT full_date, year, month, fiscal_year, fiscal_quarter
            FROM dim_date WHERE full_date IN ('2025-03-31', '2025-04-01')
            ORDER BY full_date
        """)
        if len(df) == 2:
            march, april = df.iloc[0], df.iloc[1]
            assert march["fiscal_year"] == 2024   # March 2025 is FY2024-25
            assert april["fiscal_year"] == 2025   # April 2025 starts FY2025-26
            assert april["fiscal_quarter"] == 1

    def test_festive_season_covers_october_and_november(self, warehouse):
        df = db.query(warehouse.db_path, """
            SELECT DISTINCT month FROM dim_date WHERE is_festive_season = 1
            ORDER BY month
        """)
        assert df["month"].tolist() == [10, 11]

    def test_weekend_flag_matches_day_of_week(self, warehouse):
        bad = db.query(warehouse.db_path, """
            SELECT COUNT(*) AS n FROM dim_date
            WHERE (day_of_week >= 5 AND is_weekend = 0)
               OR (day_of_week < 5 AND is_weekend = 1)
        """)
        assert bad.iloc[0]["n"] == 0


class TestReferenceData:
    def test_real_uci_rows_are_preserved(self, warehouse):
        counts = db.table_counts(warehouse.db_path)
        # The real campaign data lands in the warehouse alongside the synthetic
        # spine, so a query can compare the two without leaving the database.
        if counts.get("ref_uci_campaign"):
            assert counts["ref_uci_campaign"] == 41_188

    def test_duration_never_reaches_the_warehouse(self, warehouse):
        cols = db.query(warehouse.db_path, "PRAGMA table_info(ref_uci_campaign)")
        assert "duration" not in set(cols["name"]), "the leaky column got in"

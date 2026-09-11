"""Tests for the generation layer.

Two kinds of check here. Structural invariants -- referential integrity, grain,
no impossible dates -- which must hold for any run. And recovery checks, which
assert that the analytical machinery finds what the generator planted, since a
generator whose planted effects cannot be recovered is not useful as a test bed.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from meridian.generation.calibration import (
    CalibrationProfile,
    JointDemographics,
    LogNormalFit,
    PeerEnvelope,
)
from meridian.generation.churn import HazardSpec, generate_churn
from meridian.generation.customers import GenerationWindow, generate_customers
from meridian.generation.products import (
    CATALOGUE,
    generate_holdings,
    product_dimension,
)
from meridian.generation.transactions import SEASONALITY, generate_transactions


@pytest.fixture(scope="module")
def profile() -> CalibrationProfile:
    """A calibration profile with realistic fitted values, built offline."""
    p = CalibrationProfile(seed=20260911, generated_at="test", n_customers=1500)
    p.wealth = LogNormalFit(mu=11.36, sigma=1.25, n=1000, source="test")
    p.demographics = JointDemographics(
        columns=("age_band", "job", "marital", "education"),
        combinations=[
            ("25-34", "admin.", "single", "university.degree"),
            ("35-44", "technician", "married", "high.school"),
            ("45-54", "management", "married", "university.degree"),
            ("55-64", "retired", "married", "basic.4y"),
            ("<25", "student", "single", "high.school"),
            ("65+", "retired", "divorced", "basic.9y"),
        ],
        weights=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05],
        n_source_rows=41_188, source="test",
    )
    p.lending_rate_pct = 8.567143
    p.lending_rate_year = 2022
    p.deposit_rate_pct = 4.54
    p.base_conversion_rate = 0.1127
    p.peer_envelopes = {
        "ROA": PeerEnvelope("ROA", 1.142, 1.342, 1.702, 40),
        "NIMY": PeerEnvelope("NIMY", 3.744, 4.026, 4.226, 40),
    }
    p.peer_equity_ratio = 0.1034
    return p


@pytest.fixture(scope="module")
def window() -> GenerationWindow:
    return GenerationWindow(date(2023, 1, 1), date(2026, 6, 30))


@pytest.fixture(scope="module")
def customers(profile, window) -> pd.DataFrame:
    return generate_customers(profile, window, n=1500,
                              rng=np.random.default_rng(7))


@pytest.fixture(scope="module")
def holdings(customers, profile, window) -> pd.DataFrame:
    return generate_holdings(customers, profile, window,
                             rng=np.random.default_rng(8))


class TestDeterminism:
    """A fixed seed must give byte-identical output, or nothing is reproducible."""

    def test_same_seed_gives_identical_customers(self, profile, window):
        a = generate_customers(profile, window, n=300, rng=np.random.default_rng(1))
        b = generate_customers(profile, window, n=300, rng=np.random.default_rng(1))
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_gives_different_output(self, profile, window):
        a = generate_customers(profile, window, n=300, rng=np.random.default_rng(1))
        b = generate_customers(profile, window, n=300, rng=np.random.default_rng(2))
        assert not a["annual_income_inr"].equals(b["annual_income_inr"])


class TestCustomerStructure:
    def test_customer_id_is_unique(self, customers):
        assert customers["customer_id"].is_unique

    def test_ages_lie_inside_their_band(self, customers):
        bounds = {"<25": (18, 24), "25-34": (25, 34), "35-44": (35, 44),
                  "45-54": (45, 54), "55-64": (55, 64), "65+": (65, 88)}
        for band, (lo, hi) in bounds.items():
            sub = customers[customers["age_band"] == band]
            if not sub.empty:
                assert sub["age"].between(lo, hi).all(), f"{band} leaked outside"

    def test_income_is_never_absurdly_low(self, customers):
        # An unfloored lognormal produced a minimum of INR 800 a year.
        working = customers[~customers["job"].isin(["student", "unemployed"])]
        assert working["annual_income_inr"].min() >= 100_000

    def test_income_ordering_by_occupation_is_economically_sane(self, customers):
        med = customers.groupby("job")["annual_income_inr"].median()
        if {"management", "housemaid"} <= set(med.index):
            assert med["management"] > med["housemaid"]

    def test_credit_score_is_in_the_cibil_range(self, customers):
        assert customers["credit_score"].between(300, 900).all()

    def test_credit_score_correlates_with_income(self, customers):
        # An uncorrelated score would make every risk finding meaningless.
        r = np.corrcoef(np.log1p(customers["annual_income_inr"]),
                        customers["credit_score"])[0, 1]
        assert r > 0.2

    def test_segments_follow_income_thresholds(self, customers):
        for seg, lo in [("affluent", 1_200_000), ("priority", 3_000_000),
                        ("private", 10_000_000)]:
            sub = customers[customers["segment"] == seg]
            if not sub.empty:
                assert sub["annual_income_inr"].min() >= lo

    def test_acquisition_falls_inside_the_window(self, customers, window):
        acq = pd.to_datetime(customers["acquired_date"])
        assert acq.min() >= pd.Timestamp(window.start)
        assert acq.max() <= pd.Timestamp(window.end) + pd.Timedelta(days=28)

    def test_frailty_is_positive_and_roughly_centred(self, customers):
        # Multiplies a hazard, so it must be strictly positive.
        assert (customers["frailty"] > 0).all()
        assert 0.7 < customers["frailty"].mean() < 1.4

    def test_digital_engagement_is_a_probability(self, customers):
        assert customers["digital_engagement"].between(0, 1).all()

    def test_digital_engagement_falls_with_age(self, customers):
        r = np.corrcoef(customers["age"], customers["digital_engagement"])[0, 1]
        assert r < 0


class TestProducts:
    def test_every_product_is_priced_off_the_real_lending_rate(self, profile):
        dim = product_dimension(profile)
        for _, row in dim.iterrows():
            expected = max(profile.lending_rate_pct + row["rate_offset_pp"], 0.0)
            assert row["interest_rate_pct"] == pytest.approx(expected, abs=1e-4)

    def test_dimension_records_its_own_provenance(self, profile):
        dim = product_dimension(profile)
        # The dimension stores the rate rounded to 4dp for readability, so the
        # comparison has to admit that rounding. Compared elementwise rather
        # than with a Series == approx(scalar), which does not broadcast.
        recorded = dim["calibrated_from_lending_rate_pct"].unique()
        assert len(recorded) == 1
        assert recorded[0] == pytest.approx(profile.lending_rate_pct, abs=5e-4)

    def test_deposits_pay_less_than_lending_earns(self, profile):
        dim = product_dimension(profile)
        deposit_max = dim.loc[dim["category"] == "deposit", "interest_rate_pct"].max()
        lending_min = dim.loc[dim["category"] == "lending", "interest_rate_pct"].min()
        # The gap between the two is where net interest income comes from.
        assert deposit_max < lending_min + 1e-9

    def test_exactly_one_anchor_product(self):
        assert sum(p.is_anchor for p in CATALOGUE) == 1

    def test_no_holding_predates_the_relationship(self, customers, holdings):
        m = holdings.merge(customers[["customer_id", "acquired_date"]], on="customer_id")
        assert (pd.to_datetime(m["opened_date"])
                >= pd.to_datetime(m["acquired_date"])).all()

    def test_income_eligibility_is_enforced(self, customers, holdings):
        m = holdings.merge(
            customers[["customer_id", "annual_income_inr"]], on="customer_id"
        )
        for spec in CATALOGUE:
            if spec.min_income_inr <= 0:
                continue
            sub = m[m["product_code"] == spec.code]
            if not sub.empty:
                assert sub["annual_income_inr"].min() >= spec.min_income_inr

    def test_non_anchor_holders_also_hold_the_anchor(self, holdings):
        # Sequential cross-sell: nobody has a mortgage but no account.
        anchor = next(p.code for p in CATALOGUE if p.is_anchor)
        with_anchor = set(holdings.loc[holdings["product_code"] == anchor, "customer_id"])
        others = set(holdings.loc[holdings["product_code"] != anchor, "customer_id"])
        assert others <= with_anchor

    def test_balances_are_non_negative(self, holdings):
        assert (holdings["balance_inr"] >= 0).all()


class TestChurnHazard:
    @pytest.fixture(scope="class")
    def churn(self, customers, holdings, profile, window):
        return generate_churn(customers, holdings, profile, window,
                              rng=np.random.default_rng(9))

    def test_panel_has_one_row_per_at_risk_customer_month(self, churn):
        assert not churn.panel.duplicated(["customer_id", "month"]).any()

    def test_churn_rate_is_realistic(self, churn):
        # A retail deposit book loses roughly 10-20% of customers a year.
        implied_annual = 1 - (1 - churn.panel["hazard"].mean()) ** 12
        assert 0.05 < implied_annual < 0.30

    def test_censoring_is_explicit_and_consistent(self, churn):
        o = churn.outcomes
        assert (o["churned"] + o["censored"] == 1).all()
        assert o.loc[o["churned"] == 1, "churn_date"].notna().all()
        assert o.loc[o["censored"] == 1, "churn_date"].isna().all()

    def test_most_customers_are_censored(self, churn):
        # Right censoring is the norm in a window this short; if almost nobody
        # were censored the survival analysis would be trivial.
        assert churn.outcomes["censored"].mean() > 0.4

    def test_nobody_is_at_risk_after_churning(self, churn):
        m = churn.panel.merge(
            churn.outcomes[["customer_id", "churn_date"]], on="customer_id"
        )
        churned = m[m["churn_date"].notna()]
        assert (churned["month"] <= churned["churn_date"]).all()

    def test_nobody_is_at_risk_before_acquisition(self, churn, customers):
        m = churn.panel.merge(
            customers[["customer_id", "acquired_date"]], on="customer_id"
        )
        acq_month = pd.to_datetime(m["acquired_date"]).dt.to_period("M").dt.to_timestamp()
        assert (m["month"] >= acq_month).all()

    def test_hazard_is_a_probability(self, churn):
        assert churn.panel["hazard"].between(0, 1).all()

    def test_ground_truth_is_recorded(self, churn):
        assert "coefficients" in churn.truth
        assert churn.truth["break_month"]

    def test_frailty_is_not_exposed_in_the_panel(self, churn):
        # The whole anti-circularity argument depends on this staying hidden.
        assert "frailty" not in churn.panel.columns

    def test_break_raises_the_hazard_at_the_planted_month(self, churn):
        by_month = churn.panel.groupby("month")["hazard"].mean()
        months = list(by_month.index)
        i = months.index(churn.break_month)
        # Comparing adjacent months holds composition roughly fixed.
        assert by_month.iloc[i] > by_month.iloc[i - 1]


class TestHazardSpec:
    def test_coefficient_signs_are_economically_sensible(self):
        s = HazardSpec()
        assert s.log_tenure < 0            # longer tenure, lower churn
        assert s.n_products < 0            # more products, stickier
        assert s.digital_engagement < 0    # engaged customers stay
        assert s.txn_recency_months > 0    # lapsed customers leave
        assert s.complaint_last_quarter > 0
        assert s.structural_break_log_odds > 0

    def test_intercept_implies_a_realistic_base_rate(self):
        s = HazardSpec()
        monthly = 1 / (1 + np.exp(-s.intercept))
        annual = 1 - (1 - monthly) ** 12
        assert 0.20 < annual < 0.85  # before covariates pull it down


class TestTransactions:
    @pytest.fixture(scope="class")
    def txns(self, customers, holdings, profile, window):
        from meridian.generation.churn import generate_churn
        ch = generate_churn(customers, holdings, profile, window,
                            rng=np.random.default_rng(9))
        return generate_transactions(customers, holdings, ch.outcomes, window,
                                     profile, rng=np.random.default_rng(10)), ch

    def test_txn_id_is_unique(self, txns):
        t, _ = txns
        assert t["txn_id"].is_unique

    def test_no_transaction_after_churn(self, txns):
        t, ch = txns
        m = t.merge(ch.outcomes[["customer_id", "churn_date"]], on="customer_id")
        churned = m[m["churn_date"].notna()]
        assert (churned["txn_date"] < churned["churn_date"]).all()

    def test_amounts_are_positive(self, txns):
        t, _ = txns
        assert (t["amount_inr"] > 0).all()

    def test_amounts_follow_benford(self, txns):
        # Our own anomaly module must not flag our own generated data.
        from meridian.validation import benford_test
        t, _ = txns
        b = benford_test(t["amount_inr"])
        assert b.mad < 0.012, f"MAD {b.mad:.4f} -- worse than acceptable conformity"

    def test_amounts_are_not_clustered_on_round_numbers(self, txns):
        from meridian.validation import round_number_bias
        t, _ = txns
        assert round_number_bias(t["amount_inr"])["pct_multiple_of_100"] < 0.05

    def test_festive_season_is_the_annual_peak(self):
        # October-November is the dominant feature of Indian retail spending.
        peak = max(range(12), key=lambda i: SEASONALITY[i])
        assert peak in (9, 10)  # October or November, zero-indexed

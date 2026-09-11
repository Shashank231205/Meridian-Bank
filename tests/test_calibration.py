"""Tests for the calibration layer.

Fitted estimators are checked by recovery: generate a sample from known
parameters, fit it, and assert the fit returns what went in. That tests the
estimator rather than freezing whatever it currently produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.common.exceptions import CalibrationError
from meridian.generation.calibration import (
    CalibrationProfile,
    LogNormalFit,
    PeerEnvelope,
    assert_calibrated,
    fit_joint_demographics,
    fit_lognormal,
    fit_macro_elasticity,
)


class TestLogNormalFit:
    def test_recovers_known_parameters(self):
        # The estimator must return the parameters the sample was drawn from.
        rng = np.random.default_rng(42)
        mu_true, sigma_true = 9.5, 1.1
        sample = pd.Series(rng.lognormal(mu_true, sigma_true, 50_000))
        fit = fit_lognormal(sample, source="synthetic")
        assert fit.mu == pytest.approx(mu_true, abs=0.02)
        assert fit.sigma == pytest.approx(sigma_true, abs=0.02)

    def test_theoretical_median_matches_the_sample_median(self):
        rng = np.random.default_rng(1)
        sample = pd.Series(rng.lognormal(8.0, 0.9, 20_000))
        fit = fit_lognormal(sample, source="s")
        assert fit.theoretical_median == pytest.approx(fit.observed_median, rel=0.05)

    def test_mean_exceeds_median_for_a_right_skewed_fit(self):
        # A defining property of the lognormal, and of wealth.
        fit = LogNormalFit(mu=10.0, sigma=1.2, n=100, source="s")
        assert fit.theoretical_mean > fit.theoretical_median

    def test_non_positive_values_are_excluded(self):
        rng = np.random.default_rng(3)
        positive = rng.lognormal(8.0, 0.8, 5_000)
        mixed = pd.Series(np.concatenate([positive, np.zeros(500), -positive[:500]]))
        fit = fit_lognormal(mixed, source="s")
        assert fit.n == 5_000

    def test_too_few_positive_values_raises(self):
        with pytest.raises(CalibrationError, match="cannot fit lognormal"):
            fit_lognormal(pd.Series([1.0, 2.0, 3.0]), source="s")


class TestJointDemographics:
    def test_weights_form_a_probability_distribution(self):
        df = pd.DataFrame({
            "age_band": ["25-34", "35-44", "25-34", "45-54"],
            "job": ["admin", "technician", "admin", "management"],
        })
        j = fit_joint_demographics(df, ("age_band", "job"), source="t")
        assert sum(j.weights) == pytest.approx(1.0)
        assert all(w > 0 for w in j.weights)

    def test_repeated_combinations_are_collapsed_and_weighted(self):
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 1, 2]})
        j = fit_joint_demographics(df, ("a", "b"), source="t")
        assert j.n_combinations == 2
        assert max(j.weights) == pytest.approx(2 / 3)

    def test_preserves_correlation_that_independent_marginals_would_lose(self):
        # Only two combinations occur, though the marginals admit four. An
        # independent sampler would invent the other two.
        df = pd.DataFrame({"age": ["young"] * 50 + ["old"] * 50,
                           "job": ["student"] * 50 + ["retired"] * 50})
        j = fit_joint_demographics(df, ("age", "job"), source="t")
        assert j.n_combinations == 2
        assert ("young", "student") in j.combinations
        assert ("young", "retired") not in j.combinations

    def test_missing_columns_are_skipped_not_fatal(self):
        df = pd.DataFrame({"a": ["x"]})
        assert fit_joint_demographics(df, ("a", "absent"), source="t").columns == ("a",)

    def test_no_usable_column_raises(self):
        with pytest.raises(CalibrationError):
            fit_joint_demographics(pd.DataFrame({"z": [1]}), ("a", "b"), source="t")


class TestMacroElasticity:
    def test_recovers_a_planted_negative_relationship(self):
        # Conversion falls as the driver rises: the sign must come back negative.
        rng = np.random.default_rng(11)
        x = rng.uniform(0, 5, 6_000)
        p = 1 / (1 + np.exp(-(1.0 - 0.8 * x)))
        df = pd.DataFrame({"rate": x, "subscribed": rng.binomial(1, p)})
        e = fit_macro_elasticity(df, "rate")
        assert e is not None and e.log_odds_per_unit < 0
        assert e.conversion_high < e.conversion_low

    def test_recovers_a_planted_positive_relationship(self):
        rng = np.random.default_rng(12)
        x = rng.uniform(0, 5, 6_000)
        p = 1 / (1 + np.exp(-(-2.0 + 0.8 * x)))
        df = pd.DataFrame({"rate": x, "subscribed": rng.binomial(1, p)})
        e = fit_macro_elasticity(df, "rate")
        assert e is not None and e.log_odds_per_unit > 0

    def test_returns_none_for_an_absent_driver(self):
        df = pd.DataFrame({"subscribed": [0, 1]})
        assert fit_macro_elasticity(df, "nope") is None

    def test_returns_none_for_a_constant_driver(self):
        df = pd.DataFrame({"rate": [1.0] * 200, "subscribed": [0, 1] * 100})
        assert fit_macro_elasticity(df, "rate") is None

    def test_returns_none_for_too_small_a_sample(self):
        df = pd.DataFrame({"rate": [1.0, 2.0, 3.0], "subscribed": [0, 1, 0]})
        assert fit_macro_elasticity(df, "rate") is None


class TestPeerEnvelope:
    def test_contains_is_inclusive_at_the_quartiles(self):
        env = PeerEnvelope("ROA", q1=1.0, median=1.3, q3=1.7, n_institutions=40)
        assert env.contains(1.0) and env.contains(1.7) and env.contains(1.3)
        assert not env.contains(0.99) and not env.contains(1.71)


class TestBuildGate:
    """assert_calibrated is a gate, not a comment: it must actually stop a build."""

    @staticmethod
    def _profile() -> CalibrationProfile:
        p = CalibrationProfile(seed=1, generated_at="now")
        p.peer_envelopes = {
            "ROA": PeerEnvelope("ROA", 1.142, 1.342, 1.702, 40),
            "NIMY": PeerEnvelope("NIMY", 3.744, 4.026, 4.226, 40),
        }
        return p

    def test_passes_inside_the_envelope(self):
        r = assert_calibrated({"ROA": 1.34, "NIMY": 4.02}, self._profile())
        assert r["passed"] and r["n_checked"] == 2 and r["n_failed"] == 0

    def test_raises_outside_the_envelope(self):
        with pytest.raises(CalibrationError, match="outside the real FDIC peer envelope"):
            assert_calibrated({"ROA": 9.9}, self._profile())

    def test_error_names_the_metric_and_the_bounds(self):
        with pytest.raises(CalibrationError) as exc:
            assert_calibrated({"ROA": 9.9}, self._profile())
        msg = str(exc.value)
        assert "ROA" in msg and "1.1420" in msg and "40 banks" in msg

    def test_strict_false_reports_without_raising(self):
        r = assert_calibrated({"ROA": 9.9}, self._profile(), strict=False)
        assert not r["passed"] and r["n_failed"] == 1

    def test_metric_without_an_envelope_is_unchecked_not_passed(self):
        # Silently passing an unverifiable metric would defeat the point.
        r = assert_calibrated({"UNKNOWN": 5.0}, self._profile())
        assert r["n_checked"] == 0
        assert r["checks"][0]["status"] == "unchecked"

    def test_boundary_values_pass(self):
        p = self._profile()
        assert assert_calibrated({"ROA": 1.142}, p)["passed"]
        assert assert_calibrated({"ROA": 1.702}, p)["passed"]


class TestProfileSerialisation:
    def test_round_trips_through_json(self, tmp_path):
        from meridian.common.io import read_json
        p = CalibrationProfile(seed=20260911, generated_at="2026-09-11T00:00:00",
                               n_customers=25_000, lending_rate_pct=8.567143,
                               lending_rate_year=2022)
        p.wealth = LogNormalFit(mu=11.36, sigma=1.25, n=0, source="s")
        out = read_json(p.save(tmp_path / "c.json"))
        assert out["seed"] == 20260911
        assert out["interest"]["lending_rate_pct"] == pytest.approx(8.5671, abs=1e-4)
        assert out["interest"]["lending_rate_year"] == 2022

    def test_summary_states_the_real_lending_rate(self):
        p = CalibrationProfile(seed=1, generated_at="now",
                               lending_rate_pct=8.567143, lending_rate_year=2022)
        s = p.summary()
        assert "8.5671" in s and "2022" in s and "World Bank" in s


@pytest.mark.network
class TestAgainstRealData:
    """Runs against the ingested real tables when they are present."""

    def test_profile_carries_the_real_lending_rate(self):
        from meridian.common import get_settings
        from meridian.generation.calibration import build_profile

        s = get_settings()
        if not (s.interim_dir / "uci_campaign.csv").is_file():
            pytest.skip("ingestion has not been run")

        p = build_profile(s)
        # World Bank FR.INR.LEND for India, latest non-null: 8.567143% (2022).
        assert p.lending_rate_pct == pytest.approx(8.567143, abs=1e-4)
        assert p.lending_rate_year == 2022
        # UCI conversion, real: 11.27%.
        assert p.base_conversion_rate == pytest.approx(0.1127, abs=0.001)
        # Higher rates must suppress deposit subscription.
        assert p.elasticities["euribor3m"].log_odds_per_unit < 0
        # Peer envelopes come from real banks.
        assert p.peer_envelopes["ROA"].n_institutions >= 20

    def test_demographics_come_from_all_41188_real_rows(self):
        from meridian.common import get_settings
        from meridian.generation.calibration import build_profile

        s = get_settings()
        if not (s.interim_dir / "uci_campaign.csv").is_file():
            pytest.skip("ingestion has not been run")
        p = build_profile(s)
        assert p.demographics.n_source_rows == 41_188

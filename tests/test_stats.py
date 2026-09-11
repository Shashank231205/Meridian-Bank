"""Tests for the statistics and ML layer.

Everything here is checked against a value computed independently of this code:
a published statistical table, a worked example from a textbook or paper, or a
known parameter that a recovery test must return. Testing an estimator against
its own output proves only that it is deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from meridian.ml.kmeans import choose_k, fit_kmeans, silhouette_score
from meridian.ml.logistic import fit_logistic
from meridian.ml.metrics import (
    average_precision,
    brier_score,
    confusion_matrix,
    evaluate,
    ks_statistic,
    lift_table,
    optimal_threshold,
    roc_auc,
)
from meridian.stats.distributions import (
    betainc,
    chi2_cdf,
    chi2_ppf,
    chi2_sf,
    f_cdf,
    gammainc,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    normal_sf,
    t_cdf,
    t_ppf,
    t_two_sided_p,
)
from meridian.stats.intervals import (
    bootstrap_interval,
    mean_interval,
    wald_interval,
    wilson_interval,
)
from meridian.stats.tests import (
    anova_oneway,
    benjamini_hochberg,
    bonferroni,
    chi_square_independence,
    mann_whitney_u,
    paired_t_test,
    proportion_z_test,
    t_test,
)


class TestNormalDistribution:
    """Checked against published z-tables."""

    @pytest.mark.parametrize("z,expected", [
        (0.00, 0.5000000), (1.00, 0.8413447), (1.64, 0.9494974),
        (1.96, 0.9750021), (2.00, 0.9772499), (3.00, 0.9986501),
        (-1.00, 0.1586553), (-2.00, 0.0227501),
    ])
    def test_cdf_matches_z_table(self, z, expected):
        assert normal_cdf(z) == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize("p,expected", [
        (0.500, 0.000000), (0.900, 1.281552), (0.950, 1.644854),
        (0.975, 1.959964), (0.990, 2.326348), (0.995, 2.575829),
    ])
    def test_ppf_matches_published_quantiles(self, p, expected):
        assert normal_ppf(p) == pytest.approx(expected, abs=1e-6)

    def test_ppf_inverts_cdf(self):
        for p in [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999]:
            assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-10)

    def test_sf_is_accurate_in_the_far_tail(self):
        # 1 - cdf loses every significant digit out here; erfc does not.
        assert normal_sf(8.0) == pytest.approx(6.22096e-16, rel=1e-4)
        assert normal_sf(8.0) > 0

    def test_pdf_integrates_to_the_cdf(self):
        # Crude trapezoid check that pdf and cdf are consistent.
        xs = np.linspace(-6, 1.5, 20001)
        area = np.trapezoid([normal_pdf(x) for x in xs], xs)
        assert area == pytest.approx(normal_cdf(1.5), abs=1e-6)

    def test_symmetry(self):
        assert normal_cdf(-1.3) == pytest.approx(1 - normal_cdf(1.3), abs=1e-12)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            normal_cdf(0.0, 0.0, 0.0)


class TestStudentT:
    """Checked against published t-tables (two-sided, alpha = 0.05)."""

    @pytest.mark.parametrize("df,expected", [
        (1, 12.706), (2, 4.303), (5, 2.571), (10, 2.228),
        (20, 2.086), (30, 2.042), (60, 2.000), (100, 1.984),
    ])
    def test_critical_values_match_t_table(self, df, expected):
        assert t_ppf(0.975, df) == pytest.approx(expected, abs=0.001)

    def test_converges_to_normal_at_high_df(self):
        assert t_ppf(0.975, 100_000) == pytest.approx(1.959964, abs=1e-3)

    def test_cdf_is_symmetric(self):
        assert t_cdf(-2.0, 10) == pytest.approx(1 - t_cdf(2.0, 10), abs=1e-12)

    def test_two_sided_p_matches_hand_calculation(self):
        # t = 2.228 at df=10 is exactly the 0.05 two-sided critical value.
        assert t_two_sided_p(2.228, 10) == pytest.approx(0.05, abs=0.001)

    def test_zero_gives_p_of_one(self):
        assert t_two_sided_p(0.0, 10) == pytest.approx(1.0, abs=1e-12)


class TestChiSquared:
    """Checked against published chi-squared tables."""

    @pytest.mark.parametrize("df,expected", [
        (1, 3.841), (2, 5.991), (3, 7.815), (5, 11.070),
        (8, 15.507), (10, 18.307), (20, 31.410),
    ])
    def test_critical_values_match_table(self, df, expected):
        assert chi2_ppf(0.95, df) == pytest.approx(expected, abs=0.002)

    def test_sf_is_one_minus_cdf(self):
        assert chi2_sf(5.0, 3) == pytest.approx(1 - chi2_cdf(5.0, 3), abs=1e-12)

    def test_cdf_is_zero_at_or_below_zero(self):
        assert chi2_cdf(0.0, 3) == 0.0
        assert chi2_cdf(-1.0, 3) == 0.0

    def test_gammainc_matches_known_values(self):
        # P(1, x) = 1 - exp(-x), exactly.
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert gammainc(1.0, x) == pytest.approx(1 - math.exp(-x), abs=1e-12)


class TestIncompleteBeta:
    def test_symmetry_identity(self):
        # I_x(a,b) = 1 - I_{1-x}(b,a)
        assert betainc(2, 3, 0.4) == pytest.approx(1 - betainc(3, 2, 0.6), abs=1e-12)

    def test_uniform_case(self):
        # I_x(1,1) = x
        for x in [0.1, 0.5, 0.9]:
            assert betainc(1, 1, x) == pytest.approx(x, abs=1e-12)

    def test_boundaries(self):
        assert betainc(2, 3, 0.0) == 0.0
        assert betainc(2, 3, 1.0) == 1.0

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            betainc(2, 3, 1.5)


class TestFDistribution:
    @pytest.mark.parametrize("df1,df2,expected", [
        (1, 10, 4.965), (3, 20, 3.098), (5, 30, 2.534),
    ])
    def test_critical_values_match_f_table(self, df1, df2, expected):
        lo, hi = 0.001, 200.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f_cdf(mid, df1, df2) < 0.95:
                lo = mid
            else:
                hi = mid
        assert (lo + hi) / 2 == pytest.approx(expected, abs=0.01)


class TestWelchTTest:
    def test_matches_published_welch_example(self):
        # Standard worked example; Welch gives t = -2.46, df = 24.98, p = 0.021.
        a = [27.5, 21, 19, 23.6, 17, 17.9, 16.9, 20.1, 21.9, 22.6, 23.1, 19.6,
             19, 21.7, 21.4]
        b = [27.1, 22, 20.8, 23.4, 23.4, 23.5, 25.8, 22, 24.8, 20.2, 21.9, 22.1,
             22.9, 20.5, 24.4]
        r = t_test(a, b)
        assert r.statistic == pytest.approx(-2.46, abs=0.01)
        assert r.df == pytest.approx(24.98, abs=0.02)
        assert r.p_value == pytest.approx(0.021, abs=0.001)

    def test_welch_is_the_default(self):
        r = t_test([1, 2, 3, 4], [2, 3, 4, 5])
        assert "Welch" in r.test

    def test_welch_and_student_differ_on_unequal_variance(self):
        # The reason Welch is the default: with very different spreads and
        # group sizes, the pooled test is miscalibrated.
        rng = np.random.default_rng(3)
        small_wide = rng.normal(0, 10, 15)
        large_narrow = rng.normal(0, 1, 500)
        welch = t_test(small_wide, large_narrow, equal_var=False)
        student = t_test(small_wide, large_narrow, equal_var=True)
        assert welch.df < student.df

    def test_identical_samples_give_p_of_one(self):
        x = [1.0, 2.0, 3.0, 4.0]
        assert t_test(x, x).p_value == pytest.approx(1.0, abs=1e-9)

    def test_reports_cohens_d(self):
        r = t_test([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
        assert r.effect_name == "Cohen's d"
        assert r.effect_size == pytest.approx(-1.897, abs=0.01)

    def test_confidence_interval_brackets_the_difference(self):
        r = t_test([10, 12, 14, 16], [1, 2, 3, 4])
        assert r.ci_low < r.estimate < r.ci_high

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            t_test([1.0], [2.0, 3.0])


class TestPairedTTest:
    def test_detects_a_consistent_shift(self):
        before = [10, 12, 14, 16, 18]
        after = [12, 14, 16, 18, 20]   # every pair up by exactly 2
        r = paired_t_test(after, before)
        assert r.estimate == pytest.approx(2.0)
        assert r.p_value < 0.001


class TestAnova:
    def test_matches_published_worked_example(self):
        g1 = [6, 8, 4, 5, 3, 4]
        g2 = [8, 12, 9, 11, 6, 8]
        g3 = [13, 9, 11, 8, 7, 12]
        r = anova_oneway(g1, g2, g3)
        assert r.statistic == pytest.approx(9.26, abs=0.01)
        assert r.p_value == pytest.approx(0.0024, abs=0.0005)

    def test_reports_eta_squared(self):
        r = anova_oneway([1, 2, 3], [7, 8, 9])
        assert r.effect_name == "eta squared"
        assert 0 <= r.effect_size <= 1

    def test_single_group_raises(self):
        with pytest.raises(ValueError):
            anova_oneway([1, 2, 3])


class TestProportionTest:
    def test_matches_hand_calculation(self):
        # 40/100 vs 30/100: pooled p = 0.35, se = 0.06745, z = 1.4825
        r = proportion_z_test(40, 100, 30, 100)
        assert r.statistic == pytest.approx(1.4825, abs=0.001)
        assert r.p_value == pytest.approx(0.1382, abs=0.001)

    def test_identical_proportions_give_z_of_zero(self):
        assert proportion_z_test(50, 100, 50, 100).statistic == pytest.approx(0.0)

    def test_reports_cohens_h(self):
        r = proportion_z_test(40, 100, 30, 100)
        assert r.effect_name == "Cohen's h"


class TestChiSquareIndependence:
    def test_matches_hand_calculation(self):
        # 2x2 with equal margins: chi2 = 4.0, p = 0.0455
        r = chi_square_independence([[20, 30], [30, 20]])
        assert r.statistic == pytest.approx(4.0, abs=1e-9)
        assert r.df == 1
        assert r.p_value == pytest.approx(0.0455, abs=0.0005)

    def test_cramers_v_is_bounded(self):
        r = chi_square_independence([[20, 30], [30, 20]])
        assert 0 <= r.effect_size <= 1
        assert r.effect_size == pytest.approx(0.2, abs=1e-9)

    def test_perfect_independence_gives_zero(self):
        r = chi_square_independence([[25, 25], [25, 25]])
        assert r.statistic == pytest.approx(0.0, abs=1e-9)
        assert r.p_value == pytest.approx(1.0, abs=1e-9)

    def test_warns_on_small_expected_counts(self):
        r = chi_square_independence([[1, 2], [2, 1]])
        assert "unreliable" in r.note

    def test_too_small_a_table_raises(self):
        with pytest.raises(ValueError):
            chi_square_independence([[1, 2, 3]])


class TestMannWhitney:
    def test_perfect_separation_gives_u_of_zero(self):
        r = mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert r.statistic == 0.0
        assert r.p_value < 0.01

    def test_identical_distributions_are_not_significant(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
        assert mann_whitney_u(a, b).p_value > 0.05

    def test_handles_heavy_ties(self):
        # Without the tie correction the variance is overstated and p is wrong.
        a = [1] * 20 + [2] * 20
        b = [1] * 20 + [2] * 20
        r = mann_whitney_u(a, b)
        assert r.p_value == pytest.approx(1.0, abs=0.05)

    def test_detects_a_shift_in_skewed_data(self):
        # The case Mann-Whitney exists for: lognormal, where means mislead.
        rng = np.random.default_rng(6)
        a = rng.lognormal(0, 1, 300)
        b = rng.lognormal(0.8, 1, 300)
        assert mann_whitney_u(a, b).p_value < 0.001


class TestBenjaminiHochberg:
    def test_matches_the_original_1995_paper(self):
        # Benjamini & Hochberg (1995), Table 1: 4 rejections at alpha = 0.05.
        p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
             0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
        out = benjamini_hochberg(p, alpha=0.05)
        assert int(out["reject"].sum()) == 4

    def test_adjusted_values_are_monotone_in_the_original_order(self):
        p = [0.01, 0.02, 0.03, 0.04, 0.05]
        out = benjamini_hochberg(p).sort_values("p_value")
        assert out["p_adjusted"].is_monotonic_increasing

    def test_adjusted_values_never_exceed_one(self):
        out = benjamini_hochberg([0.9, 0.95, 0.99])
        assert (out["p_adjusted"] <= 1.0).all()

    def test_is_less_conservative_than_bonferroni(self):
        # The whole reason for choosing FDR over family-wise control.
        p = [0.001, 0.008, 0.02, 0.03, 0.04, 0.2, 0.5]
        bh = benjamini_hochberg(p, alpha=0.05)
        bonf = bonferroni(p, alpha=0.05)
        assert bh["reject"].sum() >= bonf["reject"].sum()

    def test_empty_input_is_handled(self):
        assert benjamini_hochberg([]).empty

    def test_all_null_p_values_reject_nothing(self):
        rng = np.random.default_rng(9)
        # Under the null, p-values are uniform.
        out = benjamini_hochberg(rng.uniform(0, 1, 100), alpha=0.05)
        assert out["reject"].sum() <= 5   # FDR control at 5%


class TestWilsonInterval:
    def test_matches_published_worked_example(self):
        # Wilson at 5/10, 95%: [0.2366, 0.7634]
        ci = wilson_interval(5, 10)
        assert ci.low == pytest.approx(0.2366, abs=0.0005)
        assert ci.high == pytest.approx(0.7634, abs=0.0005)

    def test_stays_inside_zero_one_at_the_boundary(self):
        # Wald gives [0, 0] here, which claims impossible certainty.
        ci = wilson_interval(0, 20)
        assert ci.low == 0.0 and 0 < ci.high < 1

        ci = wilson_interval(20, 20)
        assert ci.high == 1.0 and 0 < ci.low < 1

    def test_wald_fails_where_wilson_holds(self):
        # The reason Wilson is the project default: Wald's lower bound goes
        # negative at small counts, which is not a probability.
        wald = wald_interval(1, 30)
        wilson = wilson_interval(1, 30)
        assert wald.low < 0
        assert wilson.low >= 0

    def test_at_the_real_uci_conversion_rate_the_two_differ(self):
        # 11.27% of 41,188 -- the actual campaign figure this project reports.
        n, k = 41_188, 4_640
        wilson = wilson_interval(k, n)
        wald = wald_interval(k, n)
        assert wilson.low != pytest.approx(wald.low, abs=1e-9)
        assert wilson.contains(0.1127)

    def test_interval_narrows_as_n_grows(self):
        assert wilson_interval(50, 100).width > wilson_interval(500, 1000).width

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            wilson_interval(5, 0)
        with pytest.raises(ValueError):
            wilson_interval(11, 10)


class TestMeanAndBootstrap:
    def test_mean_interval_covers_the_true_mean(self):
        rng = np.random.default_rng(11)
        x = rng.normal(100, 15, 500)
        assert mean_interval(x).contains(100)

    def test_bootstrap_is_reproducible(self):
        x = np.random.default_rng(12).normal(0, 1, 200)
        a = bootstrap_interval(x, n_resamples=200)
        b = bootstrap_interval(x, n_resamples=200)
        assert a.low == b.low and a.high == b.high

    def test_bootstrap_works_for_the_median(self):
        rng = np.random.default_rng(13)
        x = rng.lognormal(0, 1, 400)
        ci = bootstrap_interval(x, np.median, n_resamples=300)
        assert ci.low < np.median(x) < ci.high


class TestLogisticRecovery:
    """The estimator must return the parameters the data was generated from."""

    @pytest.fixture(scope="class")
    def known_fit(self):
        rng = np.random.default_rng(42)
        n = 20_000
        X = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.normal(0, 1, n),   # true effect is exactly zero
        })
        eta = -1.5 + 0.8 * X["x1"] - 1.2 * X["x2"]
        y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
        return fit_logistic(X, y), {"(intercept)": -1.5, "x1": 0.8,
                                    "x2": -1.2, "x3": 0.0}

    def test_every_confidence_interval_covers_the_truth(self, known_fit):
        fit, truth = known_fit
        for i, name in enumerate(fit.feature_names):
            assert fit.ci_low[i] <= truth[name] <= fit.ci_high[i], \
                f"{name} CI misses the true value"

    def test_coefficients_are_close_to_the_truth(self, known_fit):
        fit, truth = known_fit
        for i, name in enumerate(fit.feature_names):
            assert fit.coefficients[i] == pytest.approx(truth[name], abs=0.06)

    def test_the_null_feature_is_not_significant(self, known_fit):
        # A feature with no effect must not be reported as a driver.
        fit, _ = known_fit
        assert fit.p_values[3] > 0.05

    def test_real_features_are_significant(self, known_fit):
        fit, _ = known_fit
        assert fit.p_values[1] < 0.001 and fit.p_values[2] < 0.001

    def test_it_converges(self, known_fit):
        fit, _ = known_fit
        assert fit.converged and fit.n_iterations < 20

    def test_standard_errors_are_positive(self, known_fit):
        fit, _ = known_fit
        assert (fit.standard_errors > 0).all()

    def test_odds_ratios_are_exponentiated_coefficients(self, known_fit):
        fit, _ = known_fit
        assert np.allclose(fit.odds_ratios, np.exp(fit.coefficients))

    def test_likelihood_ratio_test_rejects_the_null_model(self, known_fit):
        fit, _ = known_fit
        assert fit.lr_statistic > 0 and fit.lr_p_value < 1e-10

    def test_summary_frame_has_inference_columns(self, known_fit):
        # The columns sklearn does not give you, and the reason this exists.
        fit, _ = known_fit
        cols = set(fit.summary_frame().columns)
        assert {"std_error", "z_value", "p_value", "or_ci_low"} <= cols


class TestLogisticEdgeCases:
    def test_constant_outcome_raises(self):
        with pytest.raises(ValueError, match="constant"):
            fit_logistic(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), [1, 1, 1])

    def test_non_binary_outcome_raises(self):
        with pytest.raises(ValueError, match="binary"):
            fit_logistic(pd.DataFrame({"x": [1.0, 2.0]}), [0, 2])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            fit_logistic(pd.DataFrame({"x": [1.0, 2.0]}), [0, 1, 0])

    def test_separation_is_flagged_not_silently_accepted(self):
        # A perfectly separating feature gives huge coefficients that look
        # impressive and mean nothing.
        X = pd.DataFrame({"x": [-5.0, -4, -3, 3, 4, 5]})
        fit = fit_logistic(X, [0, 0, 0, 1, 1, 1], max_iter=60)
        assert fit.separation_warning

    def test_predictions_are_probabilities(self):
        rng = np.random.default_rng(15)
        X = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        y = rng.binomial(1, 0.5, 500)
        fit = fit_logistic(X, y)
        p = fit.predict_proba(X)
        assert ((p >= 0) & (p <= 1)).all()


class TestMetrics:
    def test_auc_of_perfect_ranking_is_one(self):
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_auc_of_reversed_ranking_is_zero(self):
        assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0

    def test_auc_of_constant_scores_is_one_half(self):
        assert roc_auc([0, 1, 0, 1], [0.5] * 4) == 0.5

    def test_auc_is_undefined_with_one_class(self):
        assert math.isnan(roc_auc([1, 1, 1], [0.1, 0.5, 0.9]))

    def test_confusion_matrix_cells_are_right(self):
        cm = confusion_matrix([1, 1, 0, 0], [0.9, 0.4, 0.8, 0.1], threshold=0.5)
        assert (cm.tp, cm.fn, cm.fp, cm.tn) == (1, 1, 1, 1)

    def test_precision_and_recall_hand_calculation(self):
        cm = confusion_matrix([1, 1, 1, 0], [0.9, 0.8, 0.2, 0.7], threshold=0.5)
        assert cm.precision == pytest.approx(2 / 3)
        assert cm.recall == pytest.approx(2 / 3)

    def test_mcc_is_one_for_a_perfect_classifier(self):
        cm = confusion_matrix([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2])
        assert cm.mcc == pytest.approx(1.0)

    def test_accuracy_is_misleading_under_imbalance(self):
        # The reason accuracy is not the headline metric: predicting all-zero
        # on a 1% positive rate scores 99%.
        y = [0] * 99 + [1]
        scores = [0.0] * 100
        cm = confusion_matrix(y, scores, threshold=0.5)
        assert cm.accuracy == pytest.approx(0.99)
        assert cm.recall == 0.0
        assert cm.mcc == 0.0

    def test_brier_score_rewards_calibration(self):
        confident_right = brier_score([1, 1, 0, 0], [0.99, 0.99, 0.01, 0.01])
        confident_wrong = brier_score([1, 1, 0, 0], [0.01, 0.01, 0.99, 0.99])
        assert confident_right < confident_wrong

    def test_average_precision_is_bounded(self):
        ap = average_precision([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
        assert 0 <= ap <= 1

    def test_ks_statistic_is_bounded(self):
        assert 0 <= ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) <= 1

    def test_lift_table_top_decile_beats_the_base_rate(self):
        rng = np.random.default_rng(17)
        y = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
        s = np.concatenate([rng.beta(2, 5, 900), rng.beta(5, 2, 100)])
        lift = lift_table(y, s)
        assert lift["lift"].iloc[0] > 1.0
        assert lift["cumulative_capture_pct"].iloc[-1] == pytest.approx(100.0, abs=0.1)

    def test_optimal_threshold_beats_the_default(self):
        rng = np.random.default_rng(18)
        y = np.concatenate([np.zeros(950), np.ones(50)]).astype(int)
        s = np.concatenate([rng.beta(2, 8, 950), rng.beta(6, 3, 50)])
        best = optimal_threshold(y, s, metric="f1")
        default = confusion_matrix(y, s, threshold=0.5)
        assert best["value"] >= default.f1

    def test_evaluate_returns_the_full_set(self):
        rng = np.random.default_rng(19)
        y = rng.binomial(1, 0.3, 500)
        s = rng.uniform(0, 1, 500)
        out = evaluate(y, s)
        assert {"roc_auc", "average_precision", "brier_score",
                "ks_statistic", "top_decile_lift"} <= set(out)


class TestKMeans:
    @pytest.fixture(scope="class")
    def blobs(self):
        rng = np.random.default_rng(1)
        return np.vstack([
            rng.normal([0, 0], 0.5, (300, 2)),
            rng.normal([6, 6], 0.5, (300, 2)),
            rng.normal([0, 6], 0.5, (300, 2)),
        ])

    def test_recovers_the_known_number_of_clusters(self, blobs):
        table = choose_k(blobs, range(2, 7), n_init=3)
        assert table.attrs["recommended_k"] == 3

    def test_recovers_the_known_centroids(self, blobs):
        fit = fit_kmeans(blobs, 3, seed=1, n_init=5)
        found = np.sort(fit.centroids, axis=0)
        expected = np.sort(np.array([[0, 0], [6, 6], [0, 6]], dtype=float), axis=0)
        assert np.allclose(found, expected, atol=0.3)

    def test_silhouette_is_high_for_separated_clusters(self, blobs):
        fit = fit_kmeans(blobs, 3, seed=1, n_init=5)
        assert fit.silhouette > 0.7

    def test_silhouette_is_low_for_overlapping_data(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(0, 1, (500, 2))
        fit = fit_kmeans(noise, 4, seed=2, n_init=3)
        assert fit.silhouette < 0.5

    def test_is_deterministic_for_a_fixed_seed(self, blobs):
        a = fit_kmeans(blobs, 3, seed=7, n_init=3)
        b = fit_kmeans(blobs, 3, seed=7, n_init=3)
        assert np.allclose(a.centroids, b.centroids)
        assert a.inertia == pytest.approx(b.inertia)

    def test_inertia_falls_as_k_rises(self, blobs):
        # Monotone by construction; if it ever rises, the fit is broken.
        inertias = [fit_kmeans(blobs, k, seed=3, n_init=3,
                               compute_silhouette=False).inertia
                    for k in (2, 3, 4, 5)]
        assert all(a >= b for a, b in zip(inertias[:-1], inertias[1:], strict=True))

    def test_every_point_is_assigned(self, blobs):
        fit = fit_kmeans(blobs, 3, seed=1, n_init=3)
        assert len(fit.labels) == len(blobs)
        assert set(np.unique(fit.labels)) <= {0, 1, 2}

    def test_k_larger_than_n_raises(self):
        with pytest.raises(ValueError):
            fit_kmeans(np.array([[1.0], [2.0]]), 5)

    def test_silhouette_of_a_single_cluster_is_zero(self):
        assert silhouette_score(np.random.default_rng(4).normal(0, 1, (50, 2)),
                                np.zeros(50, dtype=int)) == 0.0

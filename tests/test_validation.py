"""Tests for the validation framework.

The statistical functions are checked against values computed independently by
hand or taken from published references, not against whatever the code happens
to return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meridian.common.exceptions import ValidationError
from meridian.validation import anomaly, profiling
from meridian.validation.engine import validate
from meridian.validation.rules import (
    Severity,
    foreign_key,
    in_range,
    in_set,
    non_negative,
    not_null,
    row_count_between,
    unique,
)
from meridian.validation.suites import fdic_suite, uci_campaign_suite


class TestRulePrimitives:
    def test_not_null_counts_missing(self):
        df = pd.DataFrame({"a": [1, None, 3, None]})
        r = not_null("a").run(df, "t")
        assert not r.passed and r.n_failed == 2 and r.n_checked == 4

    def test_unique_flags_every_member_of_a_duplicate_group(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        r = unique("id").run(df, "t")
        assert r.n_failed == 2  # both rows of the pair

    def test_in_range_is_inclusive_at_the_bounds(self):
        df = pd.DataFrame({"x": [0, 5, 10, 11, -1]})
        r = in_range("x", 0, 10).run(df, "t")
        assert r.n_failed == 2

    def test_non_negative_allows_zero(self):
        assert non_negative("x").run(pd.DataFrame({"x": [0, 1]}), "t").passed

    def test_in_set_rejects_unlisted_values(self):
        df = pd.DataFrame({"c": ["a", "b", "z"]})
        assert in_set("c", ["a", "b"]).run(df, "t").n_failed == 1

    def test_foreign_key_flags_orphans(self):
        parent = pd.DataFrame({"id": [1, 2]})
        child = pd.DataFrame({"parent_id": [1, 2, 99]})
        r = foreign_key("parent_id", parent, "id").run(child, "t")
        assert r.n_failed == 1

    def test_row_count_between_fails_the_whole_table(self):
        r = row_count_between(10, 20).run(pd.DataFrame({"a": [1, 2]}), "t")
        assert not r.passed

    def test_empty_table_passes_vacuously(self):
        r = not_null("a").run(pd.DataFrame(columns=["a"]), "t")
        assert r.passed and r.n_checked == 0

    def test_rule_naming_an_absent_column_does_not_apply(self):
        assert not not_null("nope").applies_to(pd.DataFrame({"a": [1]}))

    def test_fail_rate_is_a_proportion(self):
        df = pd.DataFrame({"a": [None, None, 1, 1]})
        assert not_null("a").run(df, "t").fail_rate == 0.5


class TestEngine:
    def test_error_severity_blocks_the_build(self):
        df = pd.DataFrame({"a": [None]})
        rep = validate(df, [not_null("a", Severity.ERROR)], table="t")
        assert not rep.passed
        with pytest.raises(ValidationError):
            rep.raise_if_failed()

    def test_warning_severity_does_not_block(self):
        df = pd.DataFrame({"a": [None]})
        rep = validate(df, [not_null("a", Severity.WARNING)], table="t")
        assert rep.passed          # build proceeds
        assert len(rep.warnings) == 1  # but it is recorded
        rep.raise_if_failed()      # must not raise

    def test_missing_columns_are_skipped_not_failed(self):
        rep = validate(pd.DataFrame({"a": [1]}), [not_null("absent")], table="t")
        assert rep.passed and len(rep.skipped) == 1 and not rep.results

    def test_summary_counts_add_up(self):
        df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
        rep = validate(df, [not_null("a"), not_null("b")], table="t")
        s = rep.summary()
        assert s["n_rules_run"] == 2 and s["n_passed"] == 1 and s["n_failed"] == 1

    def test_report_serialises_to_json(self, tmp_path):
        rep = validate(pd.DataFrame({"a": [1]}), [not_null("a")], table="t")
        p = rep.save(tmp_path / "r.json")
        assert p.is_file()


class TestBenford:
    """Checked against the published law, not against our own output."""

    def test_expected_frequencies_match_log10_1_plus_1_over_d(self):
        # Canonical values: 30.1%, 17.6%, 12.5% ... 4.6%
        assert anomaly.BENFORD_EXPECTED[1] == pytest.approx(0.30103, abs=1e-5)
        assert anomaly.BENFORD_EXPECTED[2] == pytest.approx(0.17609, abs=1e-5)
        assert anomaly.BENFORD_EXPECTED[9] == pytest.approx(0.04576, abs=1e-5)

    def test_expected_frequencies_sum_to_one(self):
        assert sum(anomaly.BENFORD_EXPECTED.values()) == pytest.approx(1.0, abs=1e-12)

    def test_first_digit_ignores_scale_and_sign(self):
        s = pd.Series([1.0, -19.0, 1900.0, 0.0019, 23.0, 900.0])
        assert anomaly.first_digit(s).tolist() == [1, 1, 1, 1, 2, 9]

    def test_first_digit_drops_zero_and_non_finite(self):
        s = pd.Series([0.0, np.nan, np.inf, 5.0])
        assert anomaly.first_digit(s).tolist() == [5]

    def test_benford_conforming_sample_passes(self):
        # 10^U is Benford-distributed by construction.
        rng = np.random.default_rng(7)
        s = pd.Series(10 ** (rng.uniform(0, 6, 5000)))
        res = anomaly.benford_test(s)
        assert res.conforms and res.mad < 0.012

    def test_benford_uniform_sample_fails(self):
        # Uniform leading digits are the classic fabrication signature.
        rng = np.random.default_rng(7)
        s = pd.Series(rng.integers(100, 999, 3000).astype(float))
        assert not anomaly.benford_test(s).conforms

    def test_truncated_range_does_not_conform(self):
        # Documents why the FDIC peer panel is not the Benford reference: a band
        # spanning ~2 orders of magnitude cannot follow the law.
        rng = np.random.default_rng(7)
        s = pd.Series(10 ** rng.uniform(6, 8, 2000))  # 2 orders only
        narrow = anomaly.benford_test(s)
        wide = anomaly.benford_test(pd.Series(10 ** rng.uniform(3, 9, 2000)))
        assert wide.chi2 < narrow.chi2

    def test_empty_input_is_handled(self):
        assert anomaly.benford_test(pd.Series([], dtype=float)).n == 0

    def test_chi2_critical_values_are_the_published_8df_ones(self):
        assert anomaly.CHI2_CRIT_8DF[0.05] == 15.507
        assert anomaly.CHI2_CRIT_8DF[0.01] == 20.090


class TestMadOutliers:
    def test_detects_a_spike_the_mean_would_mask(self):
        s = pd.Series([10, 11, 10, 12, 11, 10, 11, 5000])
        assert anomaly.mad_outliers(s).sum() == 1

    def test_is_not_dragged_by_multiple_outliers(self):
        # A z-score would inflate its own std and miss these; MAD does not.
        s = pd.Series([10] * 20 + [900, 950, 1000])
        assert anomaly.mad_outliers(s).sum() == 3

    def test_constant_series_reports_nothing(self):
        assert anomaly.mad_outliers(pd.Series([5.0] * 10)).sum() == 0

    def test_falls_back_to_iqr_when_mad_is_zero(self):
        # Majority identical => MAD is 0; the fallback still catches the tail.
        s = pd.Series([5.0] * 30 + [9999.0])
        assert anomaly.mad_outliers(s).sum() >= 1

    def test_threshold_is_the_iglewicz_hoaglin_default(self):
        import inspect
        sig = inspect.signature(anomaly.mad_outliers)
        assert sig.parameters["threshold"].default == 3.5


class TestRoundNumberBias:
    def test_detects_round_amounts(self):
        s = pd.Series([100, 200, 300, 1000])
        assert anomaly.round_number_bias(s)["pct_multiple_of_100"] == 1.0

    def test_organic_amounts_score_low(self):
        s = pd.Series([1234.56, 87.21, 9903.11, 45.9])
        assert anomaly.round_number_bias(s)["pct_multiple_of_100"] == 0.0


class TestStructuralBreak:
    def test_finds_a_planted_level_shift(self):
        s = pd.Series([10.0] * 20 + [30.0] * 20)
        out = anomaly.detect_structural_break(s)
        assert out["found"] and out["break_index"] == 20
        assert out["mean_before"] == pytest.approx(10.0)
        assert out["mean_after"] == pytest.approx(30.0)

    def test_reports_nothing_for_a_short_series(self):
        assert not anomaly.detect_structural_break(pd.Series([1.0, 2.0]))["found"]

    def test_pct_change_is_signed_correctly(self):
        s = pd.Series([100.0] * 15 + [50.0] * 15)
        assert anomaly.detect_structural_break(s)["pct_change"] == pytest.approx(-0.5)


class TestDuplicateClusters:
    def test_groups_exact_repeats(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out = anomaly.duplicate_clusters(df, ["a", "b"])
        assert len(out) == 1 and out.iloc[0]["n_rows"] == 2


class TestProfiling:
    def test_numeric_profile_reports_quantiles(self):
        p = profiling.profile_column(pd.Series([1, 2, 3, 4, 5]), "x")
        assert p.is_numeric and p.median == 3 and p.minimum == 1 and p.maximum == 5

    def test_missingness_is_a_proportion(self):
        p = profiling.profile_column(pd.Series([1, None, None, 4]), "x")
        assert p.n_missing == 2 and p.pct_missing == 0.5

    def test_constant_column_is_flagged(self):
        assert profiling.profile_column(pd.Series([7, 7, 7]), "x").is_constant

    def test_categorical_profile_reports_top_values(self):
        p = profiling.profile_column(pd.Series(["a", "a", "b"]), "c")
        assert not p.is_numeric and p.top_values["a"] == 2

    def test_profile_table_covers_every_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert len(profiling.profile_table(df, table="t")) == 2

    def test_correlation_pairs_finds_a_derived_column(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "c": [9.0, 1, 7, 3]})
        out = profiling.correlation_pairs(df, threshold=0.95)
        assert len(out) == 1 and {out.iloc[0].column_a, out.iloc[0].column_b} == {"a", "b"}


class TestRealSuites:
    """The suites encode facts about the real sources; assert those facts."""

    def test_uci_row_count_is_pinned_exactly(self):
        names = [r.name for r in uci_campaign_suite()]
        assert "row_count_between[41188,41188]" in names

    def test_uci_suite_forbids_the_leaky_column(self):
        rule = next(r for r in uci_campaign_suite()
                    if r.name == "no_leaky_duration_column")
        leaky = pd.DataFrame({"age": [30], "duration": [200]})
        clean = pd.DataFrame({"age": [30]})
        assert not rule.run(leaky, "t").passed
        assert rule.run(clean, "t").passed

    def test_fdic_suite_rejects_dollar_scale_in_nimy(self):
        rule = next(r for r in fdic_suite()
                    if r.name == "nimy_is_percent_scale_not_dollars")
        # The real trap: CERT 628 reported NIM 46,983,000 and NIMY 3.02.
        assert not rule.run(pd.DataFrame({"NIMY": [46_983_000.0]}), "t").passed
        assert rule.run(pd.DataFrame({"NIMY": [3.02]}), "t").passed

    def test_fdic_suite_enforces_deposits_within_assets(self):
        rule = next(r for r in fdic_suite() if r.name == "deposits_do_not_exceed_assets")
        bad = pd.DataFrame({"DEP": [200.0], "ASSET": [100.0]})
        good = pd.DataFrame({"DEP": [80.0], "ASSET": [100.0]})
        assert not rule.run(bad, "t").passed and rule.run(good, "t").passed


class TestReportRendering:
    def test_html_is_self_contained_and_states_the_verdict(self):
        from meridian.validation.report import render_html
        rep = validate(pd.DataFrame({"a": [1]}), [not_null("a")], table="t")
        html = render_html(rep)
        assert html.startswith("<!doctype html>")
        assert "<style>" in html          # inline CSS, no external assets
        assert "http://" not in html and "https://" not in html
        assert "BUILD PASSES" in html

    def test_failing_build_is_announced(self):
        from meridian.validation.report import render_html
        rep = validate(pd.DataFrame({"a": [None]}), [not_null("a")], table="t")
        assert "BUILD BLOCKED" in render_html(rep)

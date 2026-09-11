"""Hypothesis tests, built on the from-scratch distributions.

Three choices here are deliberate and each is the less common default.

**Welch, not Student.** :func:`t_test` does not assume equal variances. Segment
comparisons in this project put 89 private-banking customers against 21,909 mass
customers, with wildly different spreads; the pooled-variance test is badly
miscalibrated in exactly that situation, and it is the default in most textbooks
and in scipy.

**Benjamini-Hochberg, always.** The analytics layer runs dozens of segment
comparisons. At alpha 0.05, twenty independent true nulls produce one false
positive on average, so uncorrected testing does not merely risk a spurious
finding -- it guarantees one. :func:`benjamini_hochberg` controls the false
discovery rate rather than the family-wise error rate, because with this many
comparisons Bonferroni would leave nothing detectable.

**Effect sizes with every test.** A p-value answers "is there a difference",
which at n = 25,000 is almost always yes. The magnitude is the business
question, so every result carries one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .distributions import chi2_sf, f_sf, normal_sf, t_two_sided_p


@dataclass
class TestResult:
    """Outcome of one hypothesis test."""

    test: str
    statistic: float
    p_value: float
    df: float | None = None
    effect_size: float | None = None
    effect_name: str = ""
    n1: int = 0
    n2: int = 0
    estimate: float | None = None      # the quantity being tested
    ci_low: float | None = None
    ci_high: float | None = None
    alpha: float = 0.05
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "statistic": round(self.statistic, 6),
            "p_value": self.p_value,
            "df": self.df,
            "significant": self.significant,
            "effect_size": None if self.effect_size is None
                           else round(self.effect_size, 6),
            "effect_name": self.effect_name,
            "n1": self.n1, "n2": self.n2,
            "estimate": None if self.estimate is None else round(self.estimate, 6),
            "ci_low": None if self.ci_low is None else round(self.ci_low, 6),
            "ci_high": None if self.ci_high is None else round(self.ci_high, 6),
            "note": self.note,
            **self.extra,
        }

    def summary(self) -> str:
        sig = "significant" if self.significant else "not significant"
        parts = [f"{self.test}: statistic={self.statistic:.4f}, p={self.p_value:.4g} ({sig})"]
        if self.effect_size is not None:
            parts.append(f"{self.effect_name}={self.effect_size:.4f}")
        if self.ci_low is not None:
            parts.append(f"95% CI [{self.ci_low:.4f}, {self.ci_high:.4f}]")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Continuous outcomes
# ---------------------------------------------------------------------------


def t_test(a, b, *, equal_var: bool = False, alpha: float = 0.05) -> TestResult:
    """Two-sample t-test. Welch's by default.

    Welch does not assume equal variances and is barely less powerful when they
    happen to be equal, so it is the better default -- particularly for segment
    comparisons where group sizes differ by two orders of magnitude.
    """
    x = np.asarray(pd.Series(a).dropna(), dtype=float)
    y = np.asarray(pd.Series(b).dropna(), dtype=float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"need at least 2 observations per group, got {n1} and {n2}")

    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)

    if equal_var:
        df = n1 + n2 - 2
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(pooled * (1 / n1 + 1 / n2))
        name = "Student t-test (equal variance)"
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        # Welch-Satterthwaite degrees of freedom.
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else n1 + n2 - 2
        name = "Welch t-test"

    if se == 0:
        return TestResult(name, 0.0, 1.0, df, 0.0, "Cohen's d", n1, n2,
                          estimate=m1 - m2, alpha=alpha,
                          note="zero standard error: both groups are constant")

    t_stat = (m1 - m2) / se
    p = t_two_sided_p(t_stat, df)

    # Cohen's d on the pooled standard deviation.
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0

    from .distributions import t_ppf
    crit = t_ppf(1 - alpha / 2, df)

    return TestResult(
        test=name, statistic=t_stat, p_value=p, df=df,
        effect_size=d, effect_name="Cohen's d", n1=n1, n2=n2,
        estimate=m1 - m2,
        ci_low=(m1 - m2) - crit * se, ci_high=(m1 - m2) + crit * se,
        alpha=alpha,
        extra={"mean1": round(m1, 6), "mean2": round(m2, 6),
               "sd1": round(math.sqrt(v1), 6), "sd2": round(math.sqrt(v2), 6)},
    )


def paired_t_test(a, b, *, alpha: float = 0.05) -> TestResult:
    """Paired t-test on the within-pair differences."""
    df_pair = pd.DataFrame({"a": pd.Series(a), "b": pd.Series(b)}).dropna()
    if len(df_pair) < 2:
        raise ValueError("need at least 2 complete pairs")
    diff = (df_pair["a"] - df_pair["b"]).to_numpy(dtype=float)
    n = len(diff)
    mean_d, sd_d = diff.mean(), diff.std(ddof=1)
    if sd_d == 0:
        # Every pair moved by exactly the same amount. If that amount is zero
        # there is nothing to detect; if it is not, the shift is perfectly
        # consistent and the evidence against a null of no change is as strong
        # as the data can express. Returning p=1.0 in both cases -- as an
        # earlier version did -- reported "no effect" for a dataset where every
        # single observation moved in the same direction by the same amount.
        if mean_d == 0:
            return TestResult("Paired t-test", 0.0, 1.0, n - 1, 0.0, "Cohen's dz",
                              n, n, estimate=0.0, alpha=alpha,
                              note="all differences are zero")
        return TestResult(
            "Paired t-test", math.inf if mean_d > 0 else -math.inf, 0.0,
            n - 1, math.inf if mean_d > 0 else -math.inf, "Cohen's dz",
            n, n, estimate=mean_d, ci_low=mean_d, ci_high=mean_d, alpha=alpha,
            note=("every pair changed by exactly the same non-zero amount; "
                  "the standard error is zero, so the t statistic is unbounded"),
        )
    se = sd_d / math.sqrt(n)
    t_stat = mean_d / se
    from .distributions import t_ppf
    crit = t_ppf(1 - alpha / 2, n - 1)
    return TestResult(
        "Paired t-test", t_stat, t_two_sided_p(t_stat, n - 1), n - 1,
        mean_d / sd_d, "Cohen's dz", n, n, estimate=mean_d,
        ci_low=mean_d - crit * se, ci_high=mean_d + crit * se, alpha=alpha,
    )


def anova_oneway(*groups, alpha: float = 0.05) -> TestResult:
    """One-way ANOVA across k groups.

    Assumes equal variances; :func:`levene_test` checks that assumption, and
    when it fails the pairwise Welch tests are the better route.
    """
    arrays = [np.asarray(pd.Series(g).dropna(), dtype=float) for g in groups]
    arrays = [a for a in arrays if len(a) > 0]
    k = len(arrays)
    if k < 2:
        raise ValueError("ANOVA needs at least two non-empty groups")

    n_total = sum(len(a) for a in arrays)
    grand_mean = sum(a.sum() for a in arrays) / n_total

    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
    ss_within = sum(((a - a.mean()) ** 2).sum() for a in arrays)
    df_between, df_within = k - 1, n_total - k

    if df_within <= 0 or ss_within == 0:
        return TestResult("One-way ANOVA", 0.0, 1.0, df_between, alpha=alpha,
                          note="zero within-group variance")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    p = f_sf(f_stat, df_between, df_within)

    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return TestResult(
        "One-way ANOVA", f_stat, p, df_between, eta_sq, "eta squared",
        n1=n_total, n2=k, alpha=alpha,
        extra={"df_within": df_within, "ss_between": round(ss_between, 4),
               "ss_within": round(ss_within, 4), "k_groups": k},
    )


def levene_test(*groups, alpha: float = 0.05) -> TestResult:
    """Levene's test for equal variances, using the median (Brown-Forsythe).

    The median-centred variant is markedly more robust to non-normal data than
    the mean-centred original, which matters here because every monetary
    variable in this project is right-skewed.
    """
    arrays = [np.asarray(pd.Series(g).dropna(), dtype=float) for g in groups]
    arrays = [a for a in arrays if len(a) > 1]
    if len(arrays) < 2:
        raise ValueError("Levene needs at least two groups with 2+ observations")
    z_groups = [np.abs(a - np.median(a)) for a in arrays]
    return TestResult(
        "Levene (Brown-Forsythe)", *_f_from_groups(z_groups), alpha=alpha,
    )


def _f_from_groups(groups: list[np.ndarray]) -> tuple[float, float, float]:
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand = sum(g.sum() for g in groups) / n_total
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = k - 1, n_total - k
    if ss_w == 0 or df_w <= 0:
        return 0.0, 1.0, float(df_b)
    f = (ss_b / df_b) / (ss_w / df_w)
    return f, f_sf(f, df_b, df_w), float(df_b)


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------


def proportion_z_test(
    successes1: int, n1: int, successes2: int, n2: int, *, alpha: float = 0.05
) -> TestResult:
    """Two-proportion z-test with a pooled variance under the null."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both samples must be non-empty")
    p1, p2 = successes1 / n1, successes2 / n2
    p_pool = (successes1 + successes2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        return TestResult("Two-proportion z-test", 0.0, 1.0, None, 0.0,
                          "Cohen's h", n1, n2, estimate=p1 - p2, alpha=alpha,
                          note="no variation in either group")

    z = (p1 - p2) / se
    p_value = 2.0 * normal_sf(abs(z))

    # Cohen's h: the arcsine-transformed difference, which unlike the raw
    # difference is comparable across the range (0.05 -> 0.10 is a bigger
    # change than 0.50 -> 0.55).
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

    # Unpooled SE for the interval, since the CI is not built under the null.
    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    from .distributions import normal_ppf
    crit = normal_ppf(1 - alpha / 2)

    return TestResult(
        "Two-proportion z-test", z, p_value, None, h, "Cohen's h", n1, n2,
        estimate=p1 - p2,
        ci_low=(p1 - p2) - crit * se_unpooled,
        ci_high=(p1 - p2) + crit * se_unpooled,
        alpha=alpha,
        extra={"p1": round(p1, 6), "p2": round(p2, 6),
               "successes1": successes1, "successes2": successes2},
    )


def chi_square_independence(table, *, alpha: float = 0.05) -> TestResult:
    """Chi-squared test of independence with Cramer's V.

    Cramer's V is reported because chi-squared itself scales with n: at 25,000
    rows almost any association is "significant", and V is the number that says
    whether it matters.
    """
    obs = np.asarray(table, dtype=float)
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2:
        raise ValueError("contingency table must be at least 2x2")

    n = obs.sum()
    if n == 0:
        raise ValueError("contingency table is empty")

    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n

    if (expected < 5).mean() > 0.2:
        note = ("more than 20% of expected counts are below 5; the "
                "chi-squared approximation is unreliable here")
    else:
        note = ""

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0)
    chi2 = float(terms.sum())
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    p = chi2_sf(chi2, df)

    # Cramer's V, bounded in [0, 1].
    v = math.sqrt(chi2 / (n * (min(obs.shape) - 1))) if n > 0 else 0.0

    return TestResult(
        "Chi-squared independence", chi2, p, df, v, "Cramer's V",
        n1=int(n), alpha=alpha, note=note,
        extra={"shape": list(obs.shape),
               "min_expected": round(float(expected.min()), 4)},
    )


# ---------------------------------------------------------------------------
# Non-parametric
# ---------------------------------------------------------------------------


def mann_whitney_u(a, b, *, alpha: float = 0.05) -> TestResult:
    """Mann-Whitney U, with a normal approximation and tie correction.

    The right test for the monetary variables in this project: balances and
    transaction amounts are lognormal, so comparing means is comparing a
    statistic the distribution barely supports. This compares distributions
    without assuming either is normal.
    """
    x = np.asarray(pd.Series(a).dropna(), dtype=float)
    y = np.asarray(pd.Series(b).dropna(), dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("both samples must be non-empty")

    combined = np.concatenate([x, y])
    ranks = _rank_with_ties(combined)
    r1 = ranks[:n1].sum()

    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mu_u = n1 * n2 / 2
    # Tie correction to the variance: without it, heavily tied data gives an
    # inflated z and a p-value that is too small.
    _, counts = np.unique(combined, return_counts=True)
    n = n1 + n2
    tie_term = (counts ** 3 - counts).sum()
    sigma_u = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))) \
        if n > 1 else 0.0

    if sigma_u == 0:
        return TestResult("Mann-Whitney U", float(u), 1.0, None, 0.0,
                          "rank-biserial r", n1, n2, alpha=alpha,
                          note="zero variance after tie correction")

    z = (u - mu_u) / sigma_u
    p = 2.0 * normal_sf(abs(z))

    # Rank-biserial correlation: the probability of superiority, rescaled.
    rb = 1 - 2 * u / (n1 * n2)

    return TestResult(
        "Mann-Whitney U", float(u), p, None, rb, "rank-biserial r", n1, n2,
        alpha=alpha,
        extra={"u1": float(u1), "u2": float(u2), "z": round(z, 6),
               "median1": float(np.median(x)), "median2": float(np.median(y))},
    )


def _rank_with_ties(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties sharing the mean of the positions they occupy."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction.

    Controls the expected proportion of false discoveries among the rejected
    hypotheses, rather than the probability of any false discovery. With dozens
    of segment comparisons, Bonferroni's family-wise control is so conservative
    that real effects go undetected; FDR is the appropriate trade for
    exploratory segment analysis, and saying which one was used is part of
    reporting the result honestly.

    Returns a frame with the original p-values, the adjusted values, and the
    reject decision, in the input order.
    """
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    if m == 0:
        return pd.DataFrame(columns=["p_value", "p_adjusted", "reject", "rank"])

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # Step-up: adjusted p is the running minimum from the largest downwards,
    # which enforces monotonicity of the adjusted values.
    adjusted_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    reject_sorted = ranked <= ranks / m * alpha
    # Reject everything up to the largest index that passes.
    if reject_sorted.any():
        cutoff = np.max(np.flatnonzero(reject_sorted))
        reject_sorted = np.zeros(m, dtype=bool)
        reject_sorted[:cutoff + 1] = True
    reject = np.empty(m, dtype=bool)
    reject[order] = reject_sorted

    rank_of = np.empty(m, dtype=int)
    rank_of[order] = ranks

    return pd.DataFrame({
        "p_value": p, "p_adjusted": adjusted, "reject": reject, "rank": rank_of,
    })


def bonferroni(p_values, *, alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni correction, for comparison against BH."""
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    if m == 0:
        return pd.DataFrame(columns=["p_value", "p_adjusted", "reject"])
    adjusted = np.clip(p * m, 0, 1)
    return pd.DataFrame({
        "p_value": p, "p_adjusted": adjusted, "reject": adjusted < alpha,
    })

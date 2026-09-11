"""Confidence intervals.

The Wilson interval is the default for proportions, not the Wald interval, and
that choice is load-bearing for this project. The real UCI campaign conversion
rate is 11.27%, and Wald -- the p +/- z*sqrt(p(1-p)/n) form that every textbook
introduces first -- is badly miscalibrated at proportions that far from 0.5. Its
actual coverage falls well below the nominal 95%, and for small or extreme
samples it produces bounds outside [0, 1], which is not merely inaccurate but
impossible.

Wilson costs one extra line of algebra and is correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .distributions import normal_ppf, t_ppf


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n: int = 0

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def margin(self) -> float:
        """Half-width. Only meaningful for symmetric intervals."""
        return self.width / 2

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": round(self.estimate, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "confidence": self.confidence,
            "method": self.method,
            "n": self.n,
        }

    def __str__(self) -> str:
        return (f"{self.estimate:.4f} "
                f"[{self.low:.4f}, {self.high:.4f}] "
                f"({self.confidence:.0%} {self.method})")


def wilson_interval(
    successes: int, n: int, *, confidence: float = 0.95
) -> Interval:
    """Wilson score interval for a proportion.

    Derived by inverting the score test rather than assuming the estimate is
    normally distributed. It stays inside [0, 1] by construction, behaves
    sensibly at zero and at n successes, and holds close to nominal coverage
    even at proportions near the boundary -- all of which Wald fails.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must lie in [0, {n}], got {successes}")

    z = normal_ppf(1 - (1 - confidence) / 2)
    p = successes / n
    z2 = z * z

    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator

    low, high = centre - spread, centre + spread
    # Snap floating-point residue at the boundaries. With zero successes the
    # algebra gives a lower bound of ~1e-17 rather than 0, which is harmless
    # arithmetically but reads as a nonzero probability in a report.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0

    return Interval(
        estimate=p,
        low=max(0.0, low),
        high=min(1.0, high),
        confidence=confidence, method="Wilson score", n=n,
    )


def wald_interval(
    successes: int, n: int, *, confidence: float = 0.95
) -> Interval:
    """Wald interval for a proportion.

    Provided for comparison only -- :func:`wilson_interval` is what the analysis
    uses. Keeping this here makes the difference demonstrable rather than
    asserted: at the real 11.27% conversion rate the two disagree materially,
    and Wald can return a lower bound below zero.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = normal_ppf(1 - (1 - confidence) / 2)
    p = successes / n
    margin = z * math.sqrt(p * (1 - p) / n)
    return Interval(p, p - margin, p + margin, confidence, "Wald (biased)", n)


def agresti_coull_interval(
    successes: int, n: int, *, confidence: float = 0.95
) -> Interval:
    """Agresti-Coull: Wald applied after adding two successes and two failures.

    Simpler to explain than Wilson and nearly as well calibrated; included so
    the report can show that the choice of method is not doing the work.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = normal_ppf(1 - (1 - confidence) / 2)
    n_tilde = n + z * z
    p_tilde = (successes + z * z / 2) / n_tilde
    margin = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return Interval(successes / n, max(0.0, p_tilde - margin),
                    min(1.0, p_tilde + margin), confidence, "Agresti-Coull", n)


def mean_interval(values, *, confidence: float = 0.95) -> Interval:
    """t-interval for a mean.

    Uses the t distribution rather than the normal because sigma is estimated
    from the sample. The difference is negligible at n = 25,000 and material at
    the segment level, where a group may have a few dozen members.
    """
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 observations")
    mean = float(x.mean())
    se = float(x.std(ddof=1)) / math.sqrt(n)
    crit = t_ppf(1 - (1 - confidence) / 2, n - 1)
    return Interval(mean, mean - crit * se, mean + crit * se,
                    confidence, "t-interval", n)


def bootstrap_interval(
    values, statistic=np.mean, *, confidence: float = 0.95,
    n_resamples: int = 2000, seed: int = 20260911,
) -> Interval:
    """Percentile bootstrap interval for an arbitrary statistic.

    The general fallback when no closed form exists -- a median, a ratio of
    sums, a trimmed mean. Percentile method: resample with replacement, compute
    the statistic each time, and take the empirical quantiles.

    Seeded, so a reported interval is reproducible. An unseeded bootstrap gives
    slightly different bounds on every run, which makes a report impossible to
    check.
    """
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 observations")

    rng = np.random.default_rng(seed)
    # Vectorised resampling: one (n_resamples, n) index matrix rather than a
    # Python loop, which matters at 2,000 resamples of 25,000 rows.
    idx = rng.integers(0, n, size=(n_resamples, n))
    stats = np.apply_along_axis(statistic, 1, x[idx])

    alpha = 1 - confidence
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return Interval(float(statistic(x)), float(low), float(high),
                    confidence, f"bootstrap percentile ({n_resamples})", n)


def difference_in_proportions_interval(
    successes1: int, n1: int, successes2: int, n2: int, *,
    confidence: float = 0.95,
) -> Interval:
    """Newcombe's interval for a difference of two proportions.

    Built from the two Wilson intervals rather than from a normal approximation
    to the difference, so it inherits Wilson's calibration.
    """
    w1 = wilson_interval(successes1, n1, confidence=confidence)
    w2 = wilson_interval(successes2, n2, confidence=confidence)
    p1, p2 = successes1 / n1, successes2 / n2
    diff = p1 - p2
    low = diff - math.sqrt((p1 - w1.low) ** 2 + (w2.high - p2) ** 2)
    high = diff + math.sqrt((w1.high - p1) ** 2 + (p2 - w2.low) ** 2)
    return Interval(diff, max(-1.0, low), min(1.0, high), confidence,
                    "Newcombe (Wilson-based)", n1 + n2)


def proportion_table(
    df: pd.DataFrame, group: str, outcome: str, *, confidence: float = 0.95,
) -> pd.DataFrame:
    """Conversion rate per group with Wilson intervals.

    The table behind every campaign chart: a rate alone invites comparing two
    groups whose intervals overlap completely.
    """
    rows = []
    for name, sub in df.groupby(group, observed=True):
        n = len(sub)
        k = int(sub[outcome].sum())
        ci = wilson_interval(k, n, confidence=confidence)
        rows.append({
            group: name, "n": n, "successes": k,
            "rate": round(ci.estimate, 6),
            "ci_low": round(ci.low, 6), "ci_high": round(ci.high, 6),
            "ci_width": round(ci.width, 6),
        })
    return pd.DataFrame(rows).sort_values("rate", ascending=False).reset_index(drop=True)

"""Churn as a discrete-time hazard, with frailty, censoring and a hidden break.

Churn is generated month by month from a hazard, not assigned as a label. That
matters because it produces the structure real attrition has -- a customer at
risk each month until they leave or the window ends -- which is what makes
survival analysis, cohort triangles and time-varying covariates meaningful
rather than decorative.

Three features exist to stop the downstream analysis being circular. A generator
whose rule a model can recover exactly produces an AUC of 0.99 and proves
nothing.

**Frailty.** Each customer carries an unobserved multiplier on their hazard.
No model in this project sees it. It is the irreducible noise that attenuates
recovered coefficients the way real unobserved heterogeneity does.

**Right censoring.** Customers still active when the window closes are censored,
not counted as retained forever. Treating censored observations as non-events
is the classic survival-analysis error and it biases every retention estimate
downward; generating real censoring means the analysis has to handle it.

**An undisclosed structural break.** A step change in the hazard is planted at a
month the analyst is not told. The report states plainly which findings are
structural-by-construction and which emerged -- and this one is meant to be
found by :func:`meridian.validation.anomaly.detect_structural_break`, not
assumed.

The linear predictor is documented here in full, because the honest claim is
"a logistic model should recover these coefficients, attenuated by frailty",
and that claim is testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..common.logging import get_logger
from .calibration import CalibrationProfile
from .customers import GenerationWindow

log = get_logger(__name__)


@dataclass(frozen=True)
class HazardSpec:
    """Coefficients of the monthly churn hazard, on the log-odds scale.

    These are the ground truth. tests/test_generation.py asserts that a logistic
    regression fitted to the generated panel recovers them in sign and rough
    magnitude -- the check that the analytical layer works, run against a case
    where the answer is known.
    """

    # Calibrated so the monthly hazard implies roughly 12% annual attrition at
    # the mean covariate profile, which is the middle of the 10-20% band Indian
    # retail banks report for savings-account attrition. An earlier value of
    # -4.35 produced 2.7% a year: arithmetically fine, but no real deposit book
    # retains customers that well, and every retention figure downstream would
    # have been quietly implausible.
    intercept: float = -2.55

    # Tenure: attrition is highest early, then settles. Modelled on log tenure
    # so the decline is steep at first and flattens, as observed in real books.
    log_tenure: float = -0.32

    # Engagement: the strongest real-world predictor. A customer who has stopped
    # transacting has usually already left.
    txn_recency_months: float = 0.44
    n_products: float = -0.38
    digital_engagement: float = -0.55

    # Economics
    log_balance: float = -0.22
    fee_paid_recently: float = 0.21

    # Demographics
    age_scaled: float = -0.16
    is_salaried: float = -0.19

    # Service experience
    complaint_last_quarter: float = 0.68

    # Macro: rates rising makes competing deposit offers more attractive.
    rate_delta_pp: float = 0.13

    # The planted break, applied from break_month onward. Sized to survive the
    # composition drift of a growing book: the at-risk population is ~4,000x
    # larger at the end of the window than the start and its mean tenure climbs
    # throughout, so a smaller shock is masked by the tenure term and the break
    # becomes undetectable in the aggregate series. At 0.85 log-odds the shift
    # is recoverable by detect_structural_break() without being so large that
    # finding it is trivial.
    structural_break_log_odds: float = 0.85

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass
class ChurnResult:
    """Generated churn panel plus the ground truth used to make it."""

    panel: pd.DataFrame          # customer x month, at-risk rows
    outcomes: pd.DataFrame       # one row per customer: churn_date or censored
    spec: HazardSpec
    break_month: pd.Timestamp
    truth: dict[str, Any] = field(default_factory=dict)

    @property
    def churn_rate(self) -> float:
        return float(self.outcomes["churned"].mean())


def generate_churn(
    customers: pd.DataFrame,
    holdings: pd.DataFrame,
    profile: CalibrationProfile,
    window: GenerationWindow,
    *,
    rng: np.random.Generator | None = None,
    break_at_fraction: float = 0.62,
) -> ChurnResult:
    """Simulate monthly churn over the window.

    Returns the at-risk panel (one row per customer per month they were still a
    customer) and per-customer outcomes with explicit censoring flags.
    """
    rng = rng or np.random.default_rng(profile.seed + 2)
    spec = HazardSpec()

    months = window.months
    n_months = len(months)
    break_idx = int(n_months * break_at_fraction)
    break_month = months[break_idx]

    n = len(customers)
    cust_ids = customers["customer_id"].to_numpy()
    acquired = pd.to_datetime(customers["acquired_date"]).to_numpy()
    frailty = customers["frailty"].to_numpy()
    digital = customers["digital_engagement"].to_numpy()
    age = customers["age"].to_numpy()
    salaried = customers["is_salaried"].to_numpy()

    # Static per-customer aggregates from the holdings book.
    prod_count = (
        holdings.groupby("customer_id").size()
        .reindex(cust_ids, fill_value=0).to_numpy()
    )
    balance = (
        holdings.groupby("customer_id")["balance_inr"].sum()
        .reindex(cust_ids, fill_value=0.0).to_numpy()
    )
    fees = (
        holdings.groupby("customer_id")["annual_fee_inr"].sum()
        .reindex(cust_ids, fill_value=0.0).to_numpy()
    )

    log_balance_z = _zscore(np.log1p(balance))
    age_scaled = (age - 40) / 15.0
    fee_flag = (fees > 0).astype(float)

    # Macro path: the real lending rate drifts over the window, so the rate
    # term in the hazard is exogenous rather than invented per customer.
    rate_path = _rate_path(profile, n_months, rng)

    active = np.ones(n, dtype=bool)
    churn_month_idx = np.full(n, -1, dtype=int)

    # Latent engagement state, evolving month to month.
    recency = np.zeros(n)
    complaint_state = np.zeros(n)

    panel_rows: list[pd.DataFrame] = []

    for m_idx, month in enumerate(months):
        # A customer is at risk only once acquired and while still active.
        joined = acquired <= np.datetime64(month)
        at_risk = active & joined
        if not at_risk.any():
            continue

        tenure_m = _tenure_months(acquired, month)
        log_tenure = np.log1p(np.maximum(tenure_m, 0))

        # Engagement drifts: mostly customers transact, sometimes they lapse.
        # Digitally engaged customers lapse less often.
        lapse_p = np.clip(0.16 - 0.10 * digital, 0.02, 0.4)
        lapsed = rng.random(n) < lapse_p
        recency = np.where(lapsed, recency + 1, 0.0)

        # Complaints arrive as a rare event with a one-quarter memory.
        new_complaint = rng.random(n) < 0.021
        complaint_state = np.where(
            new_complaint, 3.0, np.maximum(complaint_state - 1, 0)
        )

        linear = (
            spec.intercept
            + spec.log_tenure * log_tenure
            + spec.txn_recency_months * np.minimum(recency, 12)
            + spec.n_products * prod_count
            + spec.digital_engagement * digital
            + spec.log_balance * log_balance_z
            + spec.fee_paid_recently * fee_flag
            + spec.age_scaled * age_scaled
            + spec.is_salaried * salaried
            + spec.complaint_last_quarter * (complaint_state > 0)
            + spec.rate_delta_pp * rate_path[m_idx]
        )
        if m_idx >= break_idx:
            linear = linear + spec.structural_break_log_odds

        # Frailty multiplies the hazard, which is an additive shift in log-odds.
        hazard = _sigmoid(linear + np.log(frailty))
        hazard = np.where(at_risk, hazard, 0.0)

        churned_now = (rng.random(n) < hazard) & at_risk

        panel_rows.append(pd.DataFrame({
            "customer_id": cust_ids[at_risk],
            "month": month,
            "tenure_months": tenure_m[at_risk],
            "n_products": prod_count[at_risk],
            "total_balance_inr": balance[at_risk],
            "txn_recency_months": np.minimum(recency, 12)[at_risk],
            "complaint_active": (complaint_state > 0)[at_risk].astype(int),
            "digital_engagement": digital[at_risk],
            "lending_rate_delta_pp": rate_path[m_idx],
            "hazard": hazard[at_risk],
            "churned": churned_now[at_risk].astype(int),
        }))

        churn_month_idx = np.where(churned_now, m_idx, churn_month_idx)
        active = active & ~churned_now

    panel = pd.concat(panel_rows, ignore_index=True)

    churned = churn_month_idx >= 0
    # Build via a pandas Series: np.array(..., dtype=datetime64) cannot mix NaT
    # with Timestamps and raises rather than coercing.
    churn_dates = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    churn_dates.loc[churned] = months[churn_month_idx[churned]]

    outcomes = pd.DataFrame({
        "customer_id": cust_ids,
        "acquired_date": pd.to_datetime(acquired),
        "churn_date": churn_dates,
        "churned": churned.astype(int),
        # Explicit censoring flag: still active when the window closed. Treating
        # these as retained-forever is the error this column exists to prevent.
        "censored": (~churned).astype(int),
        "observed_months": np.where(
            churned,
            churn_month_idx - _month_index(acquired, months),
            len(months) - _month_index(acquired, months),
        ).clip(min=0),
    })

    result = ChurnResult(
        panel=panel, outcomes=outcomes, spec=spec, break_month=break_month,
        truth={
            "coefficients": spec.to_dict(),
            "break_month": str(break_month.date()),
            "break_month_index": break_idx,
            "structural_break_log_odds": spec.structural_break_log_odds,
            "frailty_sigma": 0.6,
            "note": (
                "Frailty is unobserved by every model in this project, so "
                "recovered coefficients are expected to be attenuated towards "
                "zero relative to these values."
            ),
        },
    )

    log.info(
        "churn: %d at-risk rows | %.2f%% of customers churned | %.2f%% censored "
        "| break planted at %s",
        len(panel), result.churn_rate * 100,
        float(outcomes["censored"].mean()) * 100, break_month.date(),
    )
    return result


# --- helpers ---------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped to avoid overflow warnings in the tails; exp(-700) underflows.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _zscore(a: np.ndarray) -> np.ndarray:
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a, dtype=float)


def _tenure_months(acquired: np.ndarray, month: pd.Timestamp) -> np.ndarray:
    acq = pd.to_datetime(acquired)
    return ((month.year - acq.year) * 12 + (month.month - acq.month)).to_numpy()


def _month_index(acquired: np.ndarray, months: pd.DatetimeIndex) -> np.ndarray:
    acq = pd.to_datetime(acquired)
    first = months[0]
    return np.clip(
        ((acq.year - first.year) * 12 + (acq.month - first.month)).to_numpy(),
        0, len(months),
    )


def _rate_path(
    profile: CalibrationProfile, n_months: int, rng: np.random.Generator
) -> np.ndarray:
    """Deviation of the lending rate from its starting level, in points.

    A random walk with mild mean reversion, centred so the first month is zero.
    The level is anchored to the real calibrated rate; only the deviation enters
    the hazard, so the coefficient is interpretable as "per point of rate move".
    """
    steps = rng.normal(0, 0.14, n_months)
    path = np.cumsum(steps)
    path = path - 0.06 * np.arange(n_months) * np.sign(path.mean() or 1)
    return path - path[0]

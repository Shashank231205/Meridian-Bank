"""Customer spine: demographics, wealth, tenure and segment.

Every customer attribute traces to a calibrated parameter rather than a taste
judgement. Demographics are bootstrapped from the real UCI joint distribution,
so the correlations between age, job, education and marital status are
inherited rather than invented; wealth is drawn from the fitted lognormal and
then *shifted* by demographic group, because income varies by occupation and
life stage in a way a single global draw would erase.

Two design choices exist specifically to stop the analysis being circular.

**Frailty.** Each customer carries an unobserved ``frailty`` multiplier -- a
latent propensity to churn that no model in this project gets to see. Without
it, a churn model fitted downstream would recover the generator's own rule
exactly and report an implausible AUC. With it, there is irreducible noise, and
the recovered coefficients are attenuated the way real ones are.

**Acquisition cohorts.** Customers join over time rather than all at once, which
is what makes cohort retention analysis meaningful instead of a formality.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..common.logging import get_logger
from .calibration import CalibrationProfile

log = get_logger(__name__)

# Meridian's footprint. Weights reflect a mid-size Indian retail bank's
# concentration in metros, with a long tail of smaller cities.
CITIES: tuple[tuple[str, str, float], ...] = (
    ("Mumbai", "Maharashtra", 0.18),
    ("Delhi", "Delhi", 0.14),
    ("Bengaluru", "Karnataka", 0.12),
    ("Chennai", "Tamil Nadu", 0.09),
    ("Hyderabad", "Telangana", 0.08),
    ("Pune", "Maharashtra", 0.07),
    ("Kolkata", "West Bengal", 0.07),
    ("Ahmedabad", "Gujarat", 0.06),
    ("Jaipur", "Rajasthan", 0.05),
    ("Lucknow", "Uttar Pradesh", 0.04),
    ("Kochi", "Kerala", 0.04),
    ("Indore", "Madhya Pradesh", 0.03),
    ("Chandigarh", "Punjab", 0.03),
)

# Acquisition channel. Branch still dominates the back book at an Indian retail
# bank, but digital acquisition grows sharply over the generated window; that
# drift is applied in :func:`_acquisition_channel` rather than fixed here.
CHANNELS: tuple[str, ...] = ("branch", "digital", "partner", "telesales", "dsa")

SEGMENTS: tuple[str, ...] = ("mass", "affluent", "priority", "private")

# Income multipliers by occupation, applied to the calibrated wealth draw.
# Relative ordering follows the occupational income structure visible in the
# UCI job field; the magnitudes are conservative.
JOB_WEALTH_MULTIPLIER: dict[str, float] = {
    "management": 2.4, "entrepreneur": 2.0, "self-employed": 1.6,
    "admin.": 1.0, "technician": 1.1, "services": 0.8, "blue-collar": 0.7,
    "housemaid": 0.5, "retired": 0.9, "student": 0.25, "unemployed": 0.35,
    "unknown": 0.9,
}

EDUCATION_WEALTH_MULTIPLIER: dict[str, float] = {
    "university.degree": 1.5, "professional.course": 1.3,
    "high.school": 1.0, "basic.9y": 0.8, "basic.6y": 0.7, "basic.4y": 0.65,
    "illiterate": 0.5, "unknown": 0.95,
}


@dataclass(frozen=True)
class GenerationWindow:
    """The period over which the synthetic bank operates."""

    start: date
    end: date

    @property
    def months(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="MS")

    @property
    def n_months(self) -> int:
        return len(self.months)


def generate_customers(
    profile: CalibrationProfile,
    window: GenerationWindow,
    *,
    n: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate the customer dimension.

    Returns one row per customer at ``customer_id`` grain.
    """
    n = n or profile.n_customers
    rng = rng or np.random.default_rng(profile.seed)

    demo = _sample_demographics(profile, n, rng)
    df = pd.DataFrame({"customer_id": _customer_ids(n)})

    for col in demo.columns:
        df[col] = demo[col].to_numpy()

    df["age"] = _age_within_band(df["age_band"], rng)
    df["gender"] = rng.choice(["F", "M"], size=n, p=[0.47, 0.53])

    # --- geography ------------------------------------------------------
    idx = rng.choice(
        len(CITIES), size=n, p=_normalise([c[2] for c in CITIES])
    )
    df["city"] = [CITIES[i][0] for i in idx]
    df["state"] = [CITIES[i][1] for i in idx]

    # --- acquisition ----------------------------------------------------
    df["acquired_date"] = _acquisition_dates(window, n, rng)
    df["acquisition_channel"] = _acquisition_channel(df["acquired_date"], window, rng)
    df["tenure_months"] = _months_between(df["acquired_date"], window.end)

    # --- wealth ---------------------------------------------------------
    df["annual_income_inr"] = _income(profile, df, rng)
    df["segment"] = _segment(df["annual_income_inr"])
    df["credit_score"] = _credit_score(df, rng)

    # --- latent ---------------------------------------------------------
    # Unobserved heterogeneity. Lognormal with mean 1 so it scales hazards
    # without shifting the population rate. Never exposed to any model.
    df["frailty"] = rng.lognormal(mean=-0.18, sigma=0.6, size=n)
    df["digital_engagement"] = _digital_engagement(df, rng)

    df["is_salaried"] = df["job"].isin(
        ["admin.", "technician", "services", "management", "blue-collar"]
    ).astype(int)

    log.info(
        "generated %d customers | median income INR %s | segments %s",
        len(df), f"{df['annual_income_inr'].median():,.0f}",
        df["segment"].value_counts().to_dict(),
    )
    return df


# --- components ------------------------------------------------------------

def _customer_ids(n: int) -> list[str]:
    return [f"CUS{i:07d}" for i in range(1, n + 1)]


def _sample_demographics(
    profile: CalibrationProfile, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Bootstrap demographic combinations from the real joint distribution."""
    j = profile.demographics
    if j is None or not j.combinations:
        raise ValueError("calibration profile carries no demographic distribution")

    picks = rng.choice(len(j.combinations), size=n, p=_normalise(j.weights))
    rows = [j.combinations[i] for i in picks]
    return pd.DataFrame(rows, columns=list(j.columns))


def _age_within_band(bands: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Spread ages uniformly inside each real band.

    The joint distribution is over bands, but a dimension needs an age. Drawing
    uniformly within the band preserves the band proportions exactly while
    avoiding 25,000 customers aged exactly 30.
    """
    bounds = {
        "<25": (18, 24), "25-34": (25, 34), "35-44": (35, 44),
        "45-54": (45, 54), "55-64": (55, 64), "65+": (65, 88),
    }
    out = np.empty(len(bands), dtype=int)
    for band, (lo, hi) in bounds.items():
        mask = (bands == band).to_numpy()
        if mask.any():
            out[mask] = rng.integers(lo, hi + 1, size=int(mask.sum()))
    # Any band not in the map (shouldn't happen) falls back to the real mean.
    unmapped = ~np.isin(bands.to_numpy(), list(bounds))
    if unmapped.any():
        out[unmapped] = 40
    return out


def _acquisition_dates(
    window: GenerationWindow, n: int, rng: np.random.Generator
) -> pd.Series:
    """Draw acquisition dates with a growing book.

    Weighted towards later months so the bank is visibly growing, which is what
    makes a cohort triangle interesting: equal cohorts would make every diagonal
    the same size and hide the effect of cohort quality on retention.
    """
    months = window.months
    # Linear growth in monthly acquisition, roughly 2.5x from first to last.
    weights = np.linspace(1.0, 2.5, len(months))
    weights = weights / weights.sum()
    picks = rng.choice(len(months), size=n, p=weights)

    chosen = months[picks]
    # Spread within the month rather than stacking every customer on the 1st.
    day_offset = rng.integers(0, 28, size=n)
    return pd.Series(chosen) + pd.to_timedelta(day_offset, unit="D")


def _acquisition_channel(
    acquired: pd.Series, window: GenerationWindow, rng: np.random.Generator
) -> np.ndarray:
    """Assign channel with digital share rising over the window.

    A static channel mix would make the channel-migration analysis vacuous. The
    drift here is the structural trend the analyst is meant to find.
    """
    n = len(acquired)
    span_days = max((window.end - window.start).days, 1)
    progress = (
        (pd.to_datetime(acquired) - pd.Timestamp(window.start)).dt.days / span_days
    ).clip(0, 1).to_numpy()

    out = np.empty(n, dtype=object)
    for i in range(n):
        t = progress[i]
        p = np.array([
            0.46 - 0.22 * t,   # branch declines
            0.18 + 0.26 * t,   # digital grows
            0.14 - 0.02 * t,   # partner roughly flat
            0.12 - 0.01 * t,   # telesales slight decline
            0.10 - 0.01 * t,   # dsa slight decline
        ])
        out[i] = CHANNELS[rng.choice(len(CHANNELS), p=p / p.sum())]
    return out


def _months_between(start: pd.Series, end: date) -> np.ndarray:
    s = pd.to_datetime(start)
    e = pd.Timestamp(end)
    return ((e.year - s.dt.year) * 12 + (e.month - s.dt.month)).clip(lower=0).to_numpy()


def _income(
    profile: CalibrationProfile, df: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Draw annual income from the calibrated lognormal, shifted by group.

    The base draw carries the calibrated dispersion; the multipliers introduce
    the occupational and educational structure that makes segmentation
    meaningful. Age contributes a mild career-progression effect.
    """
    w = profile.wealth
    if w is None:
        raise ValueError("calibration profile carries no wealth distribution")

    base = rng.lognormal(mean=w.mu, sigma=w.sigma, size=len(df))

    job_mult = df["job"].map(JOB_WEALTH_MULTIPLIER).fillna(1.0).to_numpy()
    edu_mult = df["education"].map(EDUCATION_WEALTH_MULTIPLIER).fillna(1.0).to_numpy()

    # Earnings rise into the fifties then taper; a plain linear term would make
    # 80-year-olds the wealthiest customers in the book.
    age = df["age"].to_numpy()
    age_mult = 0.55 + 0.9 * np.exp(-((age - 52) ** 2) / (2 * 17 ** 2))

    income = base * job_mult * edu_mult * age_mult * 2.4

    # Floor the left tail. The calibrated lognormal is fitted to a balance-scale
    # quantity and its lower tail runs to implausibly small numbers -- an
    # unfloored draw produced a minimum of INR 800 a year and a 25th percentile
    # of INR 88,500. People with no income do not hold the products modelled
    # here, so the banked population is truncated rather than extrapolated: the
    # floor is set near India's per-capita GDP, the same World Bank figure the
    # wealth channel is anchored to, and students and the unemployed are allowed
    # below it because for them a near-zero income is the real case.
    floor = 180_000.0
    low_income_ok = df["job"].isin(["student", "unemployed"]).to_numpy()
    soft_floor = np.where(low_income_ok, floor * 0.25, floor)

    # Smooth rather than clamp: a hard floor would pile a visible spike of
    # identical incomes at exactly the threshold, which every histogram in the
    # dashboard would show as an artefact.
    below = income < soft_floor
    income[below] = soft_floor[below] * (0.75 + 0.35 * rng.random(int(below.sum())))

    return np.round(income, -2)


def _segment(income: pd.Series) -> np.ndarray:
    """Assign a wealth segment by income threshold.

    Thresholds are round INR figures a bank would actually use for its
    proposition tiers rather than quantiles of our own output, so the segment
    mix is a property of the generated book and can be compared across runs.
    """
    inc = income.to_numpy()
    out = np.full(len(inc), "mass", dtype=object)
    out[inc >= 1_200_000] = "affluent"
    out[inc >= 3_000_000] = "priority"
    out[inc >= 10_000_000] = "private"
    return out


def _credit_score(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """CIBIL-style score, 300-900, correlated with income and age.

    Correlated rather than independent because a score unrelated to the rest of
    the record would make every downstream risk finding meaningless.
    """
    n = len(df)
    income_z = _zscore(np.log1p(df["annual_income_inr"].to_numpy()))
    age_z = _zscore(df["age"].to_numpy())
    latent = 0.45 * income_z + 0.25 * age_z + rng.normal(0, 0.85, n)
    score = 690 + 95 * latent
    return np.clip(np.round(score), 300, 900).astype(int)


def _digital_engagement(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """A 0-1 propensity to transact through digital channels.

    Falls with age and rises with income; customers acquired digitally start
    higher. Drives channel mix in the transaction generator.
    """
    n = len(df)
    age_term = -0.035 * (df["age"].to_numpy() - 40)
    income_term = 0.30 * _zscore(np.log1p(df["annual_income_inr"].to_numpy()))
    digital_acq = (df["acquisition_channel"].to_numpy() == "digital").astype(float)
    latent = 0.2 + age_term + income_term + 0.9 * digital_acq + rng.normal(0, 0.7, n)
    return np.clip(1 / (1 + np.exp(-latent)), 0.01, 0.99)


def _zscore(a: np.ndarray) -> np.ndarray:
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a, dtype=float)


def _normalise(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w / w.sum()

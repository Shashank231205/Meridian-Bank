"""Product holdings: what each customer owns, priced off the real lending rate.

Product economics are not invented. The interest a loan earns and a deposit
costs both derive from the World Bank's real India lending rate and the spread
implied by the real FDIC peer net interest margin, so the revenue the portfolio
generates is anchored to observable banking economics rather than chosen to
produce a pleasing chart.

Ownership is modelled as sequential cross-sell rather than independent draws.
A customer opens a savings account, and only then becomes eligible for a credit
card or a loan. Independent sampling would produce customers holding a mortgage
but no current account, and would destroy the cross-sell structure that the
product-affinity analysis depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common.logging import get_logger
from .calibration import CalibrationProfile
from .customers import GenerationWindow

log = get_logger(__name__)


@dataclass(frozen=True)
class ProductSpec:
    """Definition of one product in the catalogue.

    ``rate_offset_pp`` is expressed in percentage points relative to the real
    calibrated lending rate, so every price moves with the real rate rather than
    being hardcoded.
    """

    code: str
    name: str
    category: str          # deposit | lending | card | investment
    rate_offset_pp: float  # relative to the calibrated lending rate
    base_adoption: float   # baseline P(hold), before customer effects
    min_income_inr: float
    annual_fee_inr: float
    is_anchor: bool = False  # anchor products gate the rest


CATALOGUE: tuple[ProductSpec, ...] = (
    # Deposits pay a rate below the lending rate -- that gap is the margin.
    ProductSpec("SAV", "Savings Account", "deposit", -5.2, 0.97, 0, 0, is_anchor=True),  # pays ~3.4%
    ProductSpec("CUR", "Current Account", "deposit", -8.0, 0.16, 600_000, 1_200),
    ProductSpec("FD", "Fixed Deposit", "deposit", -1.8, 0.34, 250_000, 0),
    ProductSpec("RD", "Recurring Deposit", "deposit", -2.4, 0.19, 180_000, 0),
    # Lending. Offsets are set so the balance-weighted yield across the book
    # sits about 4pp over the blended cost of funds, which is what produces a
    # net interest margin inside the real FDIC peer band. Home loans dominate
    # the book by value and are priced close to the reference rate, exactly as
    # they are in the Indian market -- a secured 20-year mortgage is not a
    # high-margin product, and treating it as one was what pushed the generated
    # NIMY to 5.2% against a real peer band of 3.7-4.2%.
    ProductSpec("PL", "Personal Loan", "lending", +3.0, 0.13, 300_000, 0),
    ProductSpec("HL", "Home Loan", "lending", -0.3, 0.09, 800_000, 0),
    ProductSpec("AL", "Auto Loan", "lending", +0.9, 0.08, 500_000, 0),
    ProductSpec("GL", "Gold Loan", "lending", +1.2, 0.06, 0, 0),
    # Cards carry a headline rate far above the book, but revolve on only a
    # fraction of the outstanding balance, so the effective yield is lower than
    # the sticker rate implies.
    ProductSpec("CC", "Credit Card", "card", +14.0, 0.28, 360_000, 500),
    ProductSpec("MF", "Mutual Fund", "investment", 0.0, 0.15, 500_000, 0),
    ProductSpec("INS", "Insurance", "investment", 0.0, 0.21, 300_000, 0),
    ProductSpec("DMT", "Demat Account", "investment", 0.0, 0.09, 700_000, 300),
)

CATALOGUE_BY_CODE = {p.code: p for p in CATALOGUE}

# Segment multipliers on adoption. Wealthier customers hold more products --
# the cross-sell gradient that makes segment analysis worth doing.
SEGMENT_ADOPTION_MULTIPLIER = {
    "mass": 0.75, "affluent": 1.25, "priority": 1.7, "private": 2.1,
}


def product_dimension(profile: CalibrationProfile) -> pd.DataFrame:
    """The product dimension, priced off the real calibrated rate."""
    lending = profile.lending_rate_pct or 8.567143
    rows = []
    for p in CATALOGUE:
        rate = max(lending + p.rate_offset_pp, 0.0)
        rows.append({
            "product_code": p.code,
            "product_name": p.name,
            "category": p.category,
            "interest_rate_pct": round(rate, 4),
            "rate_offset_pp": p.rate_offset_pp,
            "annual_fee_inr": p.annual_fee_inr,
            "min_income_inr": p.min_income_inr,
            "is_anchor": int(p.is_anchor),
            # Recorded so the dimension itself documents its own provenance.
            "calibrated_from_lending_rate_pct": round(lending, 4),
        })
    df = pd.DataFrame(rows)
    log.info("product dimension: %d products priced off lending rate %.4f%%",
             len(df), lending)
    return df


def generate_holdings(
    customers: pd.DataFrame,
    profile: CalibrationProfile,
    window: GenerationWindow,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate product holdings at customer x product grain.

    Ownership is sequential: the anchor product is opened first and everything
    else is conditioned on holding it, so the resulting book has a coherent
    cross-sell structure.
    """
    rng = rng or np.random.default_rng(profile.seed + 1)
    n = len(customers)

    income = customers["annual_income_inr"].to_numpy()
    digital = customers["digital_engagement"].to_numpy()
    credit = customers["credit_score"].to_numpy()
    tenure = customers["tenure_months"].to_numpy()
    seg_mult = customers["segment"].map(SEGMENT_ADOPTION_MULTIPLIER).fillna(1.0).to_numpy()

    records: list[dict] = []
    anchor = next(p for p in CATALOGUE if p.is_anchor)

    # --- anchor -------------------------------------------------------
    holds_anchor = rng.random(n) < anchor.base_adoption
    for p in CATALOGUE:
        if p.is_anchor:
            eligible = holds_anchor
            prob = np.where(eligible, 1.0, 0.0)
        else:
            # Income eligibility is a hard gate, as it is in a real bank.
            income_ok = income >= p.min_income_inr
            # Longer-tenured customers have had more cross-sell opportunities;
            # the effect saturates rather than growing without bound.
            tenure_mult = 0.55 + 0.45 * (1 - np.exp(-tenure / 24.0))
            # Credit-gated products depend on score.
            credit_mult = (
                np.clip((credit - 550) / 250, 0.05, 1.6)
                if p.category in ("lending", "card") else 1.0
            )
            # Investment products skew digital.
            digital_mult = (
                0.6 + 0.9 * digital if p.category == "investment" else 1.0
            )
            prob = (
                p.base_adoption * seg_mult * tenure_mult
                * credit_mult * digital_mult
            )
            prob = np.where(holds_anchor & income_ok, prob, 0.0)

        if p.is_anchor:
            # Use the anchor mask itself rather than re-drawing against prob=1.0.
            # A second rng.random(n) < 1.0 is true for almost every customer but
            # it is a *different* draw, so the set written to the holdings table
            # diverged from the set the other products were gated on, leaving
            # customers holding a mortgage and no account.
            held = holds_anchor
        else:
            held = rng.random(n) < np.clip(prob, 0, 0.98)
        idx = np.flatnonzero(held)
        if idx.size == 0:
            continue

        opened = _open_dates(customers, idx, window, rng)
        balances = _balances(p, income[idx], rng)
        records.append(pd.DataFrame({
            "customer_id": customers["customer_id"].to_numpy()[idx],
            "product_code": p.code,
            "category": p.category,
            "opened_date": opened,
            "balance_inr": balances,
            "interest_rate_pct": round(
                max((profile.lending_rate_pct or 8.567143) + p.rate_offset_pp, 0.0), 4
            ),
            "annual_fee_inr": p.annual_fee_inr,
        }))

    out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    log.info(
        "generated %d holdings for %d customers (%.2f products each)",
        len(out), n, len(out) / max(n, 1),
    )
    return out


def _open_dates(
    customers: pd.DataFrame, idx: np.ndarray,
    window: GenerationWindow, rng: np.random.Generator,
) -> pd.Series:
    """Open a product at or after the customer was acquired.

    A holding predating the relationship would be a referential impossibility,
    and exactly the sort of thing the warehouse's grain tests look for.
    """
    # to_datetime over an ndarray yields a DatetimeIndex, so the difference is a
    # TimedeltaIndex -- .days directly, not .dt.days (which is Series-only).
    acquired = pd.DatetimeIndex(pd.to_datetime(customers["acquired_date"].to_numpy()[idx]))
    end = pd.Timestamp(window.end)
    span = np.clip((end - acquired).days.to_numpy(), 0, None)
    # Cross-sell concentrates early in the relationship, so a beta-like draw
    # skewed towards zero is more faithful than a uniform one.
    frac = rng.beta(1.4, 3.0, size=len(idx))
    return pd.Series(acquired + pd.to_timedelta((span * frac).astype(int), unit="D"))


def _balances(
    spec: ProductSpec, income: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw a balance appropriate to the product and the customer's income.

    Multiples of annual income by product type, with lognormal dispersion.
    Lending balances are held as positive magnitudes; the sign convention is
    applied by the revenue layer, which knows what an asset and a liability are.
    """
    n = len(income)
    noise = rng.lognormal(0, 0.55, n)

    multiples = {
        "SAV": 0.18, "CUR": 0.30, "FD": 0.55, "RD": 0.12,
        # Loan multiples are sized so the aggregate loan book lands inside the
        # real FDIC peer loan-to-deposit band (80-93%). Raising these also pulls
        # NIMY down, because a larger share of assets sits in the loan book at
        # its contractual spread rather than in low-yield surplus reserves --
        # the two ratios are not independent knobs.
        "PL": 0.62, "HL": 2.62, "AL": 0.54, "GL": 0.18,
        "CC": 0.09, "MF": 0.45, "INS": 0.25, "DMT": 0.30,
    }
    base = income * multiples.get(spec.code, 0.2) * noise

    # Products with no meaningful balance still carry a nominal one so the
    # monthly snapshot fact has a value to aggregate.
    floor = {"CC": 0.0, "DMT": 0.0}.get(spec.code, 500.0)
    return np.round(np.maximum(base, floor), -1)

"""Transactions and monthly account snapshots.

Two facts come out of this module, and having both is deliberate. A transaction
fact records individual events at event grain; a periodic snapshot records
balances at customer x product x month grain. Kimball's distinction matters
here because the questions differ: "how much did this customer spend on cards in
March" is a transaction question, "what was the deposit book worth in March" is
a snapshot question, and answering the second by summing the first is how people
get balance figures wrong.

Transaction amounts are drawn so that the aggregate conforms to Benford's law
rather than clustering on round numbers, because the project's own anomaly
module tests for exactly that and failing our own check would be embarrassing.
Volume follows real seasonality: an Indian retail bank sees a pronounced
festive-season peak around October and November.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.logging import get_logger
from .calibration import CalibrationProfile
from .customers import GenerationWindow

log = get_logger(__name__)

TXN_TYPES: tuple[tuple[str, str, float, float], ...] = (
    # (code, direction, share of volume, mean amount as fraction of monthly income)
    ("SALARY_CREDIT", "credit", 0.06, 0.85),
    ("UPI_PAYMENT", "debit", 0.34, 0.020),
    ("CARD_SPEND", "debit", 0.16, 0.045),
    ("ATM_WITHDRAWAL", "debit", 0.11, 0.060),
    ("NEFT_TRANSFER", "debit", 0.08, 0.180),
    ("BILL_PAYMENT", "debit", 0.10, 0.035),
    ("EMI_DEBIT", "debit", 0.05, 0.150),
    ("INTEREST_CREDIT", "credit", 0.04, 0.012),
    ("CHEQUE_DEPOSIT", "credit", 0.03, 0.320),
    ("FEE_DEBIT", "debit", 0.03, 0.008),
)

CHANNELS: tuple[str, ...] = ("mobile", "internet", "atm", "branch", "pos", "upi")

# Monthly seasonality multipliers, January through December. The October-November
# festive peak (Navratri, Diwali) is the dominant feature of Indian retail
# spending and is what makes the seasonality analysis show something real.
SEASONALITY: tuple[float, ...] = (
    0.94, 0.90, 1.05, 0.98, 1.00, 0.96,
    0.99, 1.02, 1.08, 1.24, 1.18, 1.06,
)

MERCHANT_CATEGORIES: tuple[str, ...] = (
    "groceries", "fuel", "dining", "travel", "utilities", "healthcare",
    "education", "apparel", "electronics", "entertainment", "insurance", "other",
)


def generate_transactions(
    customers: pd.DataFrame,
    holdings: pd.DataFrame,
    outcomes: pd.DataFrame,
    window: GenerationWindow,
    profile: CalibrationProfile,
    *,
    rng: np.random.Generator | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Generate the transaction fact.

    Customers transact from acquisition until they churn (or the window ends),
    so transaction history and churn outcome are consistent by construction --
    a churned customer has no transactions after their churn date.
    """
    rng = rng or np.random.default_rng(profile.seed + 3)

    active = customers.merge(
        outcomes[["customer_id", "churn_date", "churned"]], on="customer_id", how="left"
    )
    months = window.months

    # Monthly transaction count scales with income and digital engagement.
    income = active["annual_income_inr"].to_numpy()
    digital = active["digital_engagement"].to_numpy()
    # Monthly transaction count per customer. Sized so the fact lands near 1.8M
    # rows over the window rather than the 8.8M an unconstrained draw produced:
    # a mid-size retail bank's engaged customer makes a handful of account
    # transactions a month, not seventeen, and the extra volume bought nothing
    # analytically while making every aggregation slower.
    base_count = 1.4 + 3.4 * digital + 1.1 * _zscore(np.log1p(income))
    base_count = np.clip(base_count, 0.4, 14)

    acquired = pd.to_datetime(active["acquired_date"]).to_numpy()
    churn = pd.to_datetime(active["churn_date"]).to_numpy()

    type_codes = [t[0] for t in TXN_TYPES]
    type_dirs = {t[0]: t[1] for t in TXN_TYPES}
    type_p = np.array([t[2] for t in TXN_TYPES])
    type_p = type_p / type_p.sum()
    type_frac = {t[0]: t[3] for t in TXN_TYPES}

    frames: list[pd.DataFrame] = []
    total = 0

    for month in months:
        m_start = np.datetime64(month)
        m_end = np.datetime64(month + pd.offsets.MonthEnd(0))

        # Active this month: acquired, and not yet churned.
        live = (acquired <= m_end) & (pd.isna(churn) | (churn > m_start))
        idx = np.flatnonzero(live)
        if idx.size == 0:
            continue

        season = SEASONALITY[month.month - 1]
        lam = base_count[idx] * season
        counts = rng.poisson(lam)
        counts = np.minimum(counts, 120)
        n_txn = int(counts.sum())
        if n_txn == 0:
            continue

        cust_rep = np.repeat(active["customer_id"].to_numpy()[idx], counts)
        income_rep = np.repeat(income[idx], counts)
        digital_rep = np.repeat(digital[idx], counts)

        kinds = rng.choice(type_codes, size=n_txn, p=type_p)
        monthly_income = income_rep / 12.0

        # Lognormal dispersion around the type's mean fraction of income keeps
        # amounts spanning several orders of magnitude, which is what makes the
        # aggregate follow Benford rather than piling on round numbers.
        frac = np.array([type_frac[k] for k in kinds])
        amount = monthly_income * frac * rng.lognormal(0, 0.85, n_txn)
        amount = np.round(np.maximum(amount, 10.0), 2)

        directions = np.array([type_dirs[k] for k in kinds])

        # Channel follows digital engagement: engaged customers use mobile/UPI.
        chan_u = rng.random(n_txn)
        channels = np.where(
            chan_u < digital_rep * 0.55, "mobile",
            np.where(chan_u < digital_rep * 0.80, "upi",
                     np.where(chan_u < digital_rep * 0.90, "internet",
                              np.where(chan_u < 0.94, "pos",
                                       np.where(chan_u < 0.98, "atm", "branch")))),
        )

        day = rng.integers(1, pd.Period(month, freq="M").days_in_month + 1, n_txn)
        txn_date = pd.to_datetime(month) + pd.to_timedelta(day - 1, unit="D")

        frames.append(pd.DataFrame({
            "customer_id": cust_rep,
            "txn_date": txn_date,
            "month": month,
            "txn_type": kinds,
            "direction": directions,
            "amount_inr": amount,
            "channel": channels,
            "merchant_category": np.where(
                np.isin(kinds, ["CARD_SPEND", "UPI_PAYMENT", "POS"]),
                rng.choice(MERCHANT_CATEGORIES, size=n_txn), "",
            ),
        }))
        total += n_txn

        if max_rows and total >= max_rows:
            log.warning("transaction cap %d reached at month %s", max_rows, month.date())
            break

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "txn_id", [f"TXN{i:09d}" for i in range(1, len(out) + 1)])
    log.info(
        "generated %s transactions over %d months (%.1f per customer-month)",
        f"{len(out):,}", len(months), len(out) / max(len(customers) * len(months), 1),
    )
    return out


def generate_monthly_snapshots(
    customers: pd.DataFrame,
    holdings: pd.DataFrame,
    outcomes: pd.DataFrame,
    transactions: pd.DataFrame,
    window: GenerationWindow,
    profile: CalibrationProfile,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate the periodic snapshot fact at customer x product x month grain.

    Balances evolve with a drift and noise rather than being restated constant,
    so that trend and seasonality analyses have something real to find. Revenue
    is computed per row from the product's calibrated rate: deposits earn the
    bank the spread between the lending rate and what it pays, lending earns the
    product rate on the outstanding balance.
    """
    rng = rng or np.random.default_rng(profile.seed + 4)
    months = window.months
    lending = profile.lending_rate_pct or 8.567143

    h = holdings.merge(
        outcomes[["customer_id", "churn_date"]], on="customer_id", how="left"
    )
    h["opened_date"] = pd.to_datetime(h["opened_date"])
    h["churn_date"] = pd.to_datetime(h["churn_date"])

    frames: list[pd.DataFrame] = []

    for month in months:
        m_end = month + pd.offsets.MonthEnd(0)
        live = (h["opened_date"] <= m_end) & (
            h["churn_date"].isna() | (h["churn_date"] > month)
        )
        sub = h.loc[live]
        if sub.empty:
            continue

        age_months = (
            (month.year - sub["opened_date"].dt.year) * 12
            + (month.month - sub["opened_date"].dt.month)
        ).clip(lower=0).to_numpy()

        # Balances accrete slowly with tenure, with month-to-month noise.
        growth = (1 + 0.004) ** age_months
        noise = rng.lognormal(0, 0.07, len(sub))
        season = SEASONALITY[month.month - 1]
        balance = sub["balance_inr"].to_numpy() * growth * noise
        # Deposit balances dip in the festive months as customers spend.
        is_deposit = sub["category"].to_numpy() == "deposit"
        balance = np.where(is_deposit, balance * (2 - season), balance * season)
        balance = np.round(np.maximum(balance, 0), 2)

        rate = sub["interest_rate_pct"].to_numpy()
        category = sub["category"].to_numpy()

        # Monthly revenue to the bank.
        #   lending/card: the bank earns the product rate on the balance.
        #   deposit:      the bank earns the gap between what it can lend at and
        #                 what it pays the depositor -- the net interest margin.
        #   investment:   fee income, taken as a thin annual percentage.
        monthly = np.zeros(len(sub))
        lend_mask = np.isin(category, ["lending", "card"])
        dep_mask = category == "deposit"
        inv_mask = category == "investment"
        monthly[lend_mask] = balance[lend_mask] * rate[lend_mask] / 100 / 12
        monthly[dep_mask] = balance[dep_mask] * np.maximum(
            lending - rate[dep_mask], 0.0
        ) / 100 / 12
        monthly[inv_mask] = balance[inv_mask] * 0.9 / 100 / 12

        fee_monthly = sub["annual_fee_inr"].to_numpy() / 12.0

        frames.append(pd.DataFrame({
            "customer_id": sub["customer_id"].to_numpy(),
            "product_code": sub["product_code"].to_numpy(),
            "category": category,
            "month": month,
            "balance_inr": balance,
            "interest_rate_pct": rate,
            "interest_revenue_inr": np.round(monthly, 2),
            "fee_revenue_inr": np.round(fee_monthly, 2),
            "total_revenue_inr": np.round(monthly + fee_monthly, 2),
            "account_age_months": age_months,
        }))

    out = pd.concat(frames, ignore_index=True)
    log.info(
        "generated %s monthly snapshots | total revenue INR %s",
        f"{len(out):,}", f"{out['total_revenue_inr'].sum():,.0f}",
    )
    return out


def _zscore(a: np.ndarray) -> np.ndarray:
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a, dtype=float)

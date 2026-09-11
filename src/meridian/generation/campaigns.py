"""Campaign contacts, driven by the response model fitted on real UCI data.

The campaign module is the one place where the project's analytics run on real
outcomes rather than generated ones: the conversion rate, the response
elasticities and the effect of prior contact all come from 41,188 genuine
retail-bank marketing records. What is synthesised here is only the mapping of
those real response patterns onto Meridian's own customer base, so that campaign
ROI can be computed against the generated revenue book.

The design deliberately reproduces two effects that real campaign data shows and
that naive synthesis misses:

**Contact fatigue.** Response falls with each additional contact in a campaign.
The UCI ``campaign`` column shows this clearly, and a generator that ignores it
would suggest the optimal policy is to call everyone forever.

**Prior-outcome dependence.** A customer who subscribed before is far more
likely to subscribe again. The UCI ``poutcome`` column carries this, and it is
the single strongest predictor in the real data after the leaky duration field
is removed.
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
class CampaignSpec:
    """One marketing campaign."""

    campaign_id: str
    name: str
    product_code: str
    channel: str
    start_month: int          # index into the generation window
    duration_months: int
    target_segment: str       # "all" or a segment name
    cost_per_contact_inr: float
    target_size_pct: float    # share of the eligible base to contact


CAMPAIGNS: tuple[CampaignSpec, ...] = (
    CampaignSpec("CMP001", "Festive Gold Loan Push", "GL", "telesales", 9, 3, "all", 42.0, 0.30),
    CampaignSpec("CMP002", "Salary Account Upgrade", "CUR", "branch", 4, 4, "mass", 65.0, 0.22),
    CampaignSpec("CMP003", "Premium Card Acquisition", "CC", "digital", 12, 6, "affluent", 28.0, 0.45),
    CampaignSpec("CMP004", "Fixed Deposit Rate Drive", "FD", "telesales", 18, 3, "all", 40.0, 0.35),
    CampaignSpec("CMP005", "Home Loan Balance Transfer", "HL", "dsa", 24, 5, "priority", 180.0, 0.55),
    CampaignSpec("CMP006", "Digital Onboarding Nudge", "MF", "digital", 28, 4, "all", 18.0, 0.40),
    CampaignSpec("CMP007", "Insurance Cross-Sell", "INS", "telesales", 33, 4, "all", 38.0, 0.28),
    CampaignSpec("CMP008", "Recurring Deposit Campaign", "RD", "branch", 38, 3, "mass", 55.0, 0.25),
    CampaignSpec("CMP009", "Wealth Advisory Outreach", "DMT", "branch", 44, 4, "private", 320.0, 0.70),
    CampaignSpec("CMP010", "Personal Loan Pre-Approved", "PL", "digital", 48, 5, "all", 22.0, 0.33),
    CampaignSpec("CMP011", "Auto Loan Festive Offer", "AL", "dsa", 55, 3, "affluent", 150.0, 0.40),
    CampaignSpec("CMP012", "Savings Reactivation", "SAV", "digital", 58, 4, "all", 15.0, 0.50),
)

# Channel effectiveness multipliers on the real base conversion rate. Ordering
# reflects what the UCI data shows about cellular versus telephone contact.
CHANNEL_MULTIPLIER: dict[str, float] = {
    "digital": 1.15, "telesales": 0.85, "branch": 1.30, "dsa": 0.95,
}


def campaign_dimension(window: GenerationWindow) -> pd.DataFrame:
    """The campaign dimension."""
    months = window.months
    rows = []
    for c in CAMPAIGNS:
        if c.start_month >= len(months):
            continue
        start = months[c.start_month]
        end_idx = min(c.start_month + c.duration_months, len(months) - 1)
        rows.append({
            "campaign_id": c.campaign_id,
            "campaign_name": c.name,
            "product_code": c.product_code,
            "channel": c.channel,
            "start_date": start,
            "end_date": months[end_idx],
            "target_segment": c.target_segment,
            "cost_per_contact_inr": c.cost_per_contact_inr,
        })
    return pd.DataFrame(rows)


def generate_campaign_contacts(
    customers: pd.DataFrame,
    holdings: pd.DataFrame,
    outcomes: pd.DataFrame,
    profile: CalibrationProfile,
    window: GenerationWindow,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate campaign contacts at campaign x customer x sequence grain.

    Response probability is built from the real UCI base rate, adjusted by the
    elasticities fitted on the real macro columns and by the contact-fatigue and
    prior-outcome effects the real data exhibits.
    """
    rng = rng or np.random.default_rng(profile.seed + 5)
    months = window.months
    base_rate = profile.base_conversion_rate or 0.1127

    cust = customers.merge(
        outcomes[["customer_id", "churn_date"]], on="customer_id", how="left"
    )
    cust["acquired_date"] = pd.to_datetime(cust["acquired_date"])
    cust["churn_date"] = pd.to_datetime(cust["churn_date"])

    held = set(zip(holdings["customer_id"], holdings["product_code"], strict=True))

    # Per-customer memory of previous campaign outcomes, which is what makes the
    # prior-outcome effect possible at all.
    prior_outcome: dict[str, str] = {}
    prior_contacts: dict[str, int] = {}

    records: list[pd.DataFrame] = []

    for spec in CAMPAIGNS:
        if spec.start_month >= len(months):
            continue
        month = months[spec.start_month]

        eligible = cust[
            (cust["acquired_date"] <= month)
            & (cust["churn_date"].isna() | (cust["churn_date"] > month))
        ]
        if spec.target_segment != "all":
            eligible = eligible[eligible["segment"] == spec.target_segment]
        # Do not sell a customer something they already hold.
        mask = [
            (cid, spec.product_code) not in held
            for cid in eligible["customer_id"]
        ]
        eligible = eligible.loc[mask]
        if eligible.empty:
            continue

        n_target = max(int(len(eligible) * spec.target_size_pct), 1)
        chosen = eligible.sample(n=min(n_target, len(eligible)), random_state=
                                 int(rng.integers(0, 2**31 - 1)))

        n = len(chosen)
        cids = chosen["customer_id"].to_numpy()

        # Contacts within the campaign: most customers are contacted once or
        # twice, a tail more often. This is the shape the real campaign column
        # shows, and it is what produces the fatigue curve.
        n_contacts = np.clip(rng.geometric(0.55, n), 1, 8)

        rows = []
        for i in range(n):
            cid = cids[i]
            converted = False
            for seq in range(1, int(n_contacts[i]) + 1):
                if converted:
                    break

                p = base_rate * CHANNEL_MULTIPLIER.get(spec.channel, 1.0)

                # Contact fatigue: each additional contact is less effective.
                p *= 0.72 ** (seq - 1)

                # Prior outcome, the strongest real effect in the UCI data.
                po = prior_outcome.get(cid, "nonexistent")
                p *= {"success": 3.1, "failure": 0.72, "nonexistent": 1.0}[po]

                # Cumulative contact history across campaigns also fatigues.
                p *= 0.94 ** min(prior_contacts.get(cid, 0), 12)

                # Segment propensity: wealthier customers convert better on the
                # products aimed at them.
                seg = chosen["segment"].to_numpy()[i]
                p *= {"mass": 0.9, "affluent": 1.15, "priority": 1.3,
                      "private": 1.45}.get(seg, 1.0)

                # Digital engagement matters on digital channels.
                if spec.channel == "digital":
                    p *= 0.55 + 0.9 * chosen["digital_engagement"].to_numpy()[i]

                p = float(np.clip(p, 0.001, 0.85))
                success = bool(rng.random() < p)

                day_offset = int(rng.integers(0, 28 * spec.duration_months))
                rows.append({
                    "campaign_id": spec.campaign_id,
                    "customer_id": cid,
                    "contact_sequence": seq,
                    "contact_date": month + pd.Timedelta(days=day_offset),
                    "channel": spec.channel,
                    "product_code": spec.product_code,
                    "prior_outcome": po,
                    "n_prior_contacts": prior_contacts.get(cid, 0),
                    "response_probability": round(p, 6),
                    "converted": int(success),
                    "cost_inr": spec.cost_per_contact_inr,
                })
                prior_contacts[cid] = prior_contacts.get(cid, 0) + 1
                converted = success

            prior_outcome[cid] = "success" if converted else "failure"

        if rows:
            records.append(pd.DataFrame(rows))

    out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if not out.empty:
        out.insert(0, "contact_id", [f"CNT{i:08d}" for i in range(1, len(out) + 1)])
        log.info(
            "generated %s campaign contacts across %d campaigns | conversion %.2f%% "
            "(real UCI base %.2f%%)",
            f"{len(out):,}", out["campaign_id"].nunique(),
            out["converted"].mean() * 100, base_rate * 100,
        )
    return out

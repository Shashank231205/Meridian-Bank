"""Generation: calibrated synthetic customer, product and transaction spine."""

from .calibration import (
    CalibrationProfile,
    JointDemographics,
    LogNormalFit,
    MacroElasticity,
    PeerEnvelope,
    assert_calibrated,
    build_profile,
    fit_joint_demographics,
    fit_lognormal,
    fit_macro_elasticity,
)
from .campaigns import CAMPAIGNS, campaign_dimension, generate_campaign_contacts
from .churn import ChurnResult, HazardSpec, generate_churn
from .customers import GenerationWindow, generate_customers
from .generator import (
    DEFAULT_WINDOW,
    GeneratedData,
    compute_portfolio_ratios,
    generate_all,
)
from .products import CATALOGUE, generate_holdings, product_dimension
from .transactions import generate_monthly_snapshots, generate_transactions

__all__ = [
    "CalibrationProfile", "LogNormalFit", "JointDemographics", "MacroElasticity",
    "PeerEnvelope", "build_profile", "assert_calibrated",
    "fit_lognormal", "fit_joint_demographics", "fit_macro_elasticity",
    "GenerationWindow", "generate_customers",
    "CATALOGUE", "product_dimension", "generate_holdings",
    "HazardSpec", "ChurnResult", "generate_churn",
    "generate_transactions", "generate_monthly_snapshots",
    "CAMPAIGNS", "campaign_dimension", "generate_campaign_contacts",
    "GeneratedData", "generate_all", "compute_portfolio_ratios", "DEFAULT_WINDOW",
]

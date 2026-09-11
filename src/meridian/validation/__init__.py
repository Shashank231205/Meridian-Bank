"""Validation: declarative rules, profiling, anomaly detection, HTML reporting."""

from .anomaly import (
    BenfordResult,
    benford_test,
    detect_structural_break,
    duplicate_clusters,
    iqr_outliers,
    mad_outliers,
    round_number_bias,
)
from .engine import ValidationReport, validate, validate_many
from .profiling import ColumnProfile, correlation_pairs, profile_column, profile_table
from .report import render_html, write_report
from .rules import Rule, RuleResult, Severity
from .suites import SUITES, load_provenance, validate_ingested

__all__ = [
    "Rule", "RuleResult", "Severity",
    "ValidationReport", "validate", "validate_many",
    "ColumnProfile", "profile_column", "profile_table", "correlation_pairs",
    "BenfordResult", "benford_test", "mad_outliers", "iqr_outliers",
    "round_number_bias", "duplicate_clusters", "detect_structural_break",
    "render_html", "write_report",
    "SUITES", "validate_ingested", "load_provenance",
]

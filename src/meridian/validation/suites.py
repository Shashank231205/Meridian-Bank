"""Rule suites for the real ingested tables.

Every threshold here is justified rather than guessed, because a bound invented
to make a check pass tests nothing. Where a bound encodes a fact about the
source, the comment says which fact.
"""

from __future__ import annotations

import pandas as pd

from ..common.config import Settings, get_settings
from ..common.io import read_json
from ..common.logging import get_logger
from .anomaly import BenfordResult, benford_test
from .engine import ValidationReport, validate
from .profiling import profile_table
from .rules import (
    Rule,
    Severity,
    in_range,
    in_set,
    non_negative,
    not_null,
    row_count_between,
)

log = get_logger(__name__)


def uci_campaign_suite() -> list[Rule]:
    """Checks for the UCI Bank Marketing table.

    The row count is pinned exactly: this is a fixed, archived academic dataset
    of 41,188 rows. Any other number means the archive changed or the nested-zip
    traversal picked the wrong member, both of which invalidate every campaign
    figure downstream.
    """
    return [
        row_count_between(41_188, 41_188),
        not_null("age"),
        not_null("job"),
        not_null("subscribed"),
        # The dataset documents the contacted population as adults, 17-98.
        in_range("age", 17, 98),
        in_set("subscribed", [0, 1]),
        in_set("y", ["yes", "no"]),
        # campaign = contacts during this campaign, minimum 1 by definition.
        in_range("campaign", 1, 60),
        # pdays uses 999 as the sentinel for "not previously contacted".
        in_range("pdays", 0, 999),
        non_negative("previous"),
        # euribor3m over the 2008-2010 collection window ranged roughly 0.6-5.1.
        in_range("euribor3m", 0.0, 6.0, Severity.WARNING),
        in_set("contact", ["cellular", "telephone"], Severity.WARNING),
        # 'duration' leaks the target and must have been dropped at load.
        Rule(
            name="no_leaky_duration_column",
            description="'duration' is post-hoc and must not reach a model",
            check=lambda df: pd.Series([("duration" in df.columns)] * len(df),
                                       index=df.index),
            severity=Severity.ERROR,
        ),
    ]


def worldbank_suite() -> list[Rule]:
    """Checks for the World Bank macro panel.

    Nulls are expected and allowed: recent years genuinely have no observation
    yet. What must not happen is a null being silently read as a value, so the
    check is that at least one non-null exists per indicator, not that none are
    null.
    """
    return [
        not_null("indicator"),
        not_null("year"),
        in_range("year", 1960, 2030),
        in_range("value", -100, 1e9, Severity.WARNING),
        Rule(
            name="every_indicator_has_at_least_one_observation",
            description="an indicator that is null in every year must be surfaced",
            check=lambda df: df["indicator"].isin(
                df.groupby("indicator")["value"]
                .apply(lambda s: s.notna().sum() == 0)
                .pipe(lambda s: s[s].index)
            ),
            severity=Severity.WARNING,  # India's deposit rate is genuinely empty
            columns=("indicator", "value"),
        ),
    ]


def fdic_suite() -> list[Rule]:
    """Checks for the FDIC peer panel, centred on the unit trap.

    NIMY is a percentage and NIM is dollars in thousands. The bounds below are
    what makes a swap detectable: a real bank's net interest margin percentage
    sits in low single digits, so a NIMY of 46,983,000 fails loudly instead of
    reaching a chart axis.
    """
    return [
        not_null("CERT"),
        not_null("report_date"),
        # Percentages. Generous bounds -- a distressed bank can post a negative
        # ROA -- but nowhere near dollar scale.
        in_range("NIMY", -5, 25),
        in_range("ROA", -15, 15),
        in_range("ROE", -100, 100, Severity.WARNING),
        in_range("EEFFR", 0, 300, Severity.WARNING),
        # Dollar fields, thousands. Total assets must be positive for a going
        # concern; the peer band was selected as USD 1bn-100bn.
        Rule(
            name="assets_positive",
            description="ASSET is dollars (thousands) and must exceed zero",
            check=lambda df: pd.to_numeric(df["ASSET"], errors="coerce") <= 0,
            severity=Severity.ERROR, columns=("ASSET",),
        ),
        Rule(
            name="nim_is_dollar_scale_not_percent",
            description="NIM is dollars; a value under 100 suggests NIMY was loaded",
            check=lambda df: pd.to_numeric(df["NIM"], errors="coerce").abs() < 100,
            severity=Severity.WARNING, columns=("NIM",),
        ),
        Rule(
            name="nimy_is_percent_scale_not_dollars",
            description="NIMY is a percentage; a value over 100 means dollars leaked in",
            check=lambda df: pd.to_numeric(df["NIMY"], errors="coerce").abs() > 100,
            severity=Severity.ERROR, columns=("NIMY",),
        ),
        Rule(
            name="deposits_do_not_exceed_assets",
            description="DEP <= ASSET as an accounting identity",
            check=lambda df: (
                pd.to_numeric(df["DEP"], errors="coerce")
                > pd.to_numeric(df["ASSET"], errors="coerce") * 1.001
            ),
            severity=Severity.ERROR, columns=("DEP", "ASSET"),
        ),
    ]


def fx_suite() -> list[Rule]:
    """Checks for FX reference rates."""
    return [
        not_null("date"), not_null("rate"),
        Rule(
            name="rate_positive",
            description="an exchange rate must be strictly positive",
            check=lambda df: pd.to_numeric(df["rate"], errors="coerce") <= 0,
            severity=Severity.ERROR, columns=("rate",),
        ),
        # USD/INR has traded in the 60-110 band for the last decade; a value
        # outside it means the base and quote were inverted.
        Rule(
            name="usd_inr_plausible",
            description="USD/INR outside 50-150 suggests an inverted pair",
            check=lambda df: (
                (df["quote"] == "INR")
                & ~pd.to_numeric(df["rate"], errors="coerce").between(50, 150)
                & (df.get("base", pd.Series("USD", index=df.index)) == "USD")
            ),
            severity=Severity.WARNING, columns=("quote", "rate"),
        ),
    ]


SUITES = {
    "uci_campaign": uci_campaign_suite,
    "worldbank_macro": worldbank_suite,
    "fdic_peers": fdic_suite,
    "fx_usd_inr_series": fx_suite,
    "fx_latest": fx_suite,
}


def validate_ingested(
    settings: Settings | None = None,
) -> tuple[ValidationReport, pd.DataFrame, dict[str, BenfordResult]]:
    """Validate every table written by the ingestion layer.

    Returns the report, the stacked column profiles, and Benford results for
    the monetary columns that have enough spread to make the test meaningful.
    """
    s = settings or get_settings()
    report = ValidationReport()
    profiles: list[pd.DataFrame] = []
    benford: dict[str, BenfordResult] = {}

    for table, suite_fn in SUITES.items():
        path = s.interim_dir / f"{table}.csv"
        if not path.is_file():
            log.warning("%s not found; run ingestion first", path.name)
            continue

        df = pd.read_csv(path)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

        validate(df, suite_fn(), table=table, report=report)
        profiles.append(profile_table(df, table=table))

        # Benford needs magnitudes spanning several orders of magnitude. The
        # peer panel is deliberately filtered to a USD 1bn-100bn band, leaving
        # only ~2 orders, and a truncated range does not follow Benford -- our
        # own measurement: filtered chi2=45.97 (non-conformity) against
        # chi2=8.65, MAD=0.0076 (acceptable conformity) on the unfiltered
        # institution population spanning 4.8 orders. Testing the filtered panel
        # would report a violation that is an artefact of the filter, so the
        # unfiltered directory is used as the reference population instead.
        if table == "fdic_peers":
            benford.update(_fdic_reference_benford(s))

    report.log_summary()
    profile_df = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    return report, profile_df, benford


def load_provenance(settings: Settings | None = None) -> list[dict[str, object]]:
    """Read the ingestion manifest into rows for the report's provenance table."""
    s = settings or get_settings()
    path = s.raw_dir / "manifest.json"
    if not path.is_file():
        return []
    data = read_json(path)
    return [
        {
            "source": k,
            "status": v.get("status"),
            "rows": v.get("n_rows"),
            "fetched_at": v.get("fetched_at"),
            "sha256": (v.get("sha256") or "")[:12],
        }
        for k, v in data.get("sources", {}).items()
    ]


def _fdic_reference_benford(s: Settings) -> dict[str, BenfordResult]:
    """Benford check against the unfiltered FDIC institution population.

    Serves as a positive control for the anomaly module: real, unmanipulated
    financial magnitudes should conform, and if this ever stops conforming the
    implementation is suspect before the data is.
    """
    from ..ingestion import fdic as _fdic

    try:
        inst = _fdic.fetch_institutions(
            filters="ACTIVE:1", limit=1000,
            cache_dir=s.raw_dir, ttl_days=s.http_cache_ttl_days,
        )
    except Exception as exc:
        log.warning("Benford reference population unavailable: %s", exc)
        return {}

    assets = pd.to_numeric(inst.df.get("ASSET"), errors="coerce").dropna()
    assets = assets[assets > 0]
    if len(assets) < 200:
        return {}
    return {"FDIC total assets, unfiltered institutions (real)": benford_test(assets)}

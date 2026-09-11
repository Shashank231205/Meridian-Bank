"""Declarative data-quality rules.

A rule is a named, severity-tagged predicate over a DataFrame that returns a
:class:`RuleResult` -- pass/fail plus the offending rows. Rules are data, not
code paths: they are declared in a list and executed by the engine, so the set
of checks applied to a table is inspectable and the validation report writes
itself.

Severity drives behaviour rather than just wording:

    ERROR   halts the build. The data is not fit to analyse.
    WARNING recorded and reported; the build continues.
    INFO    observational, for the profile section of the report.

The distinction matters because a validation framework that fails on everything
gets disabled, and one that fails on nothing is decoration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class RuleResult:
    """Outcome of one rule against one table."""

    rule: str
    table: str
    severity: Severity
    passed: bool
    n_checked: int
    n_failed: int
    message: str
    sample: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fail_rate(self) -> float:
        return self.n_failed / self.n_checked if self.n_checked else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule, "table": self.table, "severity": self.severity.value,
            "passed": self.passed, "n_checked": self.n_checked,
            "n_failed": self.n_failed, "fail_rate": round(self.fail_rate, 6),
            "message": self.message, "sample": self.sample[:5],
        }


@dataclass
class Rule:
    """A named check over a DataFrame."""

    name: str
    description: str
    check: Callable[[pd.DataFrame], pd.Series]
    severity: Severity = Severity.ERROR
    columns: tuple[str, ...] = ()

    def applies_to(self, df: pd.DataFrame) -> bool:
        """A rule naming columns the table lacks is skipped, not failed."""
        return all(c in df.columns for c in self.columns)

    def run(self, df: pd.DataFrame, table: str) -> RuleResult:
        """Evaluate. ``check`` returns a boolean Series: True = row is BAD."""
        if df.empty:
            return RuleResult(self.name, table, self.severity, True, 0, 0,
                              "table is empty; nothing to check")
        bad = self.check(df).fillna(False).astype(bool)
        n_failed = int(bad.sum())
        sample = (
            df.loc[bad].head(5).to_dict(orient="records") if n_failed else []
        )
        return RuleResult(
            rule=self.name, table=table, severity=self.severity,
            passed=n_failed == 0, n_checked=len(df), n_failed=n_failed,
            message=(f"{n_failed:,} of {len(df):,} rows failed ({n_failed / len(df):.2%})"
                     if n_failed else "all rows passed"),
            sample=_jsonable(sample),
        )


def _jsonable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce sample rows so the JSON report can serialise them."""
    out = []
    for rec in records:
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                clean[str(k)] = None
            elif hasattr(v, "item"):
                clean[str(k)] = v.item()
            elif isinstance(v, pd.Timestamp):
                clean[str(k)] = v.isoformat()
            else:
                clean[str(k)] = v
        out.append(clean)
    return out


# --- rule constructors -----------------------------------------------------
# Each returns a configured Rule. Written as factories so the same check can be
# applied to different columns and thresholds per table.

def not_null(column: str, severity: Severity = Severity.ERROR) -> Rule:
    return Rule(
        name=f"not_null[{column}]",
        description=f"{column} must not be null",
        check=lambda df: df[column].isna(),
        severity=severity, columns=(column,),
    )


def unique(column: str, severity: Severity = Severity.ERROR) -> Rule:
    return Rule(
        name=f"unique[{column}]",
        description=f"{column} must be unique -- it is a key",
        check=lambda df: df[column].duplicated(keep=False),
        severity=severity, columns=(column,),
    )


def in_range(column: str, lo: float, hi: float,
             severity: Severity = Severity.ERROR) -> Rule:
    def check(df: pd.DataFrame) -> pd.Series:
        s = pd.to_numeric(df[column], errors="coerce")
        return (s < lo) | (s > hi)
    return Rule(
        name=f"in_range[{column}:{lo},{hi}]",
        description=f"{column} must lie in [{lo}, {hi}]",
        check=check, severity=severity, columns=(column,),
    )


def non_negative(column: str, severity: Severity = Severity.ERROR) -> Rule:
    return Rule(
        name=f"non_negative[{column}]",
        description=f"{column} must be >= 0",
        check=lambda df: pd.to_numeric(df[column], errors="coerce") < 0,
        severity=severity, columns=(column,),
    )


def in_set(column: str, allowed: Sequence[Any],
           severity: Severity = Severity.ERROR) -> Rule:
    allowed_set = set(allowed)
    return Rule(
        name=f"in_set[{column}]",
        description=f"{column} must be one of {sorted(map(str, allowed_set))[:6]}...",
        check=lambda df: ~df[column].isin(allowed_set),
        severity=severity, columns=(column,),
    )


def matches(column: str, pattern: str, severity: Severity = Severity.ERROR) -> Rule:
    return Rule(
        name=f"matches[{column}]",
        description=f"{column} must match /{pattern}/",
        check=lambda df: ~df[column].astype(str).str.match(pattern, na=False),
        severity=severity, columns=(column,),
    )


def foreign_key(column: str, parent: pd.DataFrame, parent_key: str,
                severity: Severity = Severity.ERROR) -> Rule:
    """Referential integrity: every child value must exist in the parent."""
    valid = set(parent[parent_key].dropna().unique())
    return Rule(
        name=f"foreign_key[{column}->{parent_key}]",
        description=f"{column} must reference an existing {parent_key}",
        check=lambda df: ~df[column].isin(valid) & df[column].notna(),
        severity=severity, columns=(column,),
    )


def row_count_between(lo: int, hi: int,
                      severity: Severity = Severity.ERROR) -> Rule:
    """Guard against a silently truncated or duplicated load.

    Implemented as an all-rows-fail check so the count appears in the report
    rather than as a bare assertion.
    """
    def check(df: pd.DataFrame) -> pd.Series:
        ok = lo <= len(df) <= hi
        return pd.Series([not ok] * len(df), index=df.index)
    return Rule(
        name=f"row_count_between[{lo},{hi}]",
        description=f"table must have between {lo:,} and {hi:,} rows",
        check=check, severity=severity,
    )


def monotonic_dates(column: str, severity: Severity = Severity.WARNING) -> Rule:
    """Flag out-of-order timestamps in a series expected to advance."""
    def check(df: pd.DataFrame) -> pd.Series:
        s = pd.to_datetime(df[column], errors="coerce")
        return s < s.shift(1)
    return Rule(
        name=f"monotonic_dates[{column}]",
        description=f"{column} should not go backwards",
        check=check, severity=severity, columns=(column,),
    )


def no_future_dates(column: str, severity: Severity = Severity.ERROR) -> Rule:
    def check(df: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(df[column], errors="coerce") > pd.Timestamp.now()
    return Rule(
        name=f"no_future_dates[{column}]",
        description=f"{column} must not be in the future",
        check=check, severity=severity, columns=(column,),
    )

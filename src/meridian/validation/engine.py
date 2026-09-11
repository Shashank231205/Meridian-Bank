"""Validation engine: run rule suites, collect results, gate the build.

The engine executes a suite of :class:`Rule` objects against a table and
aggregates the outcomes into a :class:`ValidationReport`. The report knows
whether the build should stop -- any failing ERROR rule -- which makes data
quality an enforced gate rather than a document nobody reads.

Rules naming columns that a table does not have are skipped and recorded as
such. That keeps one suite usable across the raw, interim and processed shapes
of the same table as columns get added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.exceptions import ValidationError
from ..common.io import write_json
from ..common.logging import get_logger
from .rules import Rule, RuleResult, Severity

log = get_logger(__name__)


@dataclass
class ValidationReport:
    """Aggregated results across every table validated in a run."""

    results: list[RuleResult] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def add(self, result: RuleResult) -> None:
        self.results.append(result)

    def extend(self, results: list[RuleResult]) -> None:
        self.results.extend(results)

    # --- queries -----------------------------------------------------------

    def failures(self, severity: Severity | None = None) -> list[RuleResult]:
        return [
            r for r in self.results
            if not r.passed and (severity is None or r.severity is severity)
        ]

    @property
    def errors(self) -> list[RuleResult]:
        return self.failures(Severity.ERROR)

    @property
    def warnings(self) -> list[RuleResult]:
        return self.failures(Severity.WARNING)

    @property
    def passed(self) -> bool:
        """True when no ERROR-severity rule failed."""
        return not self.errors

    def summary(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "n_rules_run": len(self.results),
            "n_rules_skipped": len(self.skipped),
            "n_passed": sum(1 for r in self.results if r.passed),
            "n_failed": sum(1 for r in self.results if not r.passed),
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "build_passes": self.passed,
            "tables": sorted({r.table for r in self.results}),
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save(self, path: Path) -> Path:
        return write_json(path, {
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
            "skipped": self.skipped,
        })

    def raise_if_failed(self) -> None:
        """Halt the build when an ERROR rule failed."""
        if self.passed:
            return
        lines = [f"  {r.table}.{r.rule}: {r.message}" for r in self.errors]
        raise ValidationError(
            f"{len(self.errors)} ERROR-severity rule(s) failed:\n" + "\n".join(lines)
        )

    def log_summary(self) -> None:
        s = self.summary()
        log.info(
            "validation: %d rules, %d passed, %d failed (%d error, %d warning)",
            s["n_rules_run"], s["n_passed"], s["n_failed"],
            s["n_errors"], s["n_warnings"],
        )
        for r in self.errors:
            log.error("  ERROR  %s.%s -- %s", r.table, r.rule, r.message)
        for r in self.warnings:
            log.warning("  WARN   %s.%s -- %s", r.table, r.rule, r.message)


def validate(
    df: pd.DataFrame, rules: list[Rule], *, table: str,
    report: ValidationReport | None = None,
) -> ValidationReport:
    """Run a rule suite against one table."""
    report = report or ValidationReport()
    for rule in rules:
        if not rule.applies_to(df):
            missing = [c for c in rule.columns if c not in df.columns]
            report.skipped.append({
                "table": table, "rule": rule.name,
                "reason": f"missing column(s): {', '.join(missing)}",
            })
            log.debug("skip %s.%s (missing %s)", table, rule.name, missing)
            continue
        report.add(rule.run(df, table))
    return report


def validate_many(
    tables: dict[str, tuple[pd.DataFrame, list[Rule]]],
) -> ValidationReport:
    """Validate several tables into one report."""
    report = ValidationReport()
    for table, (df, rules) in tables.items():
        log.info("validating %s (%d rows)", table, len(df))
        validate(df, rules, table=table, report=report)
    report.log_summary()
    return report

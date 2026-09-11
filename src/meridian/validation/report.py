"""Render a validation report as standalone HTML.

Self-contained: inline CSS, no external assets, no JavaScript dependency. It
opens from the filesystem on any machine, which is the point -- a report that
needs a server to read does not get read.

The palette matches the dashboard's, and the page respects the reader's colour
scheme rather than forcing light.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.io import atomic_write_text
from ..common.logging import get_logger
from .anomaly import BenfordResult
from .engine import ValidationReport

log = get_logger(__name__)

_CSS = """
:root {
  --bg: #ffffff; --panel: #f7f8fa; --border: #e2e5ea; --fg: #1a1d23;
  --muted: #5c6370; --pass: #1baf7a; --warn: #eda100; --fail: #eb6834;
  --accent: #2a78d6;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14161a; --panel: #1c1f25; --border: #2c313a; --fg: #e6e8ec;
    --muted: #9aa2b1; --pass: #199e70; --warn: #c98500; --fail: #d95926;
    --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  --bg: #14161a; --panel: #1c1f25; --border: #2c313a; --fg: #e6e8ec;
  --muted: #9aa2b1; --pass: #199e70; --warn: #c98500; --fail: #d95926;
  --accent: #3987e5;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  padding-block: 32px; padding-left: 20px; padding-right: 20px;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.02em; }
h2 { font-size: 17px; margin: 34px 0 12px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin: 0 0 26px; font-size: 13px; }
.banner {
  padding: 14px 18px; border-radius: 8px; font-weight: 600; margin-bottom: 26px;
  border: 1px solid;
}
.banner.ok { background: color-mix(in srgb, var(--pass) 12%, transparent);
             border-color: var(--pass); color: var(--pass); }
.banner.bad { background: color-mix(in srgb, var(--fail) 12%, transparent);
              border-color: var(--fail); color: var(--fail); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 12px; margin-bottom: 8px; }
.tile { background: var(--panel); border: 1px solid var(--border);
        border-radius: 8px; padding: 14px 16px; }
.tile .n { font-size: 24px; font-weight: 650; letter-spacing: -0.02em; }
.tile .l { color: var(--muted); font-size: 12px; text-transform: uppercase;
           letter-spacing: 0.05em; margin-top: 2px; }
.scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--border);
         white-space: nowrap; }
th { background: var(--panel); font-weight: 600; font-size: 12px;
     text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
tr:last-child td { border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 2px 9px; border-radius: 99px;
        font-size: 11px; font-weight: 650; letter-spacing: 0.03em; }
.pill.pass { background: color-mix(in srgb, var(--pass) 16%, transparent); color: var(--pass); }
.pill.fail { background: color-mix(in srgb, var(--fail) 16%, transparent); color: var(--fail); }
.pill.warn { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
.note { color: var(--muted); font-size: 12.5px; margin: 8px 0 0; }
footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--border);
         color: var(--muted); font-size: 12px; }
code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
"""


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "&mdash;"
    if isinstance(v, float):
        return f"{v:,.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    return _esc(v)


def _table(df: pd.DataFrame, *, numeric_cols: set[str] | None = None,
           max_rows: int = 300) -> str:
    if df.empty:
        return '<p class="note">Nothing to report.</p>'
    numeric_cols = numeric_cols or {
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    }
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="num">{_fmt(r[c])}</td>' if c in numeric_cols
            else f"<td>{_fmt(r[c])}</td>"
            for c in df.columns
        ) + "</tr>"
        for _, r in df.head(max_rows).iterrows()
    )
    more = (f'<p class="note">Showing {max_rows:,} of {len(df):,} rows.</p>'
            if len(df) > max_rows else "")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead>' \
           f"<tbody>{body}</tbody></table></div>{more}"


def _results_table(report: ValidationReport) -> str:
    rows = []
    for r in sorted(report.results,
                    key=lambda x: (x.passed, x.severity.value, x.table)):
        pill = "pass" if r.passed else ("fail" if r.severity.value == "ERROR" else "warn")
        status = "PASS" if r.passed else r.severity.value
        rows.append(
            f"<tr><td>{_esc(r.table)}</td><td><code>{_esc(r.rule)}</code></td>"
            f'<td><span class="pill {pill}">{status}</span></td>'
            f'<td class="num">{r.n_checked:,}</td>'
            f'<td class="num">{r.n_failed:,}</td>'
            f'<td class="num">{r.fail_rate:.2%}</td>'
            f"<td>{_esc(r.message)}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>Table</th><th>Rule</th><th>Status</th><th>Checked</th>"
        "<th>Failed</th><th>Rate</th><th>Detail</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _benford_section(results: dict[str, BenfordResult]) -> str:
    if not results:
        return ""
    blocks = []
    for label, b in results.items():
        if b.n == 0:
            continue
        rows = "".join(
            f"<tr><td>{d}</td>"
            f'<td class="num">{b.observed.get(d, 0):.2%}</td>'
            f'<td class="num">{b.expected[d]:.2%}</td>'
            f'<td class="num">{(b.observed.get(d, 0) - b.expected[d]):+.2%}</td></tr>'
            for d in range(1, 10)
        )
        pill = "pass" if b.conforms else "warn"
        blocks.append(
            f"<h2>Benford's law &mdash; {_esc(label)}</h2>"
            f'<p class="note">n = {b.n:,} &middot; &chi;&sup2; = {b.chi2:.2f} '
            f"(critical {b.critical_value} at &alpha;={b.alpha}) &middot; "
            f"MAD = {b.mad:.4f} &middot; "
            f'<span class="pill {pill}">{_esc(b.interpretation)}</span></p>'
            '<div class="scroll"><table><thead><tr><th>Digit</th><th>Observed</th>'
            "<th>Expected</th><th>Deviation</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    return "".join(blocks)


def render_html(
    report: ValidationReport,
    *,
    profiles: pd.DataFrame | None = None,
    benford: dict[str, BenfordResult] | None = None,
    title: str = "Meridian Bank &mdash; Data Validation Report",
    provenance: list[dict[str, Any]] | None = None,
) -> str:
    """Build the complete HTML document."""
    s = report.summary()
    ok = s["build_passes"]
    banner = (
        '<div class="banner ok">BUILD PASSES &mdash; no ERROR-severity rule failed</div>'
        if ok else
        f'<div class="banner bad">BUILD BLOCKED &mdash; {s["n_errors"]} '
        "ERROR-severity rule(s) failed</div>"
    )

    tiles = "".join(
        f'<div class="tile"><div class="n">{v}</div><div class="l">{lab}</div></div>'
        for lab, v in [
            ("Rules run", f"{s['n_rules_run']:,}"),
            ("Passed", f"{s['n_passed']:,}"),
            ("Errors", f"{s['n_errors']:,}"),
            ("Warnings", f"{s['n_warnings']:,}"),
            ("Tables", f"{len(s['tables']):,}"),
            ("Skipped", f"{s['n_rules_skipped']:,}"),
        ]
    )

    parts = [
        f"<h1>{title}</h1>",
        f'<p class="sub">Generated {_esc(s["generated_at"])} &middot; '
        f'tables: {_esc(", ".join(s["tables"]))}</p>',
        banner,
        f'<div class="tiles">{tiles}</div>',
        "<h2>Rule results</h2>",
        _results_table(report),
    ]

    if provenance:
        parts += ["<h2>Source provenance</h2>",
                  '<p class="note">Every figure in this report traces to one of these '
                  "fetches.</p>",
                  _table(pd.DataFrame(provenance))]

    if profiles is not None and not profiles.empty:
        cols = [c for c in ["table", "name", "dtype", "n_rows", "n_missing",
                            "pct_missing", "n_distinct", "mean", "median",
                            "minimum", "maximum", "n_outliers_iqr"]
                if c in profiles.columns]
        parts += ["<h2>Column profiles</h2>", _table(profiles[cols])]

    if benford:
        parts.append(_benford_section(benford))

    if report.skipped:
        parts += ["<h2>Skipped rules</h2>",
                  '<p class="note">Rules naming columns the table does not have are '
                  "skipped rather than failed.</p>",
                  _table(pd.DataFrame(report.skipped))]

    parts.append(
        "<footer>Meridian Bank customer intelligence platform &middot; "
        f"validation framework &middot; {datetime.now():%Y-%m-%d %H:%M}</footer>"
    )

    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head>"
        f'<body><div class="wrap">{"".join(parts)}</div></body></html>'
    )


def write_report(
    report: ValidationReport, path: Path, **kwargs: Any
) -> Path:
    """Render and write the HTML report."""
    out = atomic_write_text(path, render_html(report, **kwargs))
    log.info("validation report written: %s", out)
    return out

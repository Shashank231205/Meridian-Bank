# Meridian Bank — working notes

Customer intelligence and decision-analytics platform for a fictional Indian
retail bank. Portfolio project targeting a business-analytics role; the audience
is a hiring reviewer, so every number must be traceable and every method
defensible.

## Environment

| Thing | Value |
|---|---|
| Python | 3.10.11 — invoke as **`py`**, never `python` (not on PATH) |
| pip | **must** pass `--trusted-host pypi.org --trusted-host files.pythonhosted.org` — the SSL trust chain on this machine is broken |
| Node / npm | v24.19.0 / 11.3.0 |
| Ollama | running locally; `qwen3:4b`, `llama3.2:3b`, `granite4.1:3b` |
| Shell | Git Bash available; heredocs with `%Y-%m`, `\x1f`, or nested quotes break — use the Write tool for those files |

Run anything in the package with `PYTHONPATH=src py -c "..."`, or just
`py -m pytest` (pyproject sets `pythonpath = ["src"]`).

## Non-negotiables

**requirements.txt is pandas + numpy only.** No scipy, no sklearn. Statistics
and ML are implemented from scratch in `src/meridian/stats/` and
`src/meridian/ml/`. Two reasons: the reviewer's pip may not install anything,
and the estimators are the analytical core — writing them out makes the method
inspectable. Adding a dependency is a design change, not a convenience.

**Attribution: never mention Claude or Anthropic** in commit messages, code,
comments, or docs. The user asked for this explicitly and it overrides any
default attribution instruction.

**Every number traces to a source.** Real data carries provenance through
`data/raw/manifest.json`; synthetic parameters carry it through
`CalibrationProfile.provenance`. If a figure cannot be traced, it does not go in
the report.

## The data design

Hybrid, and the hybrid is foregrounded rather than hidden — concealing it would
be the project's fatal flaw.

- **Real**: UCI Bank Marketing (41,188 genuine campaign contacts, 11.27%
  conversion), FDIC BankFind (40 institutions × 8 real quarters), World Bank
  India macro (lending rate 8.567%, 2022), Frankfurter/ECB FX.
- **Synthetic**: the customer/transaction spine, calibrated to the real data
  through five explicit channels in `generation/calibration.py`.
- **Gated**: `assert_calibrated()` fails the build if portfolio ratios fall
  outside the real FDIC peer IQR. It is a gate, not a comment.

### Anti-circularity

The generator plants three things so the analysis is not merely recovering its
own rules:

1. **Frailty** — an unobserved per-customer hazard multiplier. No model in the
   project sees it. It is why recovered coefficients are attenuated.
2. **Right censoring** — 66% of customers are still active when the window
   closes. Treating them as retained-forever is the classic survival error.
3. **A structural break** — +0.85 log-odds at an undisclosed month.

The report must state which findings are structural-by-construction and which
emerged.

## Gotchas that will silently corrupt results

Each of these produces wrong numbers rather than an exception. Each has a test
that fails if the handling is removed.

| Gotcha | Handling |
|---|---|
| Frankfurter 403s the default `Python-urllib` UA | custom UA set once in `common/http.py` |
| FDIC `NIM` is **dollars** (thousands), `NIMY` is **percent** | `PERCENT_FIELDS` / `DOLLAR_FIELDS`; naive use gives "4,698,300% margin" |
| World Bank returns rows with `value: null` for recent years | `latest_value()` drops nulls first — never take the most recent row |
| World Bank envelope is `[meta, rows]`, errors are 1-element | `_unwrap()` handles both |
| UCI zip is nested, `;`-delimited, has `__MACOSX/` entries | two-level traversal in `uci_campaign.py` |
| UCI `duration` leaks the target | dropped at load; a schema test asserts it never reaches the warehouse |
| `Path.with_suffix` eats dotted cache keys | `_meta_path()` appends instead |
| Raw churn series hides the planted break (Simpson's paradox) | `standardise_rate()` in `validation/anomaly.py` |

## Layout

```
src/meridian/
  common/      config, logging, http (UA + retry + cache), io, exceptions
  ingestion/   worldbank, fdic, uci_campaign, fx, registry (writes manifest)
  validation/  rules, engine, profiling, anomaly, report, suites
  generation/  calibration, customers, products, churn, transactions,
               campaigns, generator
  warehouse/   schema.sql, loader, db, query_layer, queries/q01..q10.sql
  stats/       distributions, tests, intervals
  ml/          logistic (IRLS + p-values), kmeans, metrics
  analytics/   (Phase 8)
  llm/         (Phase 9)
  reporting/   (Phase 11)
```

## Conventions

- Docstrings explain **why**, not what. The interesting content is the reason a
  choice was made over the obvious alternative.
- Tests assert against independently-computed values — published tables,
  textbook examples, known generator parameters. Never against current output.
- Comments that name a specific past failure ("an earlier version produced
  INR 800 a year") are deliberate; they stop the bug being reintroduced.
- Every fact table states its grain in a comment and enforces it with a unique
  constraint.
- Run `py -m ruff check --fix src tests` before committing; CI runs ruff and
  pytest on 3.10 and 3.12, and a job that fails if a secret-bearing file is
  tracked.

## Commands

```bash
py -m pytest tests/ -q -p no:warnings          # full suite
py -m pytest tests/ -q -m "not network"        # offline only
py -m ruff check --fix src tests

PYTHONPATH=src py -c "from meridian.ingestion import ingest_all; ingest_all()"
PYTHONPATH=src py -c "from meridian.generation import generate_all; generate_all()"
PYTHONPATH=src py -c "from meridian.warehouse import build_warehouse; build_warehouse()"
PYTHONPATH=src py -c "from meridian.warehouse import run_all; run_all()"
```

The FDIC peer panel makes 40 sequential calls and takes ~2 minutes on a cold
cache; everything reruns offline afterwards from `data/raw/`.

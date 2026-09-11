# Build progress

Plan of record: `~/.claude/plans/zany-napping-anchor.md` (14 phases, ~108h).
Each phase ends in a committed, runnable state.

## Done

| # | Phase | Exit criterion | Status |
|---|---|---|---|
| 0 | Repo scaffold, `common/http.py` | Frankfurter returns 200 with custom UA; first push | done — default UA gave 403, custom gave 200, verified live |
| 1 | Four ingestion modules + registry | `data/raw` populated, WB nulls and nested zip handled | done — 42,229 real rows, 5/5 sources |
| 2 | Validation framework | `validation_report.html` on real data | done — 36 rules, 0 errors, 1 correct warning |
| 3 | `calibration.py` → `CalibrationProfile` | Prints real values | done — lending 8.5671% (2022), 988 demographic combinations from 41,188 rows |
| 4 | Generator | 25k customers, ~1.8M rows, `assert_calibrated` passes | done — 3.16M rows, all 5 ratios inside real FDIC peer IQR |
| 5 | Warehouse schema, loader, db | `meridian.db` built, FKs on | done — 2.64M rows, FK check clean, integrity ok |
| 6 | 10 SQL queries + grain tests | `q10` yields all 8 KPIs | done — all 10 run, q10 returns exactly 8 |
| 7 | `stats/` + `ml/` + textbook tests | Logistic recovers generator coefficients | done — recovery within 0.06 on known truth; signs recovered on the real panel |

**Tests: 344 passing. Ruff clean. Everything pushed to `origin/main`.**

## Remaining

| # | Phase | Scope | Exit criterion |
|---|---|---|---|
| 8 | `analytics/` | eda, customer, revenue, retention, campaign, clv, benchmark | all `outputs/tables/*.csv` produced |
| 9 | `llm/` | Ollama/Groq client, prompts, insights | `insights.json` from aggregates only, never customer rows |
| 10 | `frontend/` | Next.js 16 + React 19 + Recharts | 8 KPIs live, dark mode, responsive |
| 11 | `reporting/` | business report + RISK_ASSESSMENT | no placeholders, every number traceable |
| 12 | BRD, POLICY_REGISTER, deck | the three JD-gap artifacts | three gaps closed |
| 13 | README, docs, notebooks, deploy | polish | clean run from an empty `data/` |

## Current numbers

Regenerate with the commands in CLAUDE.md; these are from the last full run.

**Real data**
- UCI: 41,188 contacts, 11.27% conversion, `duration` dropped
- World Bank India: lending 8.567% (2022, latest non-null), branch density
  14.62/100k, account ownership 89.02%
- FDIC peers: 40 institutions, ROA IQR [1.142, 1.702], NIMY [3.744, 4.226],
  equity/assets median 10.34%
- Note: India's deposit-rate series is empty in WDI — surfaced by validation,
  not fabricated; the spread comes from the real peer NIM instead

**Generated**
- 25,000 customers, 36,928 holdings, 1.94M transactions, 565k snapshots
- Churn 33.8% over 5.5 years (~18% annualised), 66.2% censored
- Structural break planted at 2024-05-01, +0.85 log-odds
- Portfolio: ROA 1.483, ROE 14.337, NIMY 4.224, EEFFR 56.0, LTD 89.49 — all
  inside the real peer IQR

**Analysis**
- Home loans 36.15% of revenue; four products cover 75.6%
- Top CLV decile holds 87.6% of value among active customers
- Logistic on the churn panel recovers every coefficient sign;
  `complaint_active` 0.6785 against a true 0.680

## Open threads

- The planted break is invisible in the raw churn series and recovered exactly
  by `standardise_rate()`. This is the project's best analytical story —
  Simpson's paradox on its own data — and should lead the retention section of
  the report.
- `q09` CLV concentration must be quoted active-only (87.6%), not blended
  (89.8%); the blended figure is bimodal because churned customers have zero
  future value by construction.
- Phase 9 sends aggregate KPIs only to the LLM, never customer rows. Free tiers
  may train on prompts; this is a documented data-governance decision for
  `RISK_ASSESSMENT.md`.

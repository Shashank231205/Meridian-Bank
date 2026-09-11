-- q05: Cohort retention triangle
--
-- Grain: one row per acquisition cohort per months-since-acquisition.
-- The classic triangle, built by conditional aggregation rather than a pivot,
-- because SQLite has no PIVOT and because conditional aggregation is portable.
--
-- Retention is measured against the cohort's own starting size, not against the
-- previous period, so the numbers read as survival curves rather than
-- period-over-period ratios.
--
-- Cells beyond the observation window are NULL, not zero. A cohort acquired
-- last month has no month-12 retention to report, and writing zero there would
-- drag every long-horizon average towards the floor.
WITH cohorts AS (
    SELECT
        c.customer_key,
        strftime('%Y-%m', c.acquired_date) AS cohort_month,
        c.acquired_date,
        s.churned,
        s.churn_date,
        s.observed_months
    FROM dim_customer c
    JOIN fact_customer_snapshot s ON s.customer_key = c.customer_key
),
cohort_size AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM cohorts GROUP BY cohort_month
),
survival AS (
    SELECT
        co.cohort_month,
        cs.cohort_size,
        co.customer_key,
        -- Months this customer survived: to churn, or to the end of the window.
        CASE WHEN co.churned = 1
             THEN CAST((julianday(co.churn_date) - julianday(co.acquired_date)) / 30.44
                       AS INTEGER)
             ELSE co.observed_months
        END AS months_survived,
        co.churned,
        co.observed_months AS months_observed
    FROM cohorts co
    JOIN cohort_size cs ON cs.cohort_month = co.cohort_month
)
SELECT
    cohort_month,
    cohort_size,
    -- Retained at month N = survived at least N months.
    ROUND(100.0 * SUM(CASE WHEN months_survived >= 1 THEN 1 ELSE 0 END)
          / cohort_size, 2) AS m01_pct,
    ROUND(100.0 * SUM(CASE WHEN months_survived >= 3 THEN 1 ELSE 0 END)
          / cohort_size, 2) AS m03_pct,
    ROUND(100.0 * SUM(CASE WHEN months_survived >= 6 THEN 1 ELSE 0 END)
          / cohort_size, 2) AS m06_pct,
    -- NULL once the cohort is too young to have reached the horizon.
    CASE WHEN MAX(months_observed) >= 12
         THEN ROUND(100.0 * SUM(CASE WHEN months_survived >= 12 THEN 1 ELSE 0 END)
                    / cohort_size, 2) END AS m12_pct,
    CASE WHEN MAX(months_observed) >= 18
         THEN ROUND(100.0 * SUM(CASE WHEN months_survived >= 18 THEN 1 ELSE 0 END)
                    / cohort_size, 2) END AS m18_pct,
    CASE WHEN MAX(months_observed) >= 24
         THEN ROUND(100.0 * SUM(CASE WHEN months_survived >= 24 THEN 1 ELSE 0 END)
                    / cohort_size, 2) END AS m24_pct,
    CASE WHEN MAX(months_observed) >= 36
         THEN ROUND(100.0 * SUM(CASE WHEN months_survived >= 36 THEN 1 ELSE 0 END)
                    / cohort_size, 2) END AS m36_pct,
    MAX(months_observed) AS max_months_observed,
    SUM(churned)         AS n_churned
FROM survival
GROUP BY cohort_month, cohort_size
ORDER BY cohort_month;

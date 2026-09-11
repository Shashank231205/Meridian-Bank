-- q03: Monthly revenue trend with MoM, YoY and a moving average
--
-- Grain: one row per month.
-- LAG(1) gives month on month; LAG(12) gives year on year, which is the
-- comparison that matters in a seasonal business -- October beating September
-- says nothing when October is always the festive peak.
WITH monthly AS (
    SELECT
        f.month,
        COUNT(DISTINCT f.customer_key)   AS n_customers,
        SUM(f.balance_inr)               AS balance_inr,
        SUM(f.interest_revenue_inr)      AS interest_revenue_inr,
        SUM(f.fee_revenue_inr)           AS fee_revenue_inr,
        SUM(f.total_revenue_inr)         AS total_revenue_inr
    FROM fact_monthly_account f
    GROUP BY f.month
)
SELECT
    m.month,
    m.n_customers,
    ROUND(m.balance_inr, 2)          AS balance_inr,
    ROUND(m.interest_revenue_inr, 2) AS interest_revenue_inr,
    ROUND(m.fee_revenue_inr, 2)      AS fee_revenue_inr,
    ROUND(m.total_revenue_inr, 2)    AS total_revenue_inr,
    ROUND(m.total_revenue_inr / NULLIF(m.n_customers, 0), 2) AS revenue_per_customer_inr,

    ROUND(LAG(m.total_revenue_inr, 1) OVER (ORDER BY m.month), 2) AS prev_month_revenue,
    ROUND(100.0 * (m.total_revenue_inr - LAG(m.total_revenue_inr, 1) OVER (ORDER BY m.month))
          / NULLIF(LAG(m.total_revenue_inr, 1) OVER (ORDER BY m.month), 0), 2) AS mom_pct,

    ROUND(LAG(m.total_revenue_inr, 12) OVER (ORDER BY m.month), 2) AS prev_year_revenue,
    ROUND(100.0 * (m.total_revenue_inr - LAG(m.total_revenue_inr, 12) OVER (ORDER BY m.month))
          / NULLIF(LAG(m.total_revenue_inr, 12) OVER (ORDER BY m.month), 0), 2) AS yoy_pct,

    -- Trailing 3-month mean, to damp the festive spike when reading trend.
    ROUND(AVG(m.total_revenue_inr) OVER (
        ORDER BY m.month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS revenue_ma3,
    ROUND(AVG(m.total_revenue_inr) OVER (
        ORDER BY m.month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    ), 2) AS revenue_ma12,

    ROUND(SUM(m.total_revenue_inr) OVER (ORDER BY m.month), 2) AS cumulative_revenue_inr,
    d.fiscal_year,
    d.fiscal_quarter,
    d.is_festive_season
FROM monthly m
LEFT JOIN dim_date d ON d.full_date = m.month
ORDER BY m.month;

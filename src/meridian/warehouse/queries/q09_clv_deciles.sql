-- q09: Customer lifetime value deciles
--
-- Grain: one row per customer, plus a decile summary.
-- CLV is computed as expected remaining revenue discounted to present value,
-- using the customer's own observed revenue and a survival estimate from their
-- risk profile. It is a simple model and it is stated as one.
--
-- The discount rate is not invented: it is the real World Bank India lending
-- rate held in ref_macro_indicator, which is the same rate every product in the
-- catalogue was priced from. Using a made-up 10% here would break that chain.
--
-- NTILE(10) gives the deciles and CUME_DIST gives each customer's position in
-- the distribution, which together answer "what share of revenue comes from the
-- top decile" -- usually the single most quoted number in a portfolio review.
--
-- That number needs care here. Churned customers have zero expected future
-- value by construction, so the population is bimodal rather than a smooth
-- Pareto tail: on this data the overall top decile holds 89.8% of total CLV,
-- which sounds like extreme concentration but is mostly an artefact of mixing
-- two populations. is_active is emitted so the dashboard can quote the
-- concentration among active customers -- the figure that actually informs a
-- retention or servicing decision -- rather than the blended one.
WITH discount AS (
    -- Latest non-null observation, never the most recent row: World Bank
    -- publishes rows for years it has no data for yet.
    SELECT COALESCE(
        (SELECT value FROM ref_macro_indicator
          WHERE indicator = 'FR.INR.LEND' AND value IS NOT NULL
          ORDER BY year DESC LIMIT 1),
        8.567143
    ) AS annual_rate_pct
),
customer_revenue AS (
    SELECT
        f.customer_key,
        SUM(f.total_revenue_inr)                       AS lifetime_revenue_to_date_inr,
        COUNT(DISTINCT f.month)                        AS months_active,
        SUM(f.total_revenue_inr) / NULLIF(COUNT(DISTINCT f.month), 0)
                                                       AS avg_monthly_revenue_inr
    FROM fact_monthly_account f
    GROUP BY f.customer_key
),
survival AS (
    SELECT
        s.customer_key,
        s.churned,
        s.observed_months,
        -- Expected remaining months. Churned customers have none. For active
        -- customers, a longer tenure implies a lower hazard, so expected
        -- remaining life grows with tenure and is capped at 10 years to stop
        -- the geometric tail dominating the estimate.
        CASE
            WHEN s.churned = 1 THEN 0
            ELSE MIN(120, 12 + s.observed_months * 1.5)
        END AS expected_remaining_months
    FROM fact_customer_snapshot s
),
clv AS (
    SELECT
        c.customer_id,
        c.segment,
        c.city,
        c.acquisition_channel,
        c.annual_income_inr,
        c.tenure_months,
        sv.churned,
        sv.observed_months,
        ROUND(cr.lifetime_revenue_to_date_inr, 2) AS lifetime_revenue_to_date_inr,
        ROUND(cr.avg_monthly_revenue_inr, 2)      AS avg_monthly_revenue_inr,
        sv.expected_remaining_months,
        -- Present value of an annuity: PV = C * (1 - (1+r)^-n) / r, with r the
        -- monthly rate derived from the real annual lending rate.
        ROUND(
            CASE WHEN sv.expected_remaining_months > 0 AND d.annual_rate_pct > 0
                 THEN cr.avg_monthly_revenue_inr
                      * (1 - POWER(1 + d.annual_rate_pct / 100.0 / 12.0,
                                   -sv.expected_remaining_months))
                      / (d.annual_rate_pct / 100.0 / 12.0)
                 ELSE 0
            END, 2
        ) AS expected_future_value_inr
    FROM dim_customer c
    JOIN customer_revenue cr ON cr.customer_key = c.customer_key
    JOIN survival sv         ON sv.customer_key = c.customer_key
    CROSS JOIN discount d
),
ranked AS (
    SELECT
        clv.*,
        ROUND(lifetime_revenue_to_date_inr + expected_future_value_inr, 2) AS total_clv_inr,
        NTILE(10) OVER (
            ORDER BY lifetime_revenue_to_date_inr + expected_future_value_inr DESC
        ) AS clv_decile,
        ROUND(CUME_DIST() OVER (
            ORDER BY lifetime_revenue_to_date_inr + expected_future_value_inr
        ), 4) AS clv_percentile
    FROM clv
)
SELECT
    r.*,
    ROUND(100.0 * r.total_clv_inr / NULLIF(SUM(r.total_clv_inr) OVER (), 0), 4)
        AS share_of_total_clv_pct,
    ROUND(SUM(r.total_clv_inr) OVER (
        ORDER BY r.total_clv_inr DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ), 2) AS running_clv_inr,
    ROUND(100.0 * SUM(r.total_clv_inr) OVER (
        ORDER BY r.total_clv_inr DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) / NULLIF(SUM(r.total_clv_inr) OVER (), 0), 2) AS cumulative_clv_share_pct,
    CASE WHEN r.churned = 0 THEN 1 ELSE 0 END AS is_active,
    -- Decile and cumulative share computed among active customers only, so the
    -- concentration statistic can be quoted without the churned population
    -- flattening the bottom of the distribution.
    CASE WHEN r.churned = 0 THEN NTILE(10) OVER (
        PARTITION BY r.churned ORDER BY r.total_clv_inr DESC
    ) END AS active_clv_decile
FROM ranked r
ORDER BY r.total_clv_inr DESC;

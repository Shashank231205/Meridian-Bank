-- q04: Product revenue Pareto
--
-- Grain: one row per product.
-- Running total and cumulative share, to locate the 80/20 point. The window
-- ordering must match the display ordering or the running total is nonsense.
WITH product_revenue AS (
    SELECT
        p.product_code,
        p.product_name,
        p.category,
        p.interest_rate_pct,
        COUNT(DISTINCT f.customer_key) AS n_customers,
        SUM(f.balance_inr)             AS balance_inr,
        SUM(f.total_revenue_inr)       AS revenue_inr
    FROM fact_monthly_account f
    JOIN dim_product p ON p.product_key = f.product_key
    GROUP BY p.product_code, p.product_name, p.category, p.interest_rate_pct
),
ranked AS (
    SELECT
        pr.*,
        SUM(pr.revenue_inr) OVER ()                                AS total_revenue,
        SUM(pr.revenue_inr) OVER (ORDER BY pr.revenue_inr DESC
                                  ROWS BETWEEN UNBOUNDED PRECEDING
                                  AND CURRENT ROW)                 AS running_revenue,
        ROW_NUMBER() OVER (ORDER BY pr.revenue_inr DESC)           AS revenue_rank
    FROM product_revenue pr
)
SELECT
    revenue_rank,
    product_code,
    product_name,
    category,
    interest_rate_pct,
    n_customers,
    ROUND(balance_inr, 2)  AS balance_inr,
    ROUND(revenue_inr, 2)  AS revenue_inr,
    ROUND(100.0 * revenue_inr / NULLIF(total_revenue, 0), 2)     AS revenue_share_pct,
    ROUND(running_revenue, 2)                                     AS running_revenue_inr,
    ROUND(100.0 * running_revenue / NULLIF(total_revenue, 0), 2)  AS cumulative_share_pct,
    CASE WHEN 100.0 * running_revenue / NULLIF(total_revenue, 0) <= 80.0
         THEN 1 ELSE 0 END AS in_top_80_pct
FROM ranked
ORDER BY revenue_rank;

-- q10: Executive KPI rollup
--
-- Grain: one row per KPI.
-- The eight headline numbers the dashboard shows, each with its current value,
-- its prior-period comparison and the direction that counts as good. Returned
-- long rather than wide so the dashboard can render tiles by iterating rows
-- instead of hardcoding column names, and so a new KPI is a new row here rather
-- than a schema change and a frontend change.
--
-- "good_direction" is part of the data because the sign convention is not
-- universal: revenue rising is good, churn rising is not, and a tile that
-- colours both green for "up" is actively misleading.
WITH bounds AS (
    SELECT
        MAX(month) AS current_month,
        (SELECT MAX(month) FROM fact_monthly_account
          WHERE month < (SELECT MAX(month) FROM fact_monthly_account)) AS prior_month,
        (SELECT MIN(month) FROM fact_monthly_account)                  AS first_month
    FROM fact_monthly_account
),
revenue AS (
    SELECT
        SUM(CASE WHEN f.month = b.current_month THEN f.total_revenue_inr END) AS current_revenue,
        SUM(CASE WHEN f.month = b.prior_month   THEN f.total_revenue_inr END) AS prior_revenue,
        SUM(CASE WHEN f.month = b.current_month THEN f.balance_inr END)       AS current_balance,
        SUM(CASE WHEN f.month = b.prior_month   THEN f.balance_inr END)       AS prior_balance,
        COUNT(DISTINCT CASE WHEN f.month = b.current_month THEN f.customer_key END)
            AS current_customers,
        COUNT(DISTINCT CASE WHEN f.month = b.prior_month THEN f.customer_key END)
            AS prior_customers
    FROM fact_monthly_account f CROSS JOIN bounds b
),
products AS (
    SELECT
        1.0 * COUNT(*) / COUNT(DISTINCT customer_key) AS products_per_customer
    FROM fact_monthly_account
    WHERE month = (SELECT current_month FROM bounds)
),
products_prior AS (
    SELECT
        1.0 * COUNT(*) / COUNT(DISTINCT customer_key) AS products_per_customer
    FROM fact_monthly_account
    WHERE month = (SELECT prior_month FROM bounds)
),
churn AS (
    SELECT
        100.0 * SUM(churned) / COUNT(*)   AS churn_rate_pct,
        100.0 * SUM(censored) / COUNT(*)  AS retention_rate_pct,
        COUNT(*)                          AS n_customers
    FROM fact_customer_snapshot
),
txn AS (
    SELECT
        AVG(CASE WHEN month = (SELECT current_month FROM bounds) THEN amount_inr END)
            AS current_avg_txn,
        AVG(CASE WHEN month = (SELECT prior_month FROM bounds) THEN amount_inr END)
            AS prior_avg_txn,
        COUNT(CASE WHEN month = (SELECT current_month FROM bounds) THEN 1 END)
            AS current_txn_count
    FROM fact_transaction
),
campaign AS (
    SELECT
        100.0 * SUM(converted) / NULLIF(COUNT(*), 0)          AS conversion_rate_pct,
        SUM(cost_inr) / NULLIF(SUM(converted), 0)             AS cost_per_acquisition_inr
    FROM fact_campaign_contact
),
clv AS (
    SELECT AVG(lifetime_revenue) AS avg_lifetime_revenue_inr
    FROM (
        SELECT customer_key, SUM(total_revenue_inr) AS lifetime_revenue
        FROM fact_monthly_account GROUP BY customer_key
    )
)
-- KPI 1: monthly revenue
SELECT
    1 AS kpi_order,
    'monthly_revenue_inr'      AS kpi_key,
    'Monthly Revenue'          AS kpi_label,
    'currency'                 AS kpi_format,
    ROUND(r.current_revenue, 2)                                      AS current_value,
    ROUND(r.prior_revenue, 2)                                        AS prior_value,
    ROUND(100.0 * (r.current_revenue - r.prior_revenue)
          / NULLIF(r.prior_revenue, 0), 2)                           AS change_pct,
    'up'                       AS good_direction
FROM revenue r

UNION ALL
-- KPI 2: active customers
SELECT 2, 'active_customers', 'Active Customers', 'integer',
    r.current_customers, r.prior_customers,
    ROUND(100.0 * (r.current_customers - r.prior_customers)
          / NULLIF(r.prior_customers, 0), 2),
    'up'
FROM revenue r

UNION ALL
-- KPI 3: total balances under management
SELECT 3, 'total_balance_inr', 'Balances Under Management', 'currency',
    ROUND(r.current_balance, 2), ROUND(r.prior_balance, 2),
    ROUND(100.0 * (r.current_balance - r.prior_balance)
          / NULLIF(r.prior_balance, 0), 2),
    'up'
FROM revenue r

UNION ALL
-- KPI 4: revenue per customer
SELECT 4, 'revenue_per_customer_inr', 'Revenue per Customer', 'currency',
    ROUND(r.current_revenue / NULLIF(r.current_customers, 0), 2),
    ROUND(r.prior_revenue / NULLIF(r.prior_customers, 0), 2),
    ROUND(100.0 * ((r.current_revenue / NULLIF(r.current_customers, 0))
                 - (r.prior_revenue / NULLIF(r.prior_customers, 0)))
          / NULLIF(r.prior_revenue / NULLIF(r.prior_customers, 0), 0), 2),
    'up'
FROM revenue r

UNION ALL
-- KPI 5: churn rate. Rising is bad, hence good_direction = 'down'.
SELECT 5, 'churn_rate_pct', 'Churn Rate', 'percent',
    ROUND(c.churn_rate_pct, 2), NULL, NULL, 'down'
FROM churn c

UNION ALL
-- KPI 6: products per customer -- the cross-sell measure
SELECT 6, 'products_per_customer', 'Products per Customer', 'decimal',
    ROUND(p.products_per_customer, 3),
    ROUND(pp.products_per_customer, 3),
    ROUND(100.0 * (p.products_per_customer - pp.products_per_customer)
          / NULLIF(pp.products_per_customer, 0), 2),
    'up'
FROM products p CROSS JOIN products_prior pp

UNION ALL
-- KPI 7: campaign conversion rate
SELECT 7, 'campaign_conversion_pct', 'Campaign Conversion', 'percent',
    ROUND(cm.conversion_rate_pct, 2), NULL, NULL, 'up'
FROM campaign cm

UNION ALL
-- KPI 8: average transaction value
SELECT 8, 'avg_transaction_value_inr', 'Average Transaction Value', 'currency',
    ROUND(t.current_avg_txn, 2), ROUND(t.prior_avg_txn, 2),
    ROUND(100.0 * (t.current_avg_txn - t.prior_avg_txn)
          / NULLIF(t.prior_avg_txn, 0), 2),
    'up'
FROM txn t

ORDER BY kpi_order;

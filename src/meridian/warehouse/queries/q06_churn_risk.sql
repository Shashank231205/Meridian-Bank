-- q06: Churn risk flags
--
-- Grain: one row per active customer.
-- A transparent rule-based score, deliberately not a model. It exists so the
-- business has something interpretable to act on today, and so the fitted
-- logistic model in the ML layer has a baseline to beat. If the model cannot
-- beat a handful of obvious flags, the model is not earning its complexity.
--
-- Churned customers are excluded: scoring the risk of an event that has already
-- happened is how a "predictive" model ends up reporting an AUC of 1.0.
WITH latest_txn AS (
    SELECT MAX(txn_date) AS as_of FROM fact_transaction
),
engagement AS (
    SELECT
        t.customer_key,
        MAX(t.txn_date) AS last_txn_date,
        CAST((julianday((SELECT as_of FROM latest_txn))
              - julianday(MAX(t.txn_date))) / 30.44 AS INTEGER) AS months_inactive,
        COUNT(*) AS n_transactions,
        -- Trend: the recent quarter against the one before it. A collapsing
        -- transaction count is the clearest leading indicator of attrition.
        SUM(CASE WHEN t.txn_date >= date((SELECT as_of FROM latest_txn), '-3 months')
                 THEN 1 ELSE 0 END) AS txn_last_3m,
        SUM(CASE WHEN t.txn_date >= date((SELECT as_of FROM latest_txn), '-6 months')
                  AND t.txn_date <  date((SELECT as_of FROM latest_txn), '-3 months')
                 THEN 1 ELSE 0 END) AS txn_prior_3m
    FROM fact_transaction t
    GROUP BY t.customer_key
),
holdings AS (
    SELECT customer_key,
           COUNT(DISTINCT product_key) AS n_products,
           SUM(balance_inr)            AS total_balance_inr,
           SUM(total_revenue_inr)      AS monthly_revenue_inr
    FROM fact_monthly_account
    WHERE month = (SELECT MAX(month) FROM fact_monthly_account)
    GROUP BY customer_key
),
flags AS (
    SELECT
        c.customer_id,
        c.segment,
        c.tenure_months,
        c.digital_engagement,
        c.acquisition_channel,
        COALESCE(h.n_products, 0)          AS n_products,
        COALESCE(h.total_balance_inr, 0)   AS total_balance_inr,
        COALESCE(h.monthly_revenue_inr, 0) AS monthly_revenue_inr,
        COALESCE(e.months_inactive, 99)    AS months_inactive,
        COALESCE(e.txn_last_3m, 0)         AS txn_last_3m,
        COALESCE(e.txn_prior_3m, 0)        AS txn_prior_3m,

        CASE WHEN COALESCE(e.months_inactive, 99) >= 3     THEN 1 ELSE 0 END AS flag_inactive,
        CASE WHEN COALESCE(h.n_products, 0) <= 1           THEN 1 ELSE 0 END AS flag_single_product,
        CASE WHEN c.tenure_months <= 6                     THEN 1 ELSE 0 END AS flag_new_customer,
        CASE WHEN c.digital_engagement < 0.25              THEN 1 ELSE 0 END AS flag_low_digital,
        CASE WHEN COALESCE(h.total_balance_inr, 0) < 25000 THEN 1 ELSE 0 END AS flag_low_balance,
        -- Volume more than halved quarter on quarter.
        CASE WHEN COALESCE(e.txn_prior_3m, 0) > 0
              AND COALESCE(e.txn_last_3m, 0) < COALESCE(e.txn_prior_3m, 0) * 0.5
             THEN 1 ELSE 0 END AS flag_declining_activity
    FROM dim_customer c
    JOIN fact_customer_snapshot s ON s.customer_key = c.customer_key
    LEFT JOIN engagement e ON e.customer_key = c.customer_key
    LEFT JOIN holdings   h ON h.customer_key = c.customer_key
    WHERE s.churned = 0          -- score only customers still at risk
)
SELECT
    customer_id,
    segment,
    tenure_months,
    n_products,
    ROUND(total_balance_inr, 2)   AS total_balance_inr,
    ROUND(monthly_revenue_inr, 2) AS monthly_revenue_inr,
    months_inactive,
    txn_last_3m,
    txn_prior_3m,
    flag_inactive,
    flag_single_product,
    flag_new_customer,
    flag_low_digital,
    flag_low_balance,
    flag_declining_activity,
    (flag_inactive + flag_single_product + flag_new_customer
     + flag_low_digital + flag_low_balance + flag_declining_activity) AS risk_score,
    CASE
        WHEN (flag_inactive + flag_single_product + flag_new_customer
              + flag_low_digital + flag_low_balance + flag_declining_activity) >= 4
             THEN 'High'
        WHEN (flag_inactive + flag_single_product + flag_new_customer
              + flag_low_digital + flag_low_balance + flag_declining_activity) >= 2
             THEN 'Medium'
        ELSE 'Low'
    END AS risk_band,
    -- Revenue at risk: what the bank loses annually if this customer leaves.
    ROUND(monthly_revenue_inr * 12, 2) AS annual_revenue_at_risk_inr
FROM flags
ORDER BY risk_score DESC, annual_revenue_at_risk_inr DESC;

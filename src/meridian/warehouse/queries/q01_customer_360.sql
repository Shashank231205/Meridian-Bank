-- q01: Customer 360
--
-- Grain: one row per customer.
-- The single joined view of a customer: who they are, what they hold, what they
-- are worth, and whether they are still here. Every other query narrows from
-- something this shape.
--
-- LEFT JOINs throughout: a customer with no transactions is a real customer
-- with a real problem, and an INNER JOIN would hide exactly the disengaged
-- population the retention analysis needs to see.
SELECT
    c.customer_id,
    c.segment,
    c.age,
    c.age_band,
    c.gender,
    c.job,
    c.city,
    c.state,
    c.annual_income_inr,
    c.credit_score,
    c.acquisition_channel,
    c.acquired_date,
    c.tenure_months,
    c.digital_engagement,
    s.churned,
    s.censored,
    s.churn_date,
    s.observed_months,
    COALESCE(h.n_products, 0)            AS n_products,
    COALESCE(h.total_balance_inr, 0)     AS total_balance_inr,
    COALESCE(h.monthly_revenue_inr, 0)   AS monthly_revenue_inr,
    COALESCE(t.n_transactions, 0)        AS n_transactions,
    COALESCE(t.total_txn_value_inr, 0)   AS total_txn_value_inr,
    COALESCE(t.avg_txn_value_inr, 0)     AS avg_txn_value_inr,
    t.last_txn_date,
    -- Months since the last transaction: the engagement signal that matters
    -- most for churn. NULL for customers who have never transacted.
    CASE
        WHEN t.last_txn_date IS NULL THEN NULL
        ELSE CAST((julianday('now') - julianday(t.last_txn_date)) / 30.44 AS INTEGER)
    END                                  AS months_since_last_txn,
    COALESCE(cc.n_contacts, 0)           AS n_campaign_contacts,
    COALESCE(cc.n_conversions, 0)        AS n_campaign_conversions
FROM dim_customer c
LEFT JOIN fact_customer_snapshot s
       ON s.customer_key = c.customer_key
LEFT JOIN (
    SELECT customer_key,
           COUNT(DISTINCT product_key) AS n_products,
           SUM(balance_inr)            AS total_balance_inr,
           SUM(total_revenue_inr)      AS monthly_revenue_inr
    FROM fact_monthly_account
    -- Latest month only: this is a snapshot measure, not a sum over time.
    WHERE month = (SELECT MAX(month) FROM fact_monthly_account)
    GROUP BY customer_key
) h ON h.customer_key = c.customer_key
LEFT JOIN (
    SELECT customer_key,
           COUNT(*)         AS n_transactions,
           SUM(amount_inr)  AS total_txn_value_inr,
           AVG(amount_inr)  AS avg_txn_value_inr,
           MAX(txn_date)    AS last_txn_date
    FROM fact_transaction
    GROUP BY customer_key
) t ON t.customer_key = c.customer_key
LEFT JOIN (
    SELECT customer_key,
           COUNT(*)        AS n_contacts,
           SUM(converted)  AS n_conversions
    FROM fact_campaign_contact
    GROUP BY customer_key
) cc ON cc.customer_key = c.customer_key;

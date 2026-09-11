-- q02: RFM segmentation
--
-- Grain: one row per customer.
-- Recency, Frequency and Monetary scores by quintile (NTILE(5)), combined into
-- the standard RFM cell and then named.
--
-- Recency is scored in reverse: a customer who transacted yesterday should
-- score 5, not 1, so the NTILE runs on descending recency. Getting this
-- backwards is the single most common RFM error and it inverts every
-- conclusion drawn from the segment names.
WITH txn_summary AS (
    SELECT
        f.customer_key,
        MAX(f.txn_date)                                       AS last_txn_date,
        CAST(julianday((SELECT MAX(txn_date) FROM fact_transaction))
             - julianday(MAX(f.txn_date)) AS INTEGER)         AS recency_days,
        COUNT(*)                                              AS frequency,
        SUM(f.amount_inr)                                     AS monetary_inr
    FROM fact_transaction f
    -- Debits only: a salary credit is money arriving, not customer engagement
    -- with the bank's products, and including it would rank payroll accounts
    -- as the most engaged customers in the book.
    WHERE f.direction = 'debit'
    GROUP BY f.customer_key
),
scored AS (
    SELECT
        t.*,
        -- Lower recency_days = more recent = higher score, hence DESC.
        NTILE(5) OVER (ORDER BY t.recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY t.frequency  ASC)    AS f_score,
        NTILE(5) OVER (ORDER BY t.monetary_inr ASC)  AS m_score
    FROM txn_summary t
)
SELECT
    c.customer_id,
    c.segment,
    c.city,
    c.acquisition_channel,
    s.last_txn_date,
    s.recency_days,
    s.frequency,
    ROUND(s.monetary_inr, 2) AS monetary_inr,
    s.r_score,
    s.f_score,
    s.m_score,
    (s.r_score || s.f_score || s.m_score)      AS rfm_cell,
    (s.r_score + s.f_score + s.m_score)        AS rfm_total,
    CASE
        WHEN s.r_score >= 4 AND s.f_score >= 4 AND s.m_score >= 4 THEN 'Champions'
        WHEN s.r_score >= 3 AND s.f_score >= 3 AND s.m_score >= 4 THEN 'Loyal'
        WHEN s.r_score >= 4 AND s.f_score <= 2                    THEN 'New / Promising'
        WHEN s.r_score <= 2 AND s.f_score >= 4 AND s.m_score >= 4 THEN 'At Risk - High Value'
        WHEN s.r_score <= 2 AND s.f_score >= 3                    THEN 'At Risk'
        WHEN s.r_score <= 2 AND s.f_score <= 2                    THEN 'Hibernating'
        WHEN s.m_score >= 4                                       THEN 'Big Spenders'
        ELSE 'Needs Attention'
    END AS rfm_segment,
    snap.churned
FROM scored s
JOIN dim_customer c            ON c.customer_key = s.customer_key
LEFT JOIN fact_customer_snapshot snap ON snap.customer_key = s.customer_key;

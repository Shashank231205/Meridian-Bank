-- q08: Channel performance rollup
--
-- Grain: one row per acquisition channel per transaction channel.
-- Two different senses of "channel" meet here and the query keeps them apart:
-- the channel a customer was *acquired* through, and the channel they now
-- *transact* through. Conflating them is what makes channel-migration analysis
-- impossible -- the interesting finding is precisely that branch-acquired
-- customers migrate to mobile over time.
--
-- GROUPING SETS would be the natural fit; SQLite has no GROUPING SETS, so the
-- rollup is assembled with UNION ALL and a level column.
WITH base AS (
    SELECT
        c.customer_key,
        c.acquisition_channel,
        c.segment,
        ch.channel_code   AS txn_channel,
        ch.is_digital,
        t.amount_inr,
        t.txn_date,
        t.month
    FROM fact_transaction t
    JOIN dim_customer c ON c.customer_key = t.customer_key
    LEFT JOIN dim_channel ch ON ch.channel_key = t.channel_key
),
cross_tab AS (
    SELECT
        'acquisition_x_txn'            AS level,
        acquisition_channel,
        txn_channel,
        COUNT(DISTINCT customer_key)   AS n_customers,
        COUNT(*)                       AS n_transactions,
        SUM(amount_inr)                AS total_value_inr,
        AVG(amount_inr)                AS avg_txn_value_inr,
        SUM(is_digital)                AS n_digital_txns
    FROM base
    GROUP BY acquisition_channel, txn_channel
),
by_acquisition AS (
    SELECT
        'acquisition'                  AS level,
        acquisition_channel,
        NULL                           AS txn_channel,
        COUNT(DISTINCT customer_key)   AS n_customers,
        COUNT(*)                       AS n_transactions,
        SUM(amount_inr)                AS total_value_inr,
        AVG(amount_inr)                AS avg_txn_value_inr,
        SUM(is_digital)                AS n_digital_txns
    FROM base
    GROUP BY acquisition_channel
),
by_txn_channel AS (
    SELECT
        'transaction'                  AS level,
        NULL                           AS acquisition_channel,
        txn_channel,
        COUNT(DISTINCT customer_key)   AS n_customers,
        COUNT(*)                       AS n_transactions,
        SUM(amount_inr)                AS total_value_inr,
        AVG(amount_inr)                AS avg_txn_value_inr,
        SUM(is_digital)                AS n_digital_txns
    FROM base
    GROUP BY txn_channel
),
grand AS (
    SELECT
        'total'                        AS level,
        NULL                           AS acquisition_channel,
        NULL                           AS txn_channel,
        COUNT(DISTINCT customer_key)   AS n_customers,
        COUNT(*)                       AS n_transactions,
        SUM(amount_inr)                AS total_value_inr,
        AVG(amount_inr)                AS avg_txn_value_inr,
        SUM(is_digital)                AS n_digital_txns
    FROM base
),
combined AS (
    SELECT * FROM cross_tab
    UNION ALL SELECT * FROM by_acquisition
    UNION ALL SELECT * FROM by_txn_channel
    UNION ALL SELECT * FROM grand
)
SELECT
    level,
    acquisition_channel,
    txn_channel,
    n_customers,
    n_transactions,
    ROUND(total_value_inr, 2)    AS total_value_inr,
    ROUND(avg_txn_value_inr, 2)  AS avg_txn_value_inr,
    ROUND(100.0 * n_digital_txns / NULLIF(n_transactions, 0), 2) AS digital_share_pct,
    ROUND(1.0 * n_transactions / NULLIF(n_customers, 0), 2)      AS txns_per_customer,
    ROUND(total_value_inr / NULLIF(n_customers, 0), 2)           AS value_per_customer_inr
FROM combined
ORDER BY
    CASE level WHEN 'total' THEN 0 WHEN 'acquisition' THEN 1
               WHEN 'transaction' THEN 2 ELSE 3 END,
    total_value_inr DESC;

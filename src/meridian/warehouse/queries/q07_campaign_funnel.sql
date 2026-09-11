-- q07: Campaign funnel and ROI
--
-- Grain: one row per campaign.
-- Contacts, reach, conversions, cost and return. Every ratio is wrapped in
-- NULLIF so a campaign that contacted nobody yields NULL rather than aborting
-- the whole query with a division by zero -- one empty campaign should not cost
-- you the other eleven.
--
-- Revenue attribution is deliberately conservative and clearly labelled: a
-- conversion is credited with the converting customer's current monthly revenue
-- on that product, annualised. That is an attribution assumption, not a
-- measurement of incremental lift, and the column name says "attributed" so
-- nobody reads it as causal.
WITH contact_stats AS (
    SELECT
        cc.campaign_key,
        COUNT(*)                        AS n_contacts,
        COUNT(DISTINCT cc.customer_key) AS n_customers_reached,
        SUM(cc.converted)               AS n_conversions,
        SUM(cc.cost_inr)                AS total_cost_inr,
        AVG(cc.contact_sequence)        AS avg_contacts_per_customer,
        SUM(CASE WHEN cc.contact_sequence = 1 THEN 1 ELSE 0 END) AS n_first_contacts,
        SUM(CASE WHEN cc.contact_sequence = 1 AND cc.converted = 1
                 THEN 1 ELSE 0 END)     AS n_first_contact_conversions,
        SUM(CASE WHEN cc.prior_outcome = 'success' THEN 1 ELSE 0 END)
                                        AS n_prior_success_contacts,
        SUM(CASE WHEN cc.prior_outcome = 'success' AND cc.converted = 1
                 THEN 1 ELSE 0 END)     AS n_prior_success_conversions
    FROM fact_campaign_contact cc
    GROUP BY cc.campaign_key
),
converted_revenue AS (
    SELECT
        cc.campaign_key,
        SUM(f.total_revenue_inr) AS monthly_revenue_of_converts_inr
    FROM fact_campaign_contact cc
    JOIN dim_campaign dc ON dc.campaign_key = cc.campaign_key
    JOIN dim_product  dp ON dp.product_code = dc.product_code
    JOIN fact_monthly_account f
      ON f.customer_key = cc.customer_key
     AND f.product_key  = dp.product_key
     AND f.month = (SELECT MAX(month) FROM fact_monthly_account)
    WHERE cc.converted = 1
    GROUP BY cc.campaign_key
)
SELECT
    dc.campaign_id,
    dc.campaign_name,
    dc.product_code,
    dc.channel,
    dc.target_segment,
    dc.start_date,
    cs.n_contacts,
    cs.n_customers_reached,
    cs.n_conversions,
    ROUND(cs.avg_contacts_per_customer, 2) AS avg_contacts_per_customer,

    -- Conversion per contact and per customer differ whenever customers are
    -- contacted more than once. Quoting the wrong one overstates performance,
    -- so both are reported.
    ROUND(100.0 * cs.n_conversions / NULLIF(cs.n_contacts, 0), 2)
        AS conversion_rate_per_contact_pct,
    ROUND(100.0 * cs.n_conversions / NULLIF(cs.n_customers_reached, 0), 2)
        AS conversion_rate_per_customer_pct,
    ROUND(100.0 * cs.n_first_contact_conversions / NULLIF(cs.n_first_contacts, 0), 2)
        AS first_contact_conversion_pct,
    ROUND(100.0 * cs.n_prior_success_conversions
          / NULLIF(cs.n_prior_success_contacts, 0), 2)
        AS repeat_buyer_conversion_pct,

    ROUND(cs.total_cost_inr, 2) AS total_cost_inr,
    ROUND(cs.total_cost_inr / NULLIF(cs.n_conversions, 0), 2) AS cost_per_acquisition_inr,
    ROUND(COALESCE(cr.monthly_revenue_of_converts_inr, 0) * 12, 2)
        AS attributed_annual_revenue_inr,
    ROUND(
        100.0 * (COALESCE(cr.monthly_revenue_of_converts_inr, 0) * 12 - cs.total_cost_inr)
        / NULLIF(cs.total_cost_inr, 0), 2
    ) AS roi_pct
FROM dim_campaign dc
JOIN contact_stats cs          ON cs.campaign_key = dc.campaign_key
LEFT JOIN converted_revenue cr ON cr.campaign_key = dc.campaign_key
ORDER BY roi_pct DESC;

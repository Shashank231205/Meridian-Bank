-- ---------------------------------------------------------------------------
-- Meridian Bank -- dimensional warehouse (SQLite)
--
-- A Kimball star schema. The grain of every fact table is stated in its comment
-- and enforced by a unique constraint, because "what is one row here" is the
-- first question anyone should ask of a fact table and the last one people
-- bother to answer.
--
-- Two fact tables cover account balances: fact_monthly_account is a periodic
-- snapshot (customer x product x month) and fact_transaction is at transaction
-- grain. Having both is deliberate. "What was the deposit book worth in March"
-- is a snapshot question; summing transactions to answer it produces a flow
-- where a stock was asked for, which is one of the more common and more
-- expensive mistakes in banking analytics.
--
-- Reference tables preserve the real source data alongside the generated
-- spine, so a query can compare the two without leaving the warehouse.
-- ---------------------------------------------------------------------------

PRAGMA foreign_keys = ON;

-- ===========================================================================
-- DIMENSIONS
-- ===========================================================================

-- Grain: one row per calendar date.
-- A real date dimension rather than date arithmetic in every query: fiscal
-- periods in India start in April, and encoding that once here stops it being
-- re-derived (differently) in ten places.
CREATE TABLE dim_date (
    date_key            INTEGER PRIMARY KEY,   -- YYYYMMDD
    full_date           TEXT    NOT NULL UNIQUE,
    year                INTEGER NOT NULL,
    quarter             INTEGER NOT NULL,
    month               INTEGER NOT NULL,
    month_name          TEXT    NOT NULL,
    day                 INTEGER NOT NULL,
    day_of_week         INTEGER NOT NULL,
    day_name            TEXT    NOT NULL,
    week_of_year        INTEGER NOT NULL,
    is_weekend          INTEGER NOT NULL CHECK (is_weekend IN (0, 1)),
    is_month_end        INTEGER NOT NULL CHECK (is_month_end IN (0, 1)),
    month_start_date    TEXT    NOT NULL,
    -- Indian fiscal year runs April to March.
    fiscal_year         INTEGER NOT NULL,
    fiscal_quarter      INTEGER NOT NULL,
    -- Festive season drives the dominant seasonal pattern in Indian retail.
    is_festive_season   INTEGER NOT NULL CHECK (is_festive_season IN (0, 1))
);

-- Grain: one row per customer. Type 1 SCD -- attributes are overwritten, no
-- history. Stated explicitly because the choice matters: segment and income
-- change over time, and this schema does not keep that history. The monthly
-- snapshot fact carries the time-varying view instead.
CREATE TABLE dim_customer (
    customer_key        INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         TEXT    NOT NULL UNIQUE,
    age                 INTEGER NOT NULL CHECK (age BETWEEN 18 AND 120),
    age_band            TEXT    NOT NULL,
    gender              TEXT    NOT NULL,
    marital             TEXT,
    education           TEXT,
    job                 TEXT,
    is_salaried         INTEGER NOT NULL CHECK (is_salaried IN (0, 1)),
    city                TEXT    NOT NULL,
    state               TEXT    NOT NULL,
    segment             TEXT    NOT NULL,
    annual_income_inr   REAL    NOT NULL CHECK (annual_income_inr >= 0),
    credit_score        INTEGER NOT NULL CHECK (credit_score BETWEEN 300 AND 900),
    digital_engagement  REAL    NOT NULL CHECK (digital_engagement BETWEEN 0 AND 1),
    acquisition_channel TEXT    NOT NULL,
    acquired_date       TEXT    NOT NULL,
    acquired_date_key   INTEGER NOT NULL,
    tenure_months       INTEGER NOT NULL CHECK (tenure_months >= 0),
    FOREIGN KEY (acquired_date_key) REFERENCES dim_date (date_key)
);

-- Grain: one row per product in the catalogue.
CREATE TABLE dim_product (
    product_key         INTEGER PRIMARY KEY AUTOINCREMENT,
    product_code        TEXT    NOT NULL UNIQUE,
    product_name        TEXT    NOT NULL,
    category            TEXT    NOT NULL
                        CHECK (category IN ('deposit','lending','card','investment')),
    interest_rate_pct   REAL    NOT NULL CHECK (interest_rate_pct >= 0),
    rate_offset_pp      REAL    NOT NULL,
    annual_fee_inr      REAL    NOT NULL CHECK (annual_fee_inr >= 0),
    min_income_inr      REAL    NOT NULL CHECK (min_income_inr >= 0),
    is_anchor           INTEGER NOT NULL CHECK (is_anchor IN (0, 1)),
    -- The real World Bank rate every price here was derived from.
    calibrated_from_lending_rate_pct REAL NOT NULL
);

-- Grain: one row per channel.
CREATE TABLE dim_channel (
    channel_key         INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_code        TEXT    NOT NULL UNIQUE,
    channel_name        TEXT    NOT NULL,
    is_digital          INTEGER NOT NULL CHECK (is_digital IN (0, 1)),
    is_assisted         INTEGER NOT NULL CHECK (is_assisted IN (0, 1))
);

-- Grain: one row per branch.
CREATE TABLE dim_branch (
    branch_key          INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_code         TEXT    NOT NULL UNIQUE,
    branch_name         TEXT    NOT NULL,
    city                TEXT    NOT NULL,
    state               TEXT    NOT NULL,
    region              TEXT    NOT NULL
);

-- Grain: one row per marketing campaign.
CREATE TABLE dim_campaign (
    campaign_key        INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id         TEXT    NOT NULL UNIQUE,
    campaign_name       TEXT    NOT NULL,
    product_code        TEXT    NOT NULL,
    channel             TEXT    NOT NULL,
    target_segment      TEXT    NOT NULL,
    start_date          TEXT    NOT NULL,
    end_date            TEXT    NOT NULL,
    cost_per_contact_inr REAL   NOT NULL CHECK (cost_per_contact_inr >= 0)
);

-- ===========================================================================
-- FACTS
-- ===========================================================================

-- Grain: ONE ROW PER CUSTOMER PER PRODUCT PER MONTH.
-- A periodic snapshot: balances as at month end, whether or not anything moved.
-- Semi-additive -- balance_inr sums across customers and products but NOT
-- across months, since adding January's balance to February's is meaningless.
-- Revenue columns are fully additive.
CREATE TABLE fact_monthly_account (
    account_month_key   INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_key        INTEGER NOT NULL,
    product_key         INTEGER NOT NULL,
    date_key            INTEGER NOT NULL,      -- month start
    month               TEXT    NOT NULL,
    balance_inr         REAL    NOT NULL CHECK (balance_inr >= 0),
    interest_rate_pct   REAL    NOT NULL,
    interest_revenue_inr REAL   NOT NULL,
    fee_revenue_inr     REAL    NOT NULL,
    total_revenue_inr   REAL    NOT NULL,
    account_age_months  INTEGER NOT NULL CHECK (account_age_months >= 0),
    FOREIGN KEY (customer_key) REFERENCES dim_customer (customer_key),
    FOREIGN KEY (product_key)  REFERENCES dim_product  (product_key),
    FOREIGN KEY (date_key)     REFERENCES dim_date     (date_key),
    -- The grain, enforced. A duplicate here means the loader ran twice.
    UNIQUE (customer_key, product_key, month)
);

-- Grain: ONE ROW PER TRANSACTION.
-- Fully additive: amount_inr sums across every dimension including time.
CREATE TABLE fact_transaction (
    transaction_key     INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id              TEXT    NOT NULL UNIQUE,
    customer_key        INTEGER NOT NULL,
    date_key            INTEGER NOT NULL,
    channel_key         INTEGER,
    txn_date            TEXT    NOT NULL,
    month               TEXT    NOT NULL,
    txn_type            TEXT    NOT NULL,
    direction           TEXT    NOT NULL CHECK (direction IN ('credit','debit')),
    amount_inr          REAL    NOT NULL CHECK (amount_inr > 0),
    merchant_category   TEXT,
    FOREIGN KEY (customer_key) REFERENCES dim_customer (customer_key),
    FOREIGN KEY (date_key)     REFERENCES dim_date     (date_key),
    FOREIGN KEY (channel_key)  REFERENCES dim_channel  (channel_key)
);

-- Grain: ONE ROW PER CAMPAIGN PER CUSTOMER PER CONTACT ATTEMPT.
-- The sequence column is part of the grain: a customer contacted three times in
-- one campaign has three rows, which is what makes contact-fatigue analysis
-- possible. Collapsing to one row per customer would destroy it.
CREATE TABLE fact_campaign_contact (
    contact_key         INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id          TEXT    NOT NULL UNIQUE,
    campaign_key        INTEGER NOT NULL,
    customer_key        INTEGER NOT NULL,
    date_key            INTEGER NOT NULL,
    contact_sequence    INTEGER NOT NULL CHECK (contact_sequence >= 1),
    contact_date        TEXT    NOT NULL,
    channel             TEXT    NOT NULL,
    prior_outcome       TEXT    NOT NULL,
    n_prior_contacts    INTEGER NOT NULL CHECK (n_prior_contacts >= 0),
    converted           INTEGER NOT NULL CHECK (converted IN (0, 1)),
    cost_inr            REAL    NOT NULL CHECK (cost_inr >= 0),
    FOREIGN KEY (campaign_key) REFERENCES dim_campaign (campaign_key),
    FOREIGN KEY (customer_key) REFERENCES dim_customer (customer_key),
    FOREIGN KEY (date_key)     REFERENCES dim_date     (date_key),
    UNIQUE (campaign_key, customer_key, contact_sequence)
);

-- Grain: ONE ROW PER CUSTOMER.
-- An analytics output table, not a source fact: churn outcome with explicit
-- censoring. The censored flag is load-bearing -- a customer still active when
-- the window closed has not been retained forever, they simply have not been
-- observed long enough, and treating the two as the same understates retention.
CREATE TABLE fact_customer_snapshot (
    snapshot_key        INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_key        INTEGER NOT NULL UNIQUE,
    acquired_date       TEXT    NOT NULL,
    churn_date          TEXT,
    churned             INTEGER NOT NULL CHECK (churned IN (0, 1)),
    censored            INTEGER NOT NULL CHECK (censored IN (0, 1)),
    observed_months     INTEGER NOT NULL CHECK (observed_months >= 0),
    n_products          INTEGER NOT NULL DEFAULT 0,
    total_balance_inr   REAL    NOT NULL DEFAULT 0,
    total_revenue_inr   REAL    NOT NULL DEFAULT 0,
    n_transactions      INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (customer_key) REFERENCES dim_customer (customer_key),
    -- Churned and censored are complements, never both and never neither.
    CHECK (churned + censored = 1),
    -- A churn date exists if and only if the customer churned.
    CHECK ((churned = 1 AND churn_date IS NOT NULL)
        OR (churned = 0 AND churn_date IS NULL))
);

-- ===========================================================================
-- REFERENCE -- real source data, provenance preserved
-- ===========================================================================

-- Grain: one row per indicator per year. Real World Bank observations.
-- Nulls are retained rather than dropped: "no observation yet" is information,
-- and the validation layer reports on it.
CREATE TABLE ref_macro_indicator (
    macro_key           INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator           TEXT    NOT NULL,
    indicator_label     TEXT,
    country_iso3        TEXT    NOT NULL,
    year                INTEGER NOT NULL,
    value               REAL,
    UNIQUE (indicator, country_iso3, year)
);

-- Grain: one row per institution per reporting quarter. Real FDIC financials.
-- NIMY is a percentage and NIM is dollars in thousands; the column comments
-- exist because conflating them produces a four-million-percent margin.
CREATE TABLE ref_peer_financials (
    peer_key            INTEGER PRIMARY KEY AUTOINCREMENT,
    cert                INTEGER NOT NULL,
    institution_name    TEXT,
    state               TEXT,
    report_date         TEXT    NOT NULL,
    asset_thousands_usd REAL,    -- DOLLARS, thousands
    deposits_thousands_usd REAL, -- DOLLARS, thousands
    net_income_thousands_usd REAL,
    nim_thousands_usd   REAL,    -- DOLLARS, thousands -- NOT a percentage
    nimy_pct            REAL,    -- PERCENT -- the one to chart
    roa_pct             REAL,
    roe_pct             REAL,
    efficiency_ratio_pct REAL,
    UNIQUE (cert, report_date)
);

-- Grain: one row per real UCI campaign contact. 41,188 genuine records.
-- The 'duration' column is deliberately absent: it is recorded after the call
-- and leaks the outcome, so it is dropped at ingestion and never lands here.
CREATE TABLE ref_uci_campaign (
    uci_key             INTEGER PRIMARY KEY AUTOINCREMENT,
    age                 INTEGER,
    job                 TEXT,
    marital             TEXT,
    education           TEXT,
    default_credit      TEXT,
    housing             TEXT,
    loan                TEXT,
    contact             TEXT,
    month               TEXT,
    day_of_week         TEXT,
    campaign            INTEGER,
    pdays               INTEGER,
    previous            INTEGER,
    poutcome            TEXT,
    emp_var_rate        REAL,
    cons_price_idx      REAL,
    cons_conf_idx       REAL,
    euribor3m           REAL,
    nr_employed         REAL,
    subscribed          INTEGER NOT NULL CHECK (subscribed IN (0, 1))
);

-- ===========================================================================
-- INDEXES -- on the join and filter columns the ten analytical queries use
-- ===========================================================================

CREATE INDEX idx_fma_customer      ON fact_monthly_account (customer_key);
CREATE INDEX idx_fma_product       ON fact_monthly_account (product_key);
CREATE INDEX idx_fma_month         ON fact_monthly_account (month);
CREATE INDEX idx_fma_date          ON fact_monthly_account (date_key);

CREATE INDEX idx_txn_customer      ON fact_transaction (customer_key);
CREATE INDEX idx_txn_month         ON fact_transaction (month);
CREATE INDEX idx_txn_date          ON fact_transaction (date_key);
CREATE INDEX idx_txn_type          ON fact_transaction (txn_type);

CREATE INDEX idx_contact_campaign  ON fact_campaign_contact (campaign_key);
CREATE INDEX idx_contact_customer  ON fact_campaign_contact (customer_key);

CREATE INDEX idx_cust_segment      ON dim_customer (segment);
CREATE INDEX idx_cust_acquired     ON dim_customer (acquired_date);
CREATE INDEX idx_cust_channel      ON dim_customer (acquisition_channel);

CREATE INDEX idx_snapshot_churned  ON fact_customer_snapshot (churned);

-- ===========================================================================
-- VIEWS -- the joins every query would otherwise repeat
-- ===========================================================================

CREATE VIEW v_customer_360 AS
SELECT
    c.customer_id,
    c.segment,
    c.city,
    c.state,
    c.age,
    c.age_band,
    c.job,
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
    s.n_products,
    s.total_balance_inr,
    s.total_revenue_inr,
    s.n_transactions
FROM dim_customer c
LEFT JOIN fact_customer_snapshot s ON s.customer_key = c.customer_key;

CREATE VIEW v_monthly_revenue AS
SELECT
    f.month,
    p.category,
    p.product_code,
    p.product_name,
    c.segment,
    COUNT(DISTINCT f.customer_key) AS n_customers,
    SUM(f.balance_inr)             AS balance_inr,
    SUM(f.interest_revenue_inr)    AS interest_revenue_inr,
    SUM(f.fee_revenue_inr)         AS fee_revenue_inr,
    SUM(f.total_revenue_inr)       AS total_revenue_inr
FROM fact_monthly_account f
JOIN dim_product  p ON p.product_key  = f.product_key
JOIN dim_customer c ON c.customer_key = f.customer_key
GROUP BY f.month, p.category, p.product_code, p.product_name, c.segment;

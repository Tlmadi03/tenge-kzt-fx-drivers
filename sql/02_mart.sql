-- Wide FX table and the analysis mart.

CREATE OR REPLACE TABLE fx_wide AS
SELECT market_date,
       max(rate_per_unit) FILTER (WHERE code = 'USD') AS usd_kzt,   -- KZT per USD
       max(rate_per_unit) FILTER (WHERE code = 'EUR') AS eur_kzt,
       max(rate_per_unit) FILTER (WHERE code = 'RUB') AS rub_kzt    -- KZT per RUB
FROM rates
GROUP BY market_date;

-- As-of join: each trading day gets the latest Brent print on or before it.
-- Prints older than 5 calendar days are treated as missing.
CREATE OR REPLACE TABLE mart_daily AS
SELECT f.market_date,
       f.usd_kzt,
       f.eur_kzt,
       f.rub_kzt,
       f.usd_kzt / f.rub_kzt AS rub_usd,      -- RUB per USD (cross rate via KZT)
       f.eur_kzt / f.usd_kzt AS eur_usd,      -- USD per EUR (cross rate via KZT)
       CASE WHEN date_diff('day', b.obs_date, f.market_date) <= 5 THEN b.brent_usd END AS brent_usd,
       CASE WHEN date_diff('day', b.obs_date, f.market_date) <= 5 THEN b.obs_date END AS brent_obs_date,
       CASE WHEN date_diff('day', b.obs_date, f.market_date) <= 5
            THEN date_diff('day', b.obs_date, f.market_date) END AS brent_staleness_days,
       CASE WHEN date_diff('day', b.obs_date, f.market_date) <= 5
            THEN b.brent_usd * f.usd_kzt END AS brent_kzt,   -- oil price in tenge
       r.regime,
       r.regime_group
FROM fx_wide f
ASOF LEFT JOIN brent b ON f.market_date >= b.obs_date
LEFT JOIN regimes r ON f.market_date BETWEEN r.start_date AND r.end_date
ORDER BY f.market_date;

-- Consecutive-observation log returns (positional, no gap handling; gap handling
-- for multi-day horizons is done in Python, see tenge_fx.analysis.build_returns).
CREATE OR REPLACE VIEW returns_1d AS
SELECT market_date,
       regime,
       ln(usd_kzt  / lag(usd_kzt)  OVER w) AS r_usd_kzt,
       ln(rub_usd  / lag(rub_usd)  OVER w) AS r_rub_usd,
       ln(brent_usd / lag(brent_usd) OVER w) AS r_brent,
       date_diff('day', lag(market_date) OVER w, market_date) AS span_days
FROM mart_daily
WINDOW w AS (ORDER BY market_date);

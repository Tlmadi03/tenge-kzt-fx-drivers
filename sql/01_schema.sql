-- Base tables. Inputs are the cleaned CSVs written by `python -m tenge_fx.clean`;
-- `regimes` is registered from tenge_fx.regimes by build_db.py.

CREATE OR REPLACE TABLE rates AS
SELECT CAST(market_date AS DATE) AS market_date,   -- trading day the rate reflects
       CAST(nbk_date    AS DATE) AS nbk_date,      -- NBK publication date
       code,
       CAST(rate_per_unit AS DOUBLE) AS rate_per_unit  -- KZT per ONE unit of `code`
FROM read_csv_auto('{nbk_long}');

CREATE OR REPLACE TABLE brent AS
SELECT CAST(obs_date AS DATE) AS obs_date,
       CAST(brent_usd AS DOUBLE) AS brent_usd      -- USD per barrel, FRED DCOILBRENTEU
FROM read_csv_auto('{brent_clean}');

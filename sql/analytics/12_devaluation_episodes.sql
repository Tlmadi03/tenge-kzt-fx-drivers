-- Depreciation episodes: maximal runs of consecutive trading days on which USD/KZT is
-- more than 10% above its level 20 trading days earlier (gaps-and-islands).
-- Runs shorter than 3 trading days are dropped (isolated threshold crossings,
-- mostly rebounds after a strong appreciation, not devaluation episodes).
WITH base AS (
    SELECT market_date, usd_kzt,
           ln(usd_kzt / lag(usd_kzt, 20) OVER (ORDER BY market_date)) AS chg20,
           date_diff('day', lag(market_date, 20) OVER (ORDER BY market_date), market_date) AS span20
    FROM mart_daily
),
flag AS (
    SELECT *, coalesce(chg20 > ln(1.10) AND span20 <= 40, false) AS hot FROM base
),
isl AS (
    SELECT *,
           row_number() OVER (ORDER BY market_date)
         - row_number() OVER (PARTITION BY hot ORDER BY market_date) AS grp
    FROM flag
)
SELECT min(market_date)                      AS episode_start,
       max(market_date)                      AS episode_end,
       count(*)                              AS trading_days,
       round(max(chg20) * 100, 1)            AS peak_20d_depreciation_pct,
       round(min(usd_kzt), 2)                AS usd_kzt_min,
       round(max(usd_kzt), 2)                AS usd_kzt_max
FROM isl
WHERE hot
GROUP BY grp
HAVING count(*) >= 3
ORDER BY episode_start;

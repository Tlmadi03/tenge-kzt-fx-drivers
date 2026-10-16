-- Per-regime summary of daily USD/KZT behaviour and its raw 1-day correlations.
-- (Block-permutation inference is in Python; these are descriptive.)
WITH r AS (
    SELECT * FROM returns_1d WHERE r_usd_kzt IS NOT NULL AND span_days <= 7 AND regime IS NOT NULL
)
SELECT regime,
       count(*)                                          AS n_days,
       min(market_date)                                  AS first_day,
       max(market_date)                                  AS last_day,
       round(avg(r_usd_kzt) * 1e4, 2)                    AS mean_daily_move_bp,
       round(stddev_samp(r_usd_kzt) * sqrt(252) * 100, 2) AS usdkzt_vol_ann_pct,
       round(corr(r_usd_kzt, r_rub_usd), 3)              AS corr_1d_rub,
       round(corr(r_usd_kzt, r_brent), 3)                AS corr_1d_brent,
       count(r_brent)                                    AS n_with_brent
FROM r
GROUP BY regime
ORDER BY first_day;

-- Realised volatility of daily USD/KZT log returns by calendar year.
-- Returns spanning a data gap (> 7 calendar days) are excluded.
WITH r AS (
    SELECT market_date, r_usd_kzt, r_rub_usd, r_brent
    FROM returns_1d
    WHERE r_usd_kzt IS NOT NULL AND span_days <= 7
)
SELECT year(market_date)                                  AS year,
       count(*)                                           AS n_days,
       round(stddev_samp(r_usd_kzt) * sqrt(252) * 100, 2) AS usdkzt_vol_ann_pct,
       round(stddev_samp(r_rub_usd) * sqrt(252) * 100, 2) AS rubusd_vol_ann_pct,
       round(stddev_samp(r_brent)   * sqrt(252) * 100, 2) AS brent_vol_ann_pct,
       round(max(abs(r_usd_kzt)) * 100, 2)                AS max_abs_daily_move_pct
FROM r
GROUP BY 1
ORDER BY 1;

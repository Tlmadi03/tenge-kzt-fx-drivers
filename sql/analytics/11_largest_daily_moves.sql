-- 20 largest absolute one-day USD/KZT moves, with the same-day RUB/USD and Brent moves.
SELECT market_date,
       regime,
       round(r_usd_kzt * 100, 2) AS usdkzt_move_pct,   -- + = tenge weakens
       round(r_rub_usd * 100, 2) AS rubusd_move_pct,   -- + = ruble weakens
       round(r_brent   * 100, 2) AS brent_move_pct,
       span_days
FROM returns_1d
WHERE r_usd_kzt IS NOT NULL AND span_days <= 7
ORDER BY abs(r_usd_kzt) DESC
LIMIT 20;

"""Build the DuckDB database: base tables, analysis mart, and analytical query outputs."""
import duckdb

from . import config
from .regimes import regimes_frame


def run_sql_file(con, path, **fmt):
    sql = path.read_text()
    if fmt:
        sql = sql.format(**fmt)
    con.execute(sql)


def build(db_path=config.DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    reg = regimes_frame()
    con.register("regimes_df", reg)
    con.execute("CREATE TABLE regimes AS SELECT * FROM regimes_df")
    run_sql_file(con, config.SQL_DIR / "01_schema.sql",
                 nbk_long=str(config.PROCESSED / "nbk_long.csv"),
                 brent_clean=str(config.PROCESSED / "brent_clean.csv"))
    run_sql_file(con, config.SQL_DIR / "02_mart.sql")
    out_dir = config.REPORTS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted((config.SQL_DIR / "analytics").glob("*.sql")):
        df = con.execute(f.read_text()).df()
        df.to_csv(out_dir / f"{f.stem}.csv", index=False)
        print(f"== {f.stem} ({len(df)} rows)")
        print(df.head(25).to_string(index=False))
    n, a, b = con.execute("SELECT count(*), min(market_date), max(market_date) FROM mart_daily").fetchone()
    print(f"mart_daily: {n} rows {a}..{b}")
    con.close()


def main():
    build()


if __name__ == "__main__":
    main()

"""Paths, constants and the fixed random seed used across the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SQL_DIR = ROOT / "sql"
DB_PATH = PROCESSED / "tenge.duckdb"

SEED = 42

NBK_URL = "https://nationalbank.kz/rss/get_rates.cfm"
FRED_BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
ECB_HIST_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"

START_DATE = "2005-01-01"

# Currencies kept after cleaning; everything else in the NBK feed is not used.
KEEP_CODES = ("USD", "EUR", "RUB")

NBK_RAW_FILE = RAW / "nbk_rates_raw.csv"
BRENT_RAW_FILE = RAW / "brent_fred.csv"
ECB_RAW_FILE = RAW / "ecb_eurofxref_hist.csv"

# Horizons (trading days) for return correlations.
HORIZONS = (1, 5, 20)

# Permutation / bootstrap settings.
N_PERM = 5000
N_BOOT = 2000

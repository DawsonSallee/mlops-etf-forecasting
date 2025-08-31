from pathlib import Path

# This is the single source of truth for all paths in the project.
PROJECT_ROOT = Path(__file__).resolve().parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "etf_features.parquet"

# MLflow paths
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = "file:" + str(MLRUNS_DIR)

# Model registry name
MODEL_NAME = "etf-xgboost-predictor"

# Notebooks directory (for saving artifacts)
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SHAP_SUMMARY_PLOT_PATH = NOTEBOOKS_DIR / "shap_summary_champion.png"

# Ticker and date configurations
TARGET_TICKER = 'SPY'
FEATURE_TICKERS = ['QQQ', 'UVXY', 'TLT', 'GLD', 'AAPL', 'MSFT']
ALL_TICKERS = [TARGET_TICKER] + FEATURE_TICKERS
START_DATE = '2010-01-01'
END_DATE = '2025-8-18'

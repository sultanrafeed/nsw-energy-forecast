from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
MODEL_PATH = MODELS_DIR / "model.json"

TARGET_COL = "demand"
MLFLOW_EXPERIMENT = "nsw-electricity-demand"

# Columns the trained model expects, in order. Kept explicit (not inferred)
# so serve.py and train.py can never silently drift apart.
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "demand_lag_1",
    "demand_lag_48",
    "demand_rolling_mean_48",
    "temp_proxy",
]

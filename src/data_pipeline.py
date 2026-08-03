"""
Loads raw AEMO NSW price/demand CSVs from data/raw/, cleans them, and
engineers a leakage-safe feature set for time-series forecasting.

The Kaggle "Electricity Price and Demand NSW 2018-2023" dataset ships as
one or more CSVs with columns similar to:
    REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE

This module is defensive about exact column names (Kaggle exports vary
slightly by upload) and normalises them early.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import FEATURES_PATH, PROCESSED_DIR, RAW_DIR, TARGET_COL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "totaldemand": "demand",
    "total_demand": "demand",
    "demand": "demand",
    "rrp": "price",
    "settlementdate": "timestamp",
    "date": "timestamp",
}


def _load_raw() -> pd.DataFrame:
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs found in {RAW_DIR}. Download the Kaggle dataset "
            "'Electricity Price and Demand NSW 2018-2023' and place the "
            "CSV file(s) there."
        )
    frames = []
    for path in csvs:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d raw rows from %d file(s)", len(raw), len(csvs))
    return raw


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    df = df.dropna(subset=["demand"])
    # Guard against sentinel/garbage values sometimes present in raw exports
    df = df[(df["demand"] > 0) & (df["demand"] < 20000)]
    df = df.set_index("timestamp")
    # AEMO data is half-hourly; reindex to a full regular grid and
    # interpolate small gaps rather than silently dropping them.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    df = df.reindex(full_index)
    df["demand"] = df["demand"].interpolate(limit=4)
    if "price" in df.columns:
        df["price"] = df["price"].interpolate(limit=4)
    df.index.name = "timestamp"
    return df.dropna(subset=["demand"])


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # Lag features — critical for time series, and each lag is computed
    # strictly from the past, never the target's own row.
    df["demand_lag_1"] = df["demand"].shift(1)
    df["demand_lag_48"] = df["demand"].shift(48)  # same time, previous day
    df["demand_rolling_mean_48"] = df["demand"].shift(1).rolling(48).mean()

    # Placeholder for an exogenous feature (e.g. temperature). Kept as a
    # named column so the model interface is stable even before a weather
    # source is wired in — a real deployment would join BOM data here.
    df["temp_proxy"] = 0.0

    df = df.dropna(subset=["demand_lag_1", "demand_lag_48", "demand_rolling_mean_48"])
    return df


def build_features() -> pd.DataFrame:
    raw = _load_raw()
    clean = _clean(raw)
    features = _engineer_features(clean)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH)
    logger.info("Wrote %d feature rows to %s", len(features), FEATURES_PATH)
    return features


if __name__ == "__main__":
    build_features()

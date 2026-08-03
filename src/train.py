"""
Trains an XGBoost demand forecaster and logs the run to MLflow, alongside
a naive baseline for an honest lift comparison. Uses a time-ordered
holdout (never a random split) since this is a time series.
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src.config import FEATURE_COLS, FEATURES_PATH, MLFLOW_EXPERIMENT, MODEL_PATH, MODELS_DIR, TARGET_COL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOLDOUT_DAYS = 90


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    return df[df.index <= cutoff], df[df.index > cutoff]


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def train() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    train_df, test_df = _time_split(df)
    logger.info("Train: %d rows | Test (holdout): %d rows", len(train_df), len(test_df))

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        # --- Baseline: naive "same time yesterday" forecast ---
        baseline_pred = X_test["demand_lag_48"]
        baseline_metrics = _metrics(y_test, baseline_pred)
        for k, v in baseline_metrics.items():
            mlflow.log_metric(f"baseline_{k}", v)
        logger.info("Baseline metrics: %s", baseline_metrics)

        # --- Model ---
        params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        model_metrics = _metrics(y_test, preds)
        for k, v in model_metrics.items():
            mlflow.log_metric(f"model_{k}", v)
        logger.info("Model metrics: %s", model_metrics)

        lift_mape = (baseline_metrics["mape"] - model_metrics["mape"]) / baseline_metrics["mape"]
        mlflow.log_metric("mape_lift_over_baseline", lift_mape)
        logger.info("MAPE lift over naive baseline: %.1f%%", lift_mape * 100)

        mlflow.xgboost.log_model(model, artifact_path="model")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(MODEL_PATH)
        logger.info("Saved local model copy to %s for serving", MODEL_PATH)


if __name__ == "__main__":
    train()

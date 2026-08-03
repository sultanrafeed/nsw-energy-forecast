"""
FastAPI inference service for the demand forecasting model.

Loads the model saved by train.py once at startup (not per-request), and
exposes /predict, /health, and a minimal /metrics counter — enough to
demonstrate serving discipline without pulling in a full Prometheus stack.
"""
from __future__ import annotations

import logging
import time

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.config import FEATURE_COLS, MODEL_PATH
from src.schemas import PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v1"

_state = {"model": None, "request_count": 0, "started_at": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        logger.warning("No trained model found at %s — run `python -m src.train` first.", MODEL_PATH)
    else:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        _state["model"] = model
        logger.info("Model loaded from %s", MODEL_PATH)
    yield
    _state["model"] = None


app = FastAPI(
    title="NSW Electricity Demand Forecast API",
    version=MODEL_VERSION,
    description="Predicts half-hourly NSW electricity demand (MW).",
    lifespan=lifespan,
)


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        logger.warning("No trained model found at %s — run `python -m src.train` first.", MODEL_PATH)
        return
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    _state["model"] = model
    logger.info("Model loaded from %s", MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _state["model"] is not None else "degraded_no_model",
        "uptime_seconds": round(time.time() - _state["started_at"], 1),
    }


@app.get("/metrics")
def metrics() -> dict:
    return {"request_count": _state["request_count"], "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")

    row = [[getattr(payload, col) for col in FEATURE_COLS]]
    pred = float(_state["model"].predict(row)[0])
    _state["request_count"] += 1

    return PredictionResponse(predicted_demand_mw=round(pred, 2), model_version=MODEL_VERSION)

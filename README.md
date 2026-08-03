# NSW Electricity Demand & Price Forecasting Platform

Production-style ML system that forecasts half-hourly NSW electricity demand
and price using the AEMO NSW Price & Demand dataset (2018-2023, Kaggle).

Built to demonstrate an end-to-end ML engineering workflow — not just a model
in a notebook.

## Architecture

```
                 ┌─────────────┐
  raw CSVs  ───▶ │ data_pipeline│ ──▶ processed parquet (features + target)
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐        ┌───────────────┐
                 │   train.py   │ ─────▶ │  MLflow        │
                 │ (XGBoost +   │        │  tracking +    │
                 │  baseline)   │        │  model registry│
                 └─────────────┘        └───────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  serve.py    │  FastAPI /predict, /health, /metrics
                 │  (loads      │
                 │  registered  │
                 │  model)      │
                 └─────────────┘
                        │
                        ▼
                 Docker container ──▶ deployable to ECS/Cloud Run/AKS
```

## Stack
Python 3.11 · pandas · XGBoost · MLflow · FastAPI · Pydantic · Docker ·
pytest · GitHub Actions

## Getting the data
1. Download "Electricity Price and Demand NSW 2018-2023" from Kaggle.
2. Place the CSV(s) in `data/raw/`.
3. Run the pipeline (below).

## Running locally
```bash
pip install -r requirements.txt

# 1. Build features from raw data
python -m src.data_pipeline

# 2. Train + log to MLflow (starts local mlflow ui on :5000 separately)
mlflow ui &
python -m src.train

# 3. Serve the trained model
uvicorn src.serve:app --reload --port 8000
```

## Running with Docker
```bash
docker compose up --build
# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000
```

## Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "hour": 18, "day_of_week": 2, "month": 7,
        "demand_lag_1": 8123.4, "demand_lag_48": 7890.1,
        "demand_rolling_mean_48": 8000.2, "temp_proxy": 0.0
      }'
```

## What's deliberately NOT in v1 (documented as roadmap, not gaps)
- Kubernetes deployment (docker-compose is sufficient to prove the concept;
  a k8s manifest is a natural next step and is described in `docs/scaling.md`)
- Automated retraining on a schedule (Airflow/Dagster DAG — next iteration)
- Full drift monitoring dashboard (Evidently stub included in `src/monitor.py`)
- Cloud deployment (AWS ECS task definition sketched in `docs/scaling.md`)

## Metrics tracked
- MAE / RMSE / MAPE on a **time-ordered holdout** (last 3 months) — never a
  random split, to avoid leakage in a time series.
- Baseline (naive lag-48 forecast) vs XGBoost, so the model's lift over a
  trivial forecast is explicit and honest.

## CV bullet (use verbatim or adapt)
"Built and deployed an end-to-end electricity demand forecasting system on
NSW AEMO data — MLflow experiment tracking, FastAPI inference service,
Dockerized, with time-series-safe validation showing X% MAPE improvement
over a naive baseline."

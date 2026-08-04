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
                    Docker container ──▶ deployed on Render (free tier)
                    │
                    ▼
                    Weekly GitHub Actions retrain job ──▶ champion/challenger
                    gate: new model only promoted if it beats the current MAPE
                 
```
## Stack
Python 3.11 · pandas · XGBoost · MLflow · FastAPI · Pydantic · Docker ·
pytest · GitHub Actions (CI + scheduled retraining) · Render

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
curl -X POST https://nsw-energy-forecast.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
        "hour": 18, "day_of_week": 2, "month": 7,
        "demand_lag_1": 8123.4, "demand_lag_48": 7890.1,
        "demand_rolling_mean_48": 8000.2, "temp_proxy": 0.0
      }'
```

## Results

Validated with a strict time-based holdout (last 4,320 rows, ~3 months —
never shuffled, since a random split would leak future information into
training for a time series model).

| | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| Naive baseline (same time last week) | 406.1 | 577.8 | 5.42% |
| XGBoost (this model) | 114.2 | 155.0 | **1.50%** |

**72.4% reduction in MAPE over the naive baseline**, on 5 years of real AEMO
NSW demand data (91,824 training rows / 4,320 holdout rows).

## Automated retraining
`.github/workflows/retrain.yml` runs weekly, retrains on the latest data,
and only promotes the new model into production if its holdout MAPE beats
the currently deployed ("champion") model — a basic champion/challenger
gate rather than blind scheduled retraining that could silently regress
production. See `docs/scaling.md` for how this would extend to a live data
source and a proper MLflow registry stage instead of the current flat-file
champion record.

## What's deliberately NOT in v1 (documented as roadmap, not gaps)
- Kubernetes deployment (a single container on Render/Fargate is sufficient
  to prove the concept; a k8s manifest is described in `docs/scaling.md`)
- Live data ingestion for retraining (the weekly job currently re-runs
  against the same static CSV — pulling fresh AEMO data on each run is the
  natural next step, noted in `retrain.yml`)
- Full drift monitoring dashboard (PSI check exists in `src/monitor.py`;
  wiring it to a schedule + alerting is sketched in `docs/scaling.md`)
- Cloud deployment beyond Render's free tier (AWS ECS task definition
  sketched in `docs/scaling.md` for when it needs to scale past a demo)

## Metrics tracked
- MAE / RMSE / MAPE on a **time-ordered holdout** (last ~3 months) — never a
  random split, to avoid leakage in a time series.
- Baseline (naive lag-168h / "same time last week" forecast) vs XGBoost, so
  the model's lift over a trivial forecast is explicit and honest.

## CV bullet
"Built and deployed an end-to-end electricity demand forecasting system
(XGBoost, FastAPI, MLflow, Docker) on 5 years of real NSW AEMO grid data,
achieving 1.5% MAPE — a 72.4% error reduction over a naive baseline — with
automated weekly retraining via a champion/challenger CI pipeline."
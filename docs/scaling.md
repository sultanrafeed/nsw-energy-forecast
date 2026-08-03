# Scaling & Cloud Deployment Roadmap

This document exists so the repo shows deliberate engineering judgement
about what production-scale would require, without over-building a
portfolio project into something nobody will ever run at that scale.

## Cloud deployment (AWS)
- Push the Docker image to ECR.
- Deploy as an ECS Fargate service (simpler than EKS for a single
  stateless FastAPI service) behind an Application Load Balancer.
- Store the registered model in S3; MLflow's artifact store points there
  instead of local disk.
- Use SageMaker if the team already standardises on it — but for a single
  lightweight model, ECS + S3 + MLflow is materially cheaper and easier to
  reason about than a full SageMaker pipeline.

## Retraining
- A weekly retraining job via Airflow or a simple GitHub Actions cron
  (`schedule:` trigger) re-runs `data_pipeline.py` + `train.py` against
  fresh data and only promotes the new model in the MLflow registry if its
  holdout MAPE beats the currently deployed version — a basic champion/
  challenger gate.

## Monitoring at scale
- `src/monitor.py`'s PSI check would run against a rolling window of
  live request features, on a schedule, with results pushed to
  CloudWatch/Grafana and an alert if PSI > 0.2 for 3 consecutive windows.
- Prediction latency and error rate exported via the existing `/metrics`
  endpoint, scraped by Prometheus.

## Kubernetes
- Not used here deliberately — Fargate/Cloud Run gets the same production
  properties (auto-restart, horizontal scaling, zero server management)
  with far less YAML for a single-service system. A k8s Deployment +
  HPA manifest would be the natural next step if this became one service
  among many sharing a cluster.

## Cost/latency notes
- XGBoost inference here is CPU-only and sub-10ms per request — no GPU or
  batching layer needed at this scale. If request volume grew by orders
  of magnitude, the next lever would be response caching for repeated
  feature vectors (e.g. Redis) before reaching for GPU inference.

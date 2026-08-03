from fastapi.testclient import TestClient

from src.serve import app

client = TestClient(app)


def test_health_endpoint_responds():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_predict_rejects_invalid_hour():
    payload = {
        "hour": 99,  # invalid, should fail validation
        "day_of_week": 1,
        "month": 6,
        "demand_lag_1": 8000.0,
        "demand_lag_48": 7900.0,
        "demand_rolling_mean_48": 7950.0,
        "temp_proxy": 0.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_valid_payload_shape():
    payload = {
        "hour": 18,
        "day_of_week": 2,
        "month": 7,
        "demand_lag_1": 8123.4,
        "demand_lag_48": 7890.1,
        "demand_rolling_mean_48": 8000.2,
        "temp_proxy": 0.0,
    }
    resp = client.post("/predict", json=payload)
    # 503 if no model has been trained yet in this environment, 200 if it has
    assert resp.status_code in (200, 503)

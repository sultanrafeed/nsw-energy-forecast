import pandas as pd

from src.data_pipeline import _engineer_features


def test_engineer_features_no_leakage():
    idx = pd.date_range("2023-01-01", periods=200, freq="30min")
    df = pd.DataFrame({"demand": range(200)}, index=idx)

    out = _engineer_features(df)

    # lag_1 at time t must equal demand at t-1, never t itself
    assert (out["demand_lag_1"].iloc[5] == out["demand"].iloc[4])
    # rows before the max lag window (48) should have been dropped
    assert out.index.min() > idx[47]


def test_engineer_features_creates_expected_columns():
    idx = pd.date_range("2023-01-01", periods=100, freq="30min")
    df = pd.DataFrame({"demand": range(100)}, index=idx)
    out = _engineer_features(df)
    for col in ["hour", "day_of_week", "month", "demand_lag_1", "demand_lag_48", "demand_rolling_mean_48"]:
        assert col in out.columns

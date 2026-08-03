"""
Minimal drift check: compares the feature distribution of a new batch of
incoming requests against the training distribution using population
stability index (PSI). A full deployment would run this on a schedule
and push results to a dashboard (e.g. Evidently, or a Grafana panel
backed by these numbers) — this module shows the core logic without
pulling in extra infra for a portfolio project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index. >0.2 typically flags meaningful drift."""
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = expected.quantile(quantiles).values
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_pct = pd.cut(expected, breakpoints).value_counts(normalize=True).sort_index()
    actual_pct = pd.cut(actual, breakpoints).value_counts(normalize=True).sort_index()

    eps = 1e-6
    return float(
        np.sum((actual_pct - expected_pct) * np.log((actual_pct + eps) / (expected_pct + eps)))
    )


def check_drift(train_feature: pd.Series, live_feature: pd.Series, threshold: float = 0.2) -> dict:
    score = psi(train_feature, live_feature)
    return {"psi": score, "drift_detected": score > threshold, "threshold": threshold}

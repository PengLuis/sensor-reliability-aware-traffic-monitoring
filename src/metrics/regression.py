"""Regression metrics for traffic forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np


PRIMARY_METRICS = ("MAE", "RMSE", "MAPE")
HORIZON_STEPS = (3, 6, 12)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0e-5) -> float:
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.nanmean(np.abs((y_pred - y_true) / denom)) * 100.0)


def horizon_mae(y_true: np.ndarray, y_pred: np.ndarray, steps: tuple[int, ...] = HORIZON_STEPS) -> dict[str, float]:
    values: dict[str, float] = {}
    horizon = y_true.shape[1]
    for step in steps:
        if step <= horizon:
            values[f"mae_h{step}"] = mae(y_true[:, step - 1], y_pred[:, step - 1])
    return values


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    metrics.update(horizon_mae(y_true, y_pred))
    return metrics


def metric_manifest() -> dict[str, Any]:
    """Return the planned metric manifest."""

    return {
        "primary": PRIMARY_METRICS,
        "horizon_steps": HORIZON_STEPS,
    }

"""Mandatory baseline models."""

from __future__ import annotations

import numpy as np


MANDATORY_BASELINES = (
    "HistoricalAverage",
    "Persistence",
    "GRU",
    "TCN",
)


def list_mandatory_baselines() -> tuple[str, ...]:
    """Return mandatory baseline names."""

    return MANDATORY_BASELINES


def historical_average_predict(x: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the input-window mean for each forecast step."""

    avg = np.nanmean(x, axis=1, keepdims=True)
    return np.repeat(avg, horizon, axis=1)


def persistence_predict(x: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last input observation for each forecast step."""

    last = x[:, -1:, :, :]
    return np.repeat(last, horizon, axis=1)

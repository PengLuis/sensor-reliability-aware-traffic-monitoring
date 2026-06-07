"""Fault simulation protocols for traffic sensor inputs.

All functions corrupt input observations only. They return:
``(corrupted_x, fault_mask, metadata)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


REQUIRED_FAULTS = (
    "random_missing",
    "continuous_outage",
    "gaussian_noise",
    "linear_drift",
    "stuck_at_last_value",
)

SEVERITY_SCALE = {
    "low": 0.10,
    "medium": 0.30,
    "high": 0.50,
}


def list_required_faults() -> tuple[str, ...]:
    """Return mandatory fault protocol names."""

    return REQUIRED_FAULTS


def _as_batched(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x)
    if arr.ndim == 3:
        return arr[None, ...], False
    if arr.ndim == 4:
        return arr, True
    raise ValueError(f"Expected [T,N,F] or [B,T,N,F], got shape {arr.shape}")


def _restore_batch(arr: np.ndarray, mask: np.ndarray, was_batched: bool) -> tuple[np.ndarray, np.ndarray]:
    if was_batched:
        return arr, mask
    return arr[0], mask[0]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _scale_from_severity(severity: str | float | int) -> float:
    if isinstance(severity, str):
        if severity not in SEVERITY_SCALE:
            raise ValueError(f"Unknown severity: {severity}")
        return SEVERITY_SCALE[severity]
    return float(severity)


def random_missing(
    x: np.ndarray,
    rate: float,
    seed: int | None = None,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Randomly mark individual observations as missing."""

    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    batched, was_batched = _as_batched(x)
    corrupted = batched.copy()
    gen = _rng(seed)
    mask = gen.random(batched.shape) < rate
    corrupted[mask] = fill_value
    metadata = {"fault": "random_missing", "rate": rate, "seed": seed, "fill_value": fill_value}
    out, out_mask = _restore_batch(corrupted, mask, was_batched)
    return out, out_mask, metadata


def continuous_outage(
    x: np.ndarray,
    length: int,
    seed: int | None = None,
    sensor_rate: float = 0.10,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply contiguous missing blocks to a subset of sensors."""

    if length <= 0:
        raise ValueError("length must be positive")
    if not 0 < sensor_rate <= 1:
        raise ValueError("sensor_rate must be in (0, 1]")
    batched, was_batched = _as_batched(x)
    corrupted = batched.copy()
    mask = np.zeros_like(batched, dtype=bool)
    gen = _rng(seed)
    batch_size, steps, sensors, _ = batched.shape
    block = min(length, steps)
    count = max(1, int(round(sensors * sensor_rate)))
    for b in range(batch_size):
        start = int(gen.integers(0, steps - block + 1))
        sensor_idx = gen.choice(sensors, size=count, replace=False)
        mask[b, start : start + block, sensor_idx, :] = True
    corrupted[mask] = fill_value
    metadata = {
        "fault": "continuous_outage",
        "length": length,
        "sensor_rate": sensor_rate,
        "seed": seed,
        "fill_value": fill_value,
    }
    out, out_mask = _restore_batch(corrupted, mask, was_batched)
    return out, out_mask, metadata


def gaussian_noise(
    x: np.ndarray,
    severity: str | float = "low",
    train_std: float | np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Add Gaussian noise scaled by training-set standard deviation."""

    batched, was_batched = _as_batched(x)
    corrupted = batched.copy()
    scale = _scale_from_severity(severity)
    std = np.nanstd(batched) if train_std is None else train_std
    gen = _rng(seed)
    noise = gen.normal(loc=0.0, scale=scale * std, size=batched.shape)
    mask = np.ones_like(batched, dtype=bool)
    corrupted = corrupted + noise
    metadata = {"fault": "gaussian_noise", "severity": severity, "scale": scale, "seed": seed}
    out, out_mask = _restore_batch(corrupted, mask, was_batched)
    return out, out_mask, metadata


def linear_drift(
    x: np.ndarray,
    severity: str | float = "low",
    train_std: float | np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Add a linear bias drift through time."""

    batched, was_batched = _as_batched(x)
    corrupted = batched.copy()
    scale = _scale_from_severity(severity)
    std = np.nanstd(batched) if train_std is None else train_std
    steps = batched.shape[1]
    ramp = np.linspace(0.0, 1.0, steps, dtype=float).reshape(1, steps, 1, 1)
    drift = ramp * scale * std
    mask = np.ones_like(batched, dtype=bool)
    corrupted = corrupted + drift
    metadata = {"fault": "linear_drift", "severity": severity, "scale": scale, "seed": seed}
    out, out_mask = _restore_batch(corrupted, mask, was_batched)
    return out, out_mask, metadata


def stuck_at_last_value(
    x: np.ndarray,
    severity: str | float = "low",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Replace selected contiguous sensor segments with their previous value."""

    batched, was_batched = _as_batched(x)
    corrupted = batched.copy()
    mask = np.zeros_like(batched, dtype=bool)
    gen = _rng(seed)
    scale = _scale_from_severity(severity)
    batch_size, steps, sensors, _ = batched.shape
    if steps < 2:
        metadata = {"fault": "stuck_at_last_value", "severity": severity, "scale": scale, "seed": seed}
        out, out_mask = _restore_batch(corrupted, mask, was_batched)
        return out, out_mask, metadata
    count = max(1, int(round(sensors * min(scale, 1.0))))
    length = max(1, int(round((steps - 1) * min(scale, 1.0))))
    length = min(length, steps - 1)
    for b in range(batch_size):
        start = int(gen.integers(1, steps - length + 1))
        sensor_idx = gen.choice(sensors, size=count, replace=False)
        corrupted[b, start : start + length, sensor_idx, :] = corrupted[b, start - 1 : start, sensor_idx, :]
        mask[b, start : start + length, sensor_idx, :] = True
    metadata = {"fault": "stuck_at_last_value", "severity": severity, "scale": scale, "seed": seed}
    out, out_mask = _restore_batch(corrupted, mask, was_batched)
    return out, out_mask, metadata

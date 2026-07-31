"""Leakage-free raw-time split utilities for traffic forecasting datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.traffic_dataset import ForecastingSpec


@dataclass(frozen=True)
class RawTimeSplitSpec:
    """Chronological raw-time split configuration."""

    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    stride: int = 1


@dataclass(frozen=True)
class WindowIndex:
    """Raw-time coverage of one forecasting window."""

    input_start: int
    input_end: int
    target_start: int
    target_end: int


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(obj: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def split_raw_time(
    series: np.ndarray,
    split_spec: RawTimeSplitSpec,
) -> dict[str, tuple[np.ndarray, int, int]]:
    """Split raw series chronologically before any window generation."""

    total_steps = int(series.shape[0])
    if total_steps <= 0:
        raise ValueError("Raw series must contain at least one time step.")
    train_end = int(np.floor(total_steps * split_spec.train_ratio))
    val_end = int(np.floor(total_steps * (split_spec.train_ratio + split_spec.val_ratio)))
    if train_end <= 0 or val_end <= train_end or val_end >= total_steps:
        raise ValueError(
            f"Invalid raw-time split for T={total_steps}: train_end={train_end}, val_end={val_end}"
        )
    return {
        "train": (series[:train_end], 0, train_end),
        "val": (series[train_end:val_end], train_end, val_end),
        "test": (series[val_end:], val_end, total_steps),
    }


def make_windows_with_indices(
    series: np.ndarray,
    spec: ForecastingSpec,
    raw_start_index: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[WindowIndex]]:
    """Create windows within one raw-time segment and preserve raw indices."""

    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected raw series [T,N] or [T,N,F], got {arr.shape}")
    total = spec.input_length + spec.horizon
    samples = arr.shape[0] - total + 1
    if samples <= 0:
        raise ValueError(
            f"Split beginning at raw index {raw_start_index} cannot form windows: "
            f"need >= {total} steps, got {arr.shape[0]}"
        )
    starts = list(range(0, samples, max(1, int(stride))))
    x = np.stack([arr[i : i + spec.input_length] for i in starts], axis=0)
    y = np.stack(
        [arr[i + spec.input_length : i + total] for i in starts],
        axis=0,
    )
    indices = [
        WindowIndex(
            input_start=raw_start_index + i,
            input_end=raw_start_index + i + spec.input_length - 1,
            target_start=raw_start_index + i + spec.input_length,
            target_end=raw_start_index + i + total - 1,
        )
        for i in starts
    ]
    return x, y, indices


def normalize_with_raw_train(
    raw_train: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Normalize all splits using only raw train time segment statistics."""

    train_arr = np.asarray(raw_train, dtype=np.float32)
    if train_arr.ndim == 2:
        train_arr = train_arr[..., None]
    mean = np.nanmean(train_arr, axis=(0, 1), keepdims=True)
    std = np.nanstd(train_arr, axis=(0, 1), keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, (x, y) in splits.items():
        normalized[split_name] = ((x - mean) / std, (y - mean) / std)
    stats = {
        "mean": np.squeeze(mean).tolist(),
        "std": np.squeeze(std).tolist(),
        "normalization": "raw_train_mean_std",
        "normalization_scope": "raw_time_train_segment_only",
    }
    return normalized, stats


def time_index_audit(indices: dict[str, list[WindowIndex]]) -> dict[str, Any]:
    """Check that raw time points do not overlap across splits."""

    usage: dict[str, set[int]] = {}
    for split_name, split_indices in indices.items():
        covered: set[int] = set()
        for idx in split_indices:
            covered.update(range(idx.input_start, idx.input_end + 1))
            covered.update(range(idx.target_start, idx.target_end + 1))
        usage[split_name] = covered
    train_val = usage["train"] & usage["val"]
    train_test = usage["train"] & usage["test"]
    val_test = usage["val"] & usage["test"]
    return {
        "covered_time_counts": {name: len(points) for name, points in usage.items()},
        "intersections": {
            "train_val": len(train_val),
            "train_test": len(train_test),
            "val_test": len(val_test),
        },
        "passes": len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0,
    }


def build_processed_metadata(
    dataset_name: str,
    raw_source: str,
    series_shape: tuple[int, ...],
    forecasting_spec: ForecastingSpec,
    split_spec: RawTimeSplitSpec,
    raw_splits: dict[str, tuple[np.ndarray, int, int]],
    windows: dict[str, tuple[np.ndarray, np.ndarray]],
    window_indices: dict[str, list[WindowIndex]],
    normalization: dict[str, Any],
    raw_sha256: str,
    processed_sha256: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble manuscript-facing preprocessing metadata."""

    split_ranges = {
        name: {
            "raw_start_index": int(start),
            "raw_end_exclusive": int(end),
            "raw_step_count": int(segment.shape[0]),
            "window_count": int(windows[name][0].shape[0]),
        }
        for name, (segment, start, end) in raw_splits.items()
    }
    metadata: dict[str, Any] = {
        "dataset": dataset_name,
        "source": raw_source,
        "raw_shape": list(series_shape),
        "forecasting_spec": asdict(forecasting_spec),
        "split_spec": asdict(split_spec),
        "split_ranges": split_ranges,
        "window_index_examples": {
            name: [asdict(idx) for idx in values[:3]]
            for name, values in window_indices.items()
        },
        "normalization": normalization,
        "raw_source_sha256": raw_sha256,
        "processed_split_sha256": processed_sha256 or {},
    }
    if extra:
        metadata.update(extra)
    return metadata

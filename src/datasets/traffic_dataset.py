"""Dataset utilities for full-network multi-step traffic sensor forecasting."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastingSpec:
    """Shape contract for X [L,N,F] -> Y [H,N,F]."""

    input_length: int = 12
    horizon: int = 12
    feature_dim: int = 1


def expected_processed_paths(processed_dir: str | Path, dataset_name: str) -> dict[str, Path]:
    """Return conventional processed artifact paths for a dataset."""

    root = Path(processed_dir) / dataset_name.lower()
    return {
        "root": root,
        "train": root / "train.npz",
        "val": root / "val.npz",
        "test": root / "test.npz",
        "stats": root / "dataset_stats.json",
        "metadata": root / "metadata.json",
        "adjacency": root / "adjacency.npy",
        "adjacency_metadata": root / "adjacency_metadata.json",
    }


def make_windows(series: np.ndarray, spec: ForecastingSpec) -> tuple[np.ndarray, np.ndarray]:
    """Create X [samples,L,N,F] and Y [samples,H,N,F] windows."""

    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected raw series [T,N] or [T,N,F], got {arr.shape}")
    total = spec.input_length + spec.horizon
    samples = arr.shape[0] - total + 1
    if samples <= 0:
        raise ValueError(f"Need at least {total} time steps, got {arr.shape[0]}")
    x = np.stack([arr[i : i + spec.input_length] for i in range(samples)], axis=0)
    y = np.stack(
        [arr[i + spec.input_length : i + total] for i in range(samples)],
        axis=0,
    )
    return x, y


def split_windows(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Chronologically split windows into train/val/test."""

    if x.shape[0] != y.shape[0]:
        raise ValueError("X and Y sample counts differ")
    count = x.shape[0]
    train_end = int(count * train_ratio)
    val_end = train_end + int(count * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= count:
        raise ValueError(f"Invalid split for {count} samples")
    return {
        "train": (x[:train_end], y[:train_end]),
        "val": (x[train_end:val_end], y[train_end:val_end]),
        "test": (x[val_end:], y[val_end:]),
    }


def normalize_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Normalize X and Y with training X statistics only."""

    train_x = splits["train"][0]
    mean = np.nanmean(train_x, axis=(0, 1, 2), keepdims=True)
    std = np.nanstd(train_x, axis=(0, 1, 2), keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, (x, y) in splits.items():
        normalized[split] = ((x - mean) / std, (y - mean) / std)
    stats = {
        "mean": np.squeeze(mean).tolist(),
        "std": np.squeeze(std).tolist(),
        "normalization": "train_x_mean_std",
    }
    return normalized, stats


def save_processed_dataset(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    stats: dict[str, Any],
    metadata: dict[str, Any],
    processed_dir: str | Path,
    dataset_name: str,
    adjacency: np.ndarray | None = None,
    adjacency_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save processed arrays and metadata."""

    paths = expected_processed_paths(processed_dir, dataset_name)
    root = paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for split, (x, y) in splits.items():
        path = paths[split]
        np.savez_compressed(path, x=x, y=y)
        written[split] = str(path)
    paths["stats"].write_text(json.dumps(stats, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written["stats"] = str(paths["stats"])
    written["metadata"] = str(paths["metadata"])
    if adjacency is not None:
        np.save(paths["adjacency"], adjacency.astype(np.float32))
        written["adjacency"] = str(paths["adjacency"])
    if adjacency_metadata is not None:
        paths["adjacency_metadata"].write_text(json.dumps(adjacency_metadata, indent=2), encoding="utf-8")
        written["adjacency_metadata"] = str(paths["adjacency_metadata"])
    return written


def synthetic_series(steps: int, sensors: int, features: int, seed: int) -> np.ndarray:
    """Generate synthetic data for smoke testing only."""

    rng = np.random.default_rng(seed)
    time = np.arange(steps, dtype=np.float32)[:, None, None]
    sensor_offsets = np.linspace(0.0, 1.0, sensors, dtype=np.float32)[None, :, None]
    seasonal = np.sin(time / 12.0) + np.cos(time / 24.0)
    noise = rng.normal(0.0, 0.05, size=(steps, sensors, features)).astype(np.float32)
    return seasonal + sensor_offsets + noise


def load_h5_series(path: str | Path) -> np.ndarray:
    """Load a traffic dataset from an HDF5 file using pandas/PyTables."""

    import pandas as pd

    frame = pd.read_hdf(path)
    values = frame.to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected HDF5 table values [T,N], got {values.shape}")
    return values[..., None]


def load_csv_series(path: str | Path) -> np.ndarray:
    """Load a traffic dataset from a CSV file with timestamps in column 0."""

    import pandas as pd

    frame = pd.read_csv(path, index_col=0)
    values = frame.to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected CSV values [T,N], got {values.shape}")
    return values[..., None]


def load_adjacency_pickle(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load METR-LA adjacency pickle as adjacency plus metadata."""

    with Path(path).open("rb") as handle:
        obj = pickle.load(handle, encoding="latin1")
    metadata: dict[str, Any] = {"source": str(path)}
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        sensor_ids, sensor_id_to_ind, adjacency = obj[0], obj[1], obj[2]
        metadata.update(
            {
                "format": "sensor_ids_id_map_adjacency",
                "sensor_count": len(sensor_ids),
                "sensor_ids": list(sensor_ids),
                "id_map_size": len(sensor_id_to_ind),
            }
        )
    else:
        adjacency = obj
        metadata.update({"format": "adjacency_only"})
    arr = np.asarray(adjacency, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got {arr.shape}")
    metadata["adjacency_shape"] = list(arr.shape)
    metadata["nonzero_edges"] = int(np.count_nonzero(arr))
    return arr, metadata


def write_missing_data_status(
    processed_dir: str | Path,
    dataset_name: str,
    raw_path: str | Path,
    spec: ForecastingSpec,
) -> Path:
    """Record that real data is missing without fabricating outputs."""

    paths = expected_processed_paths(processed_dir, dataset_name)
    paths["root"].mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset_name,
        "status": "raw_file_missing",
        "raw_path": str(raw_path),
        "spec": asdict(spec),
        "note": "No arrays were generated because the real raw dataset file is absent.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths["metadata"]

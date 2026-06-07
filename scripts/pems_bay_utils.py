"""Utilities for PEMS-BAY import and audit scripts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPEED_NAMES = [
    "PEMS-BAY.csv",
    "pems-bay.csv",
    "pems_bay.csv",
    "PEMS_BAY.csv",
    "pems-bay.h5",
    "pems_bay.h5",
    "PEMS-BAY.h5",
    "PEMS_BAY.h5",
]
ADJ_NAMES = [
    "adj_mx_PEMS-BAY.pkl",
    "adj_mx_pems_bay.pkl",
    "adj_mx_bay.pkl",
    "adj_mx_PEMS_BAY.pkl",
    "adj_mx_pems-bay.pkl",
]


def candidate_dirs(search_root: str | Path) -> list[Path]:
    root = Path(search_root)
    raw_root = root / "raw" if root.name.lower() != "raw" else root
    return [
        raw_root / "pems-bay",
        raw_root / "PEMS-BAY",
        raw_root / "pemsbay",
        raw_root,
        root,
    ]


def discover_file(search_root: str | Path, names: list[str], kind: str) -> Path | None:
    dirs = candidate_dirs(search_root)
    for directory in dirs:
        for name in names:
            path = directory / name
            if path.exists() and path.is_file():
                return path
    suffixes = {".csv", ".h5", ".hdf5"} if kind == "speed" else {".pkl"}
    keywords = ["pems", "bay"] if kind == "speed" else ["adj", "pems", "bay"]
    matches: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            lower = path.name.lower()
            if all(k in lower for k in keywords) or (kind == "adj" and lower == "adj_mx_bay.pkl"):
                matches.append(path)
    return sorted(set(matches), key=lambda p: (len(str(p)), str(p).lower()))[0] if matches else None


def load_speed_frame(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        timestamp_status = "absent"
        timestamp_values = None
        if frame.shape[1] > 1:
            first = frame.iloc[:, 0]
            parsed = pd.to_datetime(first, errors="coerce")
            numeric_first = pd.to_numeric(first, errors="coerce")
            if parsed.notna().mean() > 0.95 and numeric_first.notna().mean() < 0.95:
                timestamp_status = "real"
                timestamp_values = parsed.astype("datetime64[ns]")
                frame = frame.iloc[:, 1:]
            elif not np.issubdtype(frame.dtypes.iloc[0], np.number):
                timestamp_status = "label_index"
                timestamp_values = first.astype(str)
                frame = frame.iloc[:, 1:]
        frame = frame.apply(pd.to_numeric, errors="coerce")
        meta = {
            "format": "csv",
            "timestamp_status": timestamp_status,
            "timestamps": timestamp_values,
            "columns": [str(c) for c in frame.columns],
        }
        return frame, meta
    if suffix in {".h5", ".hdf5"}:
        hdf = pd.HDFStore(path, mode="r")
        try:
            keys = hdf.keys()
            if not keys:
                raise ValueError(f"No HDF5 keys in {path}")
            key = keys[0]
        finally:
            hdf.close()
        frame = pd.read_hdf(path, key=key)
        timestamp_status = "real" if isinstance(frame.index, pd.DatetimeIndex) else "absent"
        meta = {
            "format": "h5",
            "hdf_key": key,
            "timestamp_status": timestamp_status,
            "timestamps": frame.index if timestamp_status == "real" else None,
            "columns": [str(c) for c in frame.columns],
        }
        frame = frame.apply(pd.to_numeric, errors="coerce")
        return frame, meta
    raise ValueError(f"Unsupported speed file extension: {path.suffix}")


def frame_stats(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    finite_values = values[finite]
    total = int(values.size)
    return {
        "shape": list(values.shape),
        "time_steps": int(values.shape[0]),
        "sensor_count": int(values.shape[1]) if values.ndim == 2 else None,
        "nan_ratio": float(np.isnan(values).sum() / total) if total else None,
        "inf_ratio": float(np.isinf(values).sum() / total) if total else None,
        "zero_ratio": float((values == 0).sum() / total) if total else None,
        "min": float(np.min(finite_values)) if finite_values.size else None,
        "max": float(np.max(finite_values)) if finite_values.size else None,
        "mean": float(np.mean(finite_values)) if finite_values.size else None,
        "std": float(np.std(finite_values)) if finite_values.size else None,
    }


def load_adjacency(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    with path.open("rb") as handle:
        obj = pickle.load(handle, encoding="latin1")
    metadata: dict[str, Any] = {"source": str(path)}
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        sensor_ids, sensor_id_to_ind, adjacency = obj[0], obj[1], obj[2]
        metadata.update(
            {
                "format": "sensor_ids_id_map_adjacency",
                "sensor_count": len(sensor_ids),
                "sensor_ids": [str(x) for x in sensor_ids],
                "id_map_size": len(sensor_id_to_ind),
            }
        )
    else:
        adjacency = obj
        metadata["format"] = "adjacency_only"
    arr = np.asarray(adjacency, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got {arr.shape}")
    metadata.update(
        {
            "adjacency_shape": list(arr.shape),
            "nonzero_edges": int(np.count_nonzero(arr)),
            "is_symmetric": bool(np.allclose(arr, arr.T, equal_nan=True)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
        }
    )
    return arr, metadata


def write_blocker(processed_dir: str | Path, output_dir: str | Path, reason: str, details: dict[str, Any]) -> None:
    processed = Path(processed_dir)
    output = Path(output_dir)
    processed.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    body = [
        "# PEMS-BAY Data Blocker Report",
        "",
        f"- Reason: {reason}",
        f"- Details: `{json.dumps(details, ensure_ascii=False)}`",
        "- No synthetic data was generated.",
        "- No METR-LA processed artifacts were modified.",
    ]
    text = "\n".join(body) + "\n"
    (processed / "DATA_BLOCKER_REPORT.md").write_text(text, encoding="utf-8")
    (output / "DATA_BLOCKER_REPORT.md").write_text(text, encoding="utf-8")


def load_npz_pair(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["x"].astype(np.float32), data["y"].astype(np.float32)


def inverse_transform(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return arr * std + mean


def safe_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = np.maximum(np.abs(true), 1.0)
    mape = float(np.mean(np.abs(diff) / denom) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def add_identity_features(x: np.ndarray, start_index: int = 0) -> np.ndarray:
    samples, length, sensors, _ = x.shape
    out = np.zeros((samples, length, sensors, 3), dtype=np.float32)
    out[..., 0:1] = x[..., 0:1]
    sample_idx = np.arange(samples, dtype=np.int64)[:, None]
    step_idx = np.arange(length, dtype=np.int64)[None, :]
    time_idx = start_index + sample_idx + step_idx
    tod = (time_idx % 288).astype(np.float32) / 288.0
    dow = ((time_idx // 288) % 7).astype(np.float32) / 7.0
    out[..., 1] = tod[:, :, None]
    out[..., 2] = dow[:, :, None]
    return out

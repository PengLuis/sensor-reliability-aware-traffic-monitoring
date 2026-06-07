"""Audit processed PEMS-BAY forecasting artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pems_bay_utils import inverse_transform, load_npz_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/pems-bay")
    parser.add_argument("--output-dir", default="experiments/pems-bay-data-import")
    return parser.parse_args()


def finite(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all())


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}
    required = {name: data_dir / f"{name}.npz" for name in ["train", "val", "test"]}
    checks["split_files_exist"] = all(path.exists() for path in required.values())
    stats_path = data_dir / "dataset_stats.json"
    meta_path = data_dir / "metadata.json"
    adj_path = data_dir / "adjacency.npy"
    time_path = data_dir / "time_metadata.json"
    checks["dataset_stats_exists"] = stats_path.exists()
    checks["metadata_exists"] = meta_path.exists()
    checks["adjacency_exists"] = adj_path.exists()
    checks["time_metadata_exists"] = time_path.exists()
    if not checks["split_files_exist"]:
        raise SystemExit("Missing processed split files.")

    arrays = {split: load_npz_pair(path) for split, path in required.items()}
    shapes = {split: {"x": list(pair[0].shape), "y": list(pair[1].shape)} for split, pair in arrays.items()}
    train_x, train_y = arrays["train"]
    n = train_x.shape[2]
    checks["x_y_shapes_correct"] = all(x.ndim == 4 and y.ndim == 4 and x.shape[1:] == (12, n, 1) and y.shape[1:] == (12, n, 1) for x, y in arrays.values())
    checks["input_length_12"] = all(pair[0].shape[1] == 12 for pair in arrays.values())
    checks["horizon_12"] = all(pair[1].shape[1] == 12 for pair in arrays.values())
    checks["sensor_count_consistent"] = all(pair[0].shape[2] == n and pair[1].shape[2] == n for pair in arrays.values())
    checks["split_counts_valid"] = train_x.shape[0] > 0 and arrays["val"][0].shape[0] > 0 and arrays["test"][0].shape[0] > 0
    checks["all_arrays_finite"] = all(finite(x) and finite(y) for x, y in arrays.values())
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    checks["scaler_train_x_only"] = stats.get("normalization") == "train_x_mean_std" and stats.get("scaler_source") == "train_x_only"
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    inv_train = inverse_transform(train_x[: min(512, train_x.shape[0])], mean, std)
    checks["inverse_transform_plausible"] = bool(np.isfinite(inv_train).all() and -10.0 <= float(inv_train.min()) <= float(inv_train.max()) <= 150.0)
    if adj_path.exists():
        adj = np.load(adj_path)
        checks["adjacency_shape_n_n"] = list(adj.shape) == [n, n]
        checks["adjacency_n_matches_data_n"] = adj.shape[0] == n
        checks["adjacency_finite"] = bool(np.isfinite(adj).all())
    else:
        checks["adjacency_shape_n_n"] = False
        checks["adjacency_n_matches_data_n"] = False
        checks["adjacency_finite"] = False
    time_meta = json.loads(time_path.read_text(encoding="utf-8")) if time_path.exists() else {}
    checks["timestamp_status_documented"] = bool(time_meta.get("timestamp_status"))
    checks["target_y_not_corrupted"] = meta.get("target_y_corrupted") is False
    checks["synthetic_data_not_generated"] = meta.get("synthetic_data_generated") is False
    status = "PASS" if all(bool(v) for v in checks.values()) else "FAIL"
    audit = {
        "status": status,
        "checks": checks,
        "shapes": shapes,
        "sensor_count": int(n),
        "split_counts": {split: int(pair[0].shape[0]) for split, pair in arrays.items()},
        "inverse_transform_sample_range": {"min": float(inv_train.min()), "max": float(inv_train.max())},
        "metadata_timestamp_status": time_meta.get("timestamp_status", "MISSING"),
    }
    (out_dir / "processed_dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Processed Dataset Audit Summary",
        "",
        f"- Status: `{status}`",
        f"- Shapes: `{json.dumps(shapes)}`",
        f"- Sensor count: `{n}`",
        f"- Split counts: `{audit['split_counts']}`",
        f"- Scaler train-only: `{checks['scaler_train_x_only']}`",
        f"- Inverse transform plausible: `{checks['inverse_transform_plausible']}`",
        f"- Adjacency shape matches: `{checks['adjacency_n_matches_data_n']}`",
        f"- Timestamp status: `{audit['metadata_timestamp_status']}`",
        f"- Target Y not corrupted: `{checks['target_y_not_corrupted']}`",
    ]
    (out_dir / "processed_dataset_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

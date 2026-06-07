"""Preprocess real PEMS-BAY data into the project forecasting format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pems_bay_utils import (  # noqa: E402
    ADJ_NAMES,
    SPEED_NAMES,
    discover_file,
    frame_stats,
    load_adjacency,
    load_speed_frame,
    write_blocker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--dataset-name", default="pems-bay")
    parser.add_argument("--input-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output-dir", default="experiments/pems-bay-data-import")
    return parser.parse_args()


def make_windows(series: np.ndarray, input_length: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    total = input_length + horizon
    count = series.shape[0] - total + 1
    if count <= 0:
        raise ValueError(f"Need at least {total} time steps, got {series.shape[0]}")
    x = np.stack([series[i : i + input_length] for i in range(count)], axis=0)
    y = np.stack([series[i + input_length : i + total] for i in range(count)], axis=0)
    return x.astype(np.float32), y.astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_root = Path(args.processed_dir) / args.dataset_name
    processed_root.mkdir(parents=True, exist_ok=True)

    speed_path = discover_file(args.raw_dir, SPEED_NAMES, "speed")
    adj_path = discover_file(args.raw_dir, ADJ_NAMES, "adj")
    if speed_path is None:
        write_blocker(processed_root, out_dir, "raw PEMS-BAY speed file missing", {"raw_dir": args.raw_dir})
        raise SystemExit(2)
    if adj_path is None:
        write_blocker(processed_root, out_dir, "raw PEMS-BAY adjacency file missing", {"raw_dir": args.raw_dir})
        raise SystemExit(2)

    frame, speed_meta = load_speed_frame(speed_path)
    raw_stats = frame_stats(frame)
    filled_frame = frame.copy()
    imputation = "none"
    if not np.isfinite(filled_frame.to_numpy(dtype=np.float64)).all():
        filled_frame = filled_frame.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        imputation = "ffill_bfill_for_nonfinite_raw_values"
    if not np.isfinite(filled_frame.to_numpy(dtype=np.float64)).all():
        write_blocker(processed_root, out_dir, "raw PEMS-BAY contains unresolved NaN/Inf after fill", {"speed_file": str(speed_path)})
        raise SystemExit(2)

    series = filled_frame.to_numpy(dtype=np.float32)[..., None]
    x, y = make_windows(series, args.input_length, args.horizon)
    count = x.shape[0]
    train_end = int(count * args.train_ratio)
    val_end = train_end + int(count * args.val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= count:
        write_blocker(processed_root, out_dir, "invalid chronological split", {"sample_count": count})
        raise SystemExit(2)
    splits = {
        "train": (x[:train_end], y[:train_end]),
        "val": (x[train_end:val_end], y[train_end:val_end]),
        "test": (x[val_end:], y[val_end:]),
    }
    train_x = splits["train"][0]
    mean = float(np.mean(train_x))
    std = float(np.std(train_x))
    if std < 1.0e-6:
        std = 1.0
    stats = {"mean": mean, "std": std, "normalization": "train_x_mean_std", "scaler_source": "train_x_only"}

    for split, (sx, sy) in splits.items():
        np.savez_compressed(processed_root / f"{split}.npz", x=((sx - mean) / std).astype(np.float32), y=((sy - mean) / std).astype(np.float32))

    adj, adj_meta = load_adjacency(adj_path)
    if adj.shape != (series.shape[1], series.shape[1]):
        write_blocker(
            processed_root,
            out_dir,
            "adjacency shape mismatches speed sensor count",
            {"adjacency_shape": list(adj.shape), "sensor_count": int(series.shape[1])},
        )
        raise SystemExit(2)
    np.save(processed_root / "adjacency.npy", adj.astype(np.float32))

    timestamp_status = speed_meta["timestamp_status"]
    time_meta = {
        "timestamp_status": timestamp_status if timestamp_status == "real" else "inferred_contiguous_5_minute_indices",
        "real_timestamp_metadata_exists": bool(timestamp_status == "real"),
        "tod_dow_source": "real_timestamp_index" if timestamp_status == "real" else "inferred_from_contiguous_5_minute_indices",
        "sampling_minutes": 5,
        "split_start_indices": {"train": 0, "val": train_end, "test": val_end},
        "note": "Day-of-week is inferred unless real timestamp metadata exists.",
    }
    if timestamp_status == "real" and speed_meta.get("timestamps") is not None:
        ts = speed_meta["timestamps"]
        time_meta["start_timestamp"] = str(ts.iloc[0] if hasattr(ts, "iloc") else ts[0])
        time_meta["end_timestamp"] = str(ts.iloc[-1] if hasattr(ts, "iloc") else ts[-1])

    metadata = {
        "dataset": "PEMS-BAY",
        "status": "processed",
        "source": str(speed_path),
        "adjacency_source": str(adj_path),
        "spec": {"input_length": args.input_length, "horizon": args.horizon, "feature_dim": 1},
        "raw_shape": list(series.shape),
        "raw_stats_before_imputation": raw_stats,
        "imputation": imputation,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "splits": {name: {"x_shape": list(pair[0].shape), "y_shape": list(pair[1].shape)} for name, pair in splits.items()},
        "normalization": stats,
        "target_y_corrupted": False,
        "synthetic_data_generated": False,
    }
    (processed_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (processed_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (processed_root / "adjacency_metadata.json").write_text(json.dumps(adj_meta, indent=2), encoding="utf-8")
    (processed_root / "time_metadata.json").write_text(json.dumps(time_meta, indent=2), encoding="utf-8")
    report = [
        "# PEMS-BAY Preprocessing Report",
        "",
        f"- Speed source: `{speed_path}`",
        f"- Adjacency source: `{adj_path}`",
        f"- Raw shape: `{list(series.shape)}`",
        f"- X shape: `{list(x.shape)}`",
        f"- Y shape: `{list(y.shape)}`",
        f"- Split counts: train `{train_end}`, val `{val_end - train_end}`, test `{count - val_end}`",
        f"- Normalization: train X only, mean `{mean}`, std `{std}`",
        f"- Timestamp status: `{time_meta['timestamp_status']}`",
        "- Target Y was not corrupted.",
        "- No synthetic data was generated.",
    ]
    (processed_root / "preprocessing_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {"status": "PASS", "processed_dir": str(processed_root), "metadata": metadata}
    (out_dir / "preprocessing_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

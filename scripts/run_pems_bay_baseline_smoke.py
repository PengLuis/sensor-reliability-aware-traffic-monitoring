"""Run PEMS-BAY persistence smoke and OfficialStyleSTID shape sanity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pems_bay_utils import add_identity_features, inverse_transform, load_npz_pair, safe_metrics  # noqa: E402
from src.models.strong_backbones import OfficialStyleSTID  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/pems-bay")
    parser.add_argument("--output-dir", default="experiments/pems-bay-data-import")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    mean = float(stats["mean"])
    std = float(stats["std"])
    test_x, test_y = load_npz_pair(data_dir / "test.npz")
    pred_norm = np.repeat(test_x[:, -1:, :, :], test_y.shape[1], axis=1)
    pred = inverse_transform(pred_norm, mean, std)
    true = inverse_transform(test_y, mean, std)
    metrics = safe_metrics(pred, true)
    row = {
        "model": "Persistence",
        "fault": "clean",
        "metrics_scale": "original",
        **metrics,
        "h3_mae": float(np.mean(np.abs(pred[:, 2] - true[:, 2]))),
        "h6_mae": float(np.mean(np.abs(pred[:, 5] - true[:, 5]))),
        "h12_mae": float(np.mean(np.abs(pred[:, 11] - true[:, 11]))),
        "y_pred_shape": list(pred.shape),
        "y_true_shape": list(true.shape),
    }
    with (out_dir / "persistence_smoke_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    device = resolve_device(args.device)
    n = int(test_x.shape[2])
    model = OfficialStyleSTID(sensors=n, input_length=12, input_dim=3, horizon=12).to(device)
    model.train()
    batch = add_identity_features(test_x[: min(4, test_x.shape[0])], 0)
    xb = torch.from_numpy(batch).to(device)
    target = torch.randn((xb.shape[0], 12, n, 1), device=device)
    pred_t = model(xb)
    loss = torch.mean(torch.abs(pred_t - target))
    loss.backward()
    grad_ok = model.time_series_emb_layer.weight.grad is not None and model.regression_layer.weight.grad is not None
    sanity = {
        "status": "PASS" if list(pred_t.shape) == [xb.shape[0], 12, n, 1] and grad_ok and torch.isfinite(pred_t).all().item() else "FAIL",
        "device": str(device),
        "sensor_count": n,
        "input_shape": list(xb.shape),
        "output_shape": list(pred_t.shape),
        "expected_output_shape": [xb.shape[0], 12, n, 1],
        "gradients_exist_for_core_layers": bool(grad_ok),
        "finite_output": bool(torch.isfinite(pred_t).all().item()),
    }
    (out_dir / "pems_bay_model_shape_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    summary = [
        "# PEMS-BAY Baseline Smoke Summary",
        "",
        f"- Persistence clean MAE: `{row['mae']}`",
        f"- Persistence h3/h6/h12 MAE: `{row['h3_mae']}` / `{row['h6_mae']}` / `{row['h12_mae']}`",
        f"- Prediction shape: `{row['y_pred_shape']}`",
        f"- Target shape: `{row['y_true_shape']}`",
        f"- OfficialStyleSTID shape sanity: `{sanity['status']}`",
        f"- Device: `{device}`",
        "- No deep model training was run.",
    ]
    (out_dir / "baseline_smoke_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    result = {"persistence": row, "shape_sanity": sanity}
    print(json.dumps(result, indent=2))
    if sanity["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

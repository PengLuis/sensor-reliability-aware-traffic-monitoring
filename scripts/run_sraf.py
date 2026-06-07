"""Run SRAF-GRU or SRAF-TCN."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.regression import regression_metrics  # noqa: E402
from src.models.sraf import SRAFModel  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["GRU", "TCN"], default="GRU")
    parser.add_argument("--dataset", default="synthetic_smoke")
    parser.add_argument("--data-dir", default="data/processed/synthetic_smoke")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-reliability-gating", action="store_true")
    parser.add_argument("--no-mask-encoding", action="store_true")
    parser.add_argument("--no-temporal-repair", action="store_true")
    parser.add_argument("--no-spatial-repair", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for debug smoke.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for debug smoke.")
    return parser


def _load_split(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(data_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def _adjacency(sensors: int, data_dir: str | Path) -> torch.Tensor:
    path = Path(data_dir) / "adjacency.npy"
    if path.exists():
        arr = np.load(path).astype(np.float32)
        if arr.shape != (sensors, sensors):
            raise ValueError(f"Adjacency shape {arr.shape} does not match sensors {sensors}")
        return torch.from_numpy(arr)
    matrix = torch.zeros((sensors, sensors), dtype=torch.float32)
    for i in range(sensors):
        if i > 0:
            matrix[i, i - 1] = 1.0
        if i < sensors - 1:
            matrix[i, i + 1] = 1.0
    return matrix


def _write_metrics(path: Path, row: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    train_x, train_y = _load_split(args.data_dir, "train")
    test_x, test_y = _load_split(args.data_dir, "test")
    if args.max_train_samples is not None:
        train_x = train_x[: args.max_train_samples]
        train_y = train_y[: args.max_train_samples]
    if args.max_test_samples is not None:
        test_x = test_x[: args.max_test_samples]
        test_y = test_y[: args.max_test_samples]
    _, input_length, sensors, features = train_x.shape
    horizon = train_y.shape[1]
    model = SRAFModel(
        sensors=sensors,
        features=features,
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        backbone=args.backbone,
        alpha=args.alpha,
        no_reliability_gating=args.no_reliability_gating,
        no_mask_encoding=args.no_mask_encoding,
        no_temporal_repair=args.no_temporal_repair,
        no_spatial_repair=args.no_spatial_repair,
    )
    adjacency = _adjacency(sensors, args.data_dir)
    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y)
    test_x_t = torch.from_numpy(test_x)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_fn = nn.MSELoss()
    logs: list[str] = []
    start = perf_counter()
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(train_x_t, adjacency=adjacency)
        loss = loss_fn(pred, train_y_t)
        loss.backward()
        optimizer.step()
        logs.append(f"epoch={epoch + 1},loss={float(loss.detach().cpu()):.6f}")
    training_time = perf_counter() - start
    model.eval()
    infer_start = perf_counter()
    with torch.no_grad():
        pred = model(test_x_t, adjacency=adjacency).cpu().numpy()
    inference_time = perf_counter() - infer_start
    if pred.shape != test_y.shape:
        raise RuntimeError(f"SRAF output shape {pred.shape} does not match target {test_y.shape}")
    metrics = regression_metrics(test_y, pred)
    row = {
        "dataset": args.dataset,
        "model": f"SRAF-{args.backbone}",
        **metrics,
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
    }
    suffixes = []
    if args.no_reliability_gating:
        suffixes.append("no_gating")
    if args.no_mask_encoding:
        suffixes.append("no_mask")
    if args.no_temporal_repair:
        suffixes.append("no_temporal")
    if args.no_spatial_repair:
        suffixes.append("no_spatial")
    run_name = f"sraf_{args.backbone.lower()}" + (("_" + "_".join(suffixes)) if suffixes else "")
    run_dir = Path(args.output_dir) / args.dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(run_dir / "metrics.csv", row)
    np.savez_compressed(run_dir / "predictions.npz", y_pred=pred, y_true=test_y)
    (run_dir / "config_resolved.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run_dir / "train_log.txt").write_text("\n".join(logs), encoding="utf-8")
    return {"status": "completed", "run_dir": str(run_dir), "metrics": row, "output_shape": list(pred.shape)}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

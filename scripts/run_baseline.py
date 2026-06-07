"""Run mandatory baseline models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.regression import regression_metrics  # noqa: E402
from src.models.baselines import historical_average_predict, persistence_predict  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["HistoricalAverage", "Persistence", "GRU", "TCN"], required=True)
    parser.add_argument("--dataset", default="synthetic_smoke")
    parser.add_argument("--data-dir", default="data/processed/synthetic_smoke")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1, help="Neural smoke training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples for debug smoke.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test samples for debug smoke.")
    return parser


def _load_split(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(data_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def _run_neural(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float], list[str]]:
    import torch
    from torch import nn

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
    flat_dim = sensors * features
    train_x_t = torch.from_numpy(train_x.reshape(train_x.shape[0], input_length, flat_dim))
    train_y_t = torch.from_numpy(train_y.reshape(train_y.shape[0], horizon * flat_dim))
    test_x_t = torch.from_numpy(test_x.reshape(test_x.shape[0], input_length, flat_dim))

    if args.model == "GRU":
        model = _GRUBaseline(flat_dim, args.hidden_dim, horizon * flat_dim)
    else:
        model = _TCNBaseline(flat_dim, args.hidden_dim, horizon * flat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_fn = nn.MSELoss()
    logs: list[str] = []
    start = perf_counter()
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(train_x_t)
        loss = loss_fn(pred, train_y_t)
        loss.backward()
        optimizer.step()
        logs.append(f"epoch={epoch + 1},loss={float(loss.detach().cpu()):.6f}")
    train_time = perf_counter() - start
    model.eval()
    infer_start = perf_counter()
    with torch.no_grad():
        pred = model(test_x_t).cpu().numpy().reshape(test_x.shape[0], horizon, sensors, features)
    inference_time = perf_counter() - infer_start
    extra = {
        "training_time_sec": train_time,
        "inference_time_sec": inference_time,
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
    }
    return pred, extra, logs


class _GRUBaseline:  # thin wrapper assigned below after torch import
    pass


class _TCNBaseline:
    pass


def _install_torch_models() -> None:
    import torch
    from torch import nn

    class GRUBaseline(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, hidden = self.gru(x)
            return self.head(hidden[-1])

    class TCNBaseline(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.net(x.transpose(1, 2))[..., : x.shape[1]]
            pooled = features.mean(dim=-1)
            return self.head(pooled)

    globals()["_GRUBaseline"] = GRUBaseline
    globals()["_TCNBaseline"] = TCNBaseline


def _write_metrics(path: Path, row: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    test_x, test_y = _load_split(args.data_dir, "test")
    if args.max_test_samples is not None:
        test_x = test_x[: args.max_test_samples]
        test_y = test_y[: args.max_test_samples]
    horizon = test_y.shape[1]
    logs: list[str] = []
    extra: dict[str, float] = {}
    if args.model == "HistoricalAverage":
        pred = historical_average_predict(test_x, horizon)
    elif args.model == "Persistence":
        pred = persistence_predict(test_x, horizon)
    else:
        _install_torch_models()
        pred, extra, logs = _run_neural(args)

    metrics = regression_metrics(test_y, pred)
    metrics.update(extra)
    row = {"dataset": args.dataset, "model": args.model, **metrics}
    run_dir = Path(args.output_dir) / args.dataset / args.model.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(run_dir / "metrics.csv", row)
    np.savez_compressed(run_dir / "predictions.npz", y_pred=pred, y_true=test_y)
    (run_dir / "config_resolved.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run_dir / "train_log.txt").write_text("\n".join(logs) if logs else "no training required\n", encoding="utf-8")
    return {"status": "completed", "run_dir": str(run_dir), "metrics": row}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

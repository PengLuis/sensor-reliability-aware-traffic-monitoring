"""Tiny-overfit check for OfficialStyleSTID on METR-LA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_strong_clean_backbone_integration import (  # noqa: E402
    add_stid_identity_features,
    iter_batches,
    load_split,
    resolve_device,
)
from src.models.strong_backbones import OfficialStyleSTID  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-official-style-stid-code-repair")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def eval_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size=batch_size, shuffle=False, seed=0, epoch=0):
            xb_t = torch.from_numpy(xb.astype(np.float32)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            pred = model(xb_t)
            losses.append(float(torch.mean(torch.abs(pred - yb_t)).detach().cpu()))
    return float(np.mean(losses))


def main() -> None:
    args = build_parser().parse_args()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_x_base, train_y = load_split(data_dir, "train")
    val_x_base, val_y = load_split(data_dir, "val")
    train_x_base, train_y = train_x_base[:512], train_y[:512]
    val_x_base, val_y = val_x_base[:128], val_y[:128]
    train_x = add_stid_identity_features(train_x_base, 0)
    val_x = add_stid_identity_features(val_x_base, 23974)

    model = OfficialStyleSTID(sensors=train_x.shape[2], input_length=train_x.shape[1], input_dim=3, horizon=train_y.shape[1])
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            xb_t = torch.from_numpy(xb.astype(np.float32)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            pred = model(xb_t)
            loss = torch.mean(torch.abs(pred - yb_t))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        val_loss = eval_loss(model, val_x, val_y, args.batch_size, device)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(f"tiny-overfit epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            break

    first_loss = float(rows[0]["train_loss"]) if rows else math.inf
    final_loss = float(rows[-1]["train_loss"]) if rows else math.inf
    best_loss = min(float(r["train_loss"]) for r in rows) if rows else math.inf
    decrease_ratio = (first_loss - best_loss) / first_loss if first_loss and math.isfinite(first_loss) else math.nan
    status = "PASS" if math.isfinite(best_loss) and best_loss < first_loss * 0.90 else "FAIL"

    write_csv(out_dir / "official_style_stid_tiny_overfit_metrics.csv", rows)
    summary = [
        "# OfficialStyleSTID Tiny Overfit",
        "",
        f"- Status: `{status}`",
        f"- Device: `{device}`",
        f"- Train samples: `512`",
        f"- Validation samples: `128`",
        f"- First train loss: `{first_loss:.6f}`",
        f"- Final train loss: `{final_loss:.6f}`",
        f"- Best train loss: `{best_loss:.6f}`",
        f"- Relative best-loss decrease: `{decrease_ratio:.6f}`",
        f"- Runtime seconds: `{perf_counter() - start:.3f}`",
    ]
    (out_dir / "official_style_stid_tiny_overfit_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"status": status, "first_train_loss": first_loss, "best_train_loss": best_loss}, indent=2), flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

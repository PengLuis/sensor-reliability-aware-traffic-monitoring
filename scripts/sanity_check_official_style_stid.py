"""Forward and gradient sanity check for OfficialStyleSTID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.strong_backbones import OfficialStyleSTID  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/metr-la-official-style-stid-code-repair")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def grad_exists(params: list[torch.nn.Parameter]) -> bool:
    return any(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0.0 for p in params)


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = OfficialStyleSTID(
        sensors=207,
        input_length=12,
        input_dim=3,
        horizon=12,
    )
    x = torch.randn(4, 12, 207, 3)
    offsets = torch.arange(12).view(1, 12, 1).expand(4, -1, 207)
    x[..., 1] = (offsets % 288).to(torch.float32) / 288.0
    x[..., 2] = ((offsets // 288) % 7).to(torch.float32) / 7.0
    target = torch.randn(4, 12, 207, 1)

    pred = model(x)
    shape_ok = tuple(pred.shape) == (4, 12, 207, 1)
    loss = torch.mean(torch.abs(pred - target))
    loss.backward()

    encoder_params: list[torch.nn.Parameter] = []
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.Conv2d):
            encoder_params.extend(list(module.parameters()))

    checks = {
        "time_series_emb_layer": grad_exists(list(model.time_series_emb_layer.parameters())),
        "node_emb": model.node_emb is not None and grad_exists(list(model.node_emb.parameters())),
        "time_in_day_emb": model.time_in_day_emb is not None and grad_exists(list(model.time_in_day_emb.parameters())),
        "day_in_week_emb": model.day_in_week_emb is not None and grad_exists(list(model.day_in_week_emb.parameters())),
        "encoder_conv2d_layers": grad_exists(encoder_params),
        "regression_layer": grad_exists(list(model.regression_layer.parameters())),
    }
    all_gradients_ok = all(checks.values())
    result = {
        "status": "PASS" if shape_ok and all_gradients_ok and torch.isfinite(pred).all().item() else "FAIL",
        "seed": args.seed,
        "input_shape": [4, 12, 207, 3],
        "output_shape": list(pred.shape),
        "expected_output_shape": [4, 12, 207, 1],
        "shape_ok": shape_ok,
        "loss": float(loss.detach().cpu()),
        "prediction_finite": bool(torch.isfinite(pred).all().item()),
        "tod_norm_min": float(x[..., 1].min()),
        "tod_norm_max": float(x[..., 1].max()),
        "dow_norm_min": float(x[..., 2].min()),
        "dow_norm_max": float(x[..., 2].max()),
        "gradient_checks": checks,
    }
    (out_dir / "official_style_stid_forward_sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = [
        "# OfficialStyleSTID Forward Sanity",
        "",
        f"- Status: `{result['status']}`",
        f"- Output shape: `{result['output_shape']}`",
        f"- Loss: `{result['loss']:.6f}`",
        f"- Gradient checks: `{checks}`",
    ]
    (out_dir / "official_style_stid_forward_sanity_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

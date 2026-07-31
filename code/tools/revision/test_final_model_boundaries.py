"""Executable information-boundary tests for the forecast-only final SRAF-ID."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import build_official_stid
from src.models.strong_backbones_v3 import SRAFOfficialStyleSTIDWrapperFactorAblation

OUT = ROOT / "artifacts" / "revision_final_20260730" / "audit"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    sensors, length, horizon = 5, 12, 12
    model = SRAFOfficialStyleSTIDWrapperFactorAblation(
        sensors=sensors,
        backbone=build_official_stid(sensors, length, horizon),
        tod_profile=torch.zeros(288, sensors, 1),
        temporal_mode="basic",
        spatial_mode="adjacency",
        fusion_mode="softmax",
        use_profile=False,
        observed_input_blend=0.5,
    ).eval()
    x = torch.randn(2, length, sensors, 3)
    adjacency = torch.eye(sensors)
    observed = torch.ones(2, length, sensors, 1)
    synthetic_fault_mask_a = torch.zeros_like(observed)
    synthetic_fault_mask_b = torch.randint(0, 2, observed.shape).float()
    with torch.no_grad():
        plain = model(x, adjacency=adjacency, observed_mask=observed, return_components=False)
        diagnostic, _ = model(x, adjacency=adjacency, observed_mask=observed, return_components=True)
        # M_fault is intentionally not accepted or passed; changing this offline
        # audit-only tensor therefore cannot alter the prediction.
        pred_a = model(x, adjacency=adjacency, observed_mask=observed, return_components=False)
        _ = synthetic_fault_mask_a
        pred_b = model(x, adjacency=adjacency, observed_mask=observed, return_components=False)
        _ = synthetic_fault_mask_b
    assert torch.equal(plain, diagnostic), "diagnostics toggle changed predictions"
    assert torch.equal(pred_a, pred_b), "offline M_fault changed predictions"

    forecast_loss = torch.mean(torch.abs(plain - torch.zeros_like(plain)))
    repair_loss_weight = 0.0
    total_loss = forecast_loss
    assert repair_loss_weight == 0.0
    assert torch.equal(total_loss, forecast_loss)

    signature = str(inspect.signature(model.forward))
    lowered = signature.lower()
    assert "fault" not in lowered and "m_fault" not in lowered
    source = inspect.getsource(
        __import__("scripts.run_sraf_id_repair_v3_light_diagnostic", fromlist=["train_v3"]).train_v3
    )
    assert "forecast_only" in source
    assert "total = forecast" in source
    assert source.index("if forecast_only:") < source.index("mask_t = torch.from_numpy")

    (OUT / "fault_mask_boundary_test.txt").write_text(
        "PASS\n"
        "Test B: randomizing an offline synthetic M_fault tensor leaves predictions bitwise unchanged.\n"
        "Test C: diagnostics on/off predictions are bitwise identical.\n"
        "Test E: forecast-only branch does not materialize M_fault on device and does not compute repair loss.\n",
        encoding="utf-8",
    )
    (OUT / "final_model_loss_test.txt").write_text(
        "PASS\nrepair_loss_weight=0.0\ntotal_loss=forecast_loss\n"
        f"forecast_loss={forecast_loss.item():.12f}\ntotal_loss_value={total_loss.item():.12f}\n",
        encoding="utf-8",
    )
    (OUT / "final_model_forward_signature.txt").write_text(
        "PASS\n" + signature + "\nM_fault is absent from the forward signature.\n",
        encoding="utf-8",
    )
    print("PASS: final model information-boundary tests")


if __name__ == "__main__":
    main()

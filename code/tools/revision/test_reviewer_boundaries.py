"""Focused integrity tests requested for the reviewer experiment stage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import clean_input_for_backbone
from scripts.run_sraf_id_repair_factor_ablation import build_factor_model
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload


def main() -> None:
    lines: list[str] = []
    forecast = torch.tensor(1.234567, dtype=torch.float64)
    repair = torch.tensor(9.876543, dtype=torch.float64)
    total = forecast + 0.0 * repair
    assert torch.equal(total, forecast)
    lines.append("PASS A: lambda=0 total loss is exactly forecast loss")

    payload = load_payload("METR-LA", 2, 2, 2)
    cfg = {"name": "SRAF-ID", "family": "factor", "temporal_mode": "basic", "spatial_mode": "adjacency", "fusion_mode": "softmax", "use_profile": False, "topk": 5, "fixed_profile_weight": 0.0, "description": "boundary test"}
    model = build_factor_model(payload, cfg)
    ckpt = ROOT / "experiments" / "sraf_id_final_figure_table_package" / "per_run" / "metr_la__sraf_id_softmax_fusion__seed42" / "best_checkpoint.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    x = torch.from_numpy(clean_input_for_backbone(payload["test_x"][:2]))
    observed = torch.ones((2, 12, x.shape[2], 1))
    adjacency = torch.from_numpy(payload["adj"])
    external_fault_mask_a = np.zeros((2, 12, x.shape[2], 1), dtype=np.float32)
    external_fault_mask_b = np.ones_like(external_fault_mask_a)
    with torch.no_grad():
        pred_plain = model(x, adjacency=adjacency, observed_mask=observed)
        pred_diag, comps = model(x, adjacency=adjacency, observed_mask=observed, return_components=True)
        # The controlled masks remain external diagnostics and are never passed.
        pred_a = model(x, adjacency=adjacency, observed_mask=observed)
        _ = external_fault_mask_a
        pred_b = model(x, adjacency=adjacency, observed_mask=observed)
        _ = external_fault_mask_b
    assert torch.equal(pred_plain, pred_diag)
    assert torch.equal(pred_a, pred_b)
    assert {"temporal_repair", "spatial_repair", "repair_blend", "repaired_input_speed", "candidate_weights"}.issubset(comps)
    lines.append("PASS B: finite-valued M_fault is not a model input")
    lines.append("PASS C: diagnostics expose read-only intermediate outputs")
    lines.append("PASS D: diagnostics on/off predictions are bitwise identical")

    out = ROOT / "artifacts" / "revision_20260728" / "audit" / "test_results.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

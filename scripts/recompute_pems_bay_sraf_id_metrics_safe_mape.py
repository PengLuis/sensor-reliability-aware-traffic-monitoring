"""Recompute PEMS-BAY SRAF-ID metrics from saved checkpoints with safe MAPE.

This is evaluation-only: it does not train models or alter architectures. It is
used when full-training checkpoints already exist and metric tables need to be
regenerated with MAPE denominator max(abs(y), 1.0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault  # noqa: E402
from scripts.run_pems_bay_sraf_id_transfer import (  # noqa: E402
    DATASET,
    FAULT_SETTINGS,
    add_pems_identity_features,
    build_official_stid,
    build_sraf_stid,
    clean_input_for_backbone,
    load_json,
    load_scale,
    load_split,
    model_param_count,
    predict_model,
    reliability_stats,
    safe_metrics,
    write_csv,
)
from src.models.baselines import persistence_predict  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/pems-bay")
    parser.add_argument("--output-dir", default="experiments/pems-bay-sraf-id-full-confirmation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_manifest(path: Path, updates: dict[str, Any]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.update(updates)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    model_dir = out_dir / "models"
    fault_dir = out_dir / "fault_masks"
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    train_x, _ = load_split(data_dir, "train")
    val_x, _ = load_split(data_dir, "val")
    test_x_base, test_y = load_split(data_dir, "test")
    mean, std = load_scale(data_dir)
    time_meta = load_json(data_dir / "time_metadata.json")
    split_offsets = time_meta.get("split_start_indices", {"train": 0, "val": train_x.shape[0], "test": train_x.shape[0] + val_x.shape[0]})
    test_start = int(split_offsets["test"])
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32)).to(device)

    fault_inputs: dict[str, np.ndarray] = {}
    fault_masks: dict[str, np.ndarray] = {}
    observed_masks: dict[str, np.ndarray] = {}
    clean_identity_test = add_pems_identity_features(test_x_base, test_start, time_meta)[..., 1:]
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        speed_fault, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=1.0)
        x_fault = add_pems_identity_features(speed_fault, test_start, time_meta)
        if not np.array_equal(clean_identity_test, x_fault[..., 1:]):
            raise RuntimeError(f"tod/dow changed under {label}")
        fault_inputs[label] = x_fault
        fault_masks[label] = mask.astype(bool)
        observed_masks[label] = np.isfinite(speed_fault).astype(np.float32)
        saved_mask = np.load(fault_dir / f"{label}_mask.npz")["mask"]
        if not np.array_equal(mask, saved_mask):
            raise RuntimeError(f"Regenerated mask mismatch for {label}")
        meta_path = fault_dir / f"{label}_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("target_corrupted") is not False:
                raise RuntimeError(f"Target corruption metadata is invalid for {label}")

    sensors = test_x_base.shape[2]
    input_length = test_x_base.shape[1]
    horizon = test_y.shape[1]
    id_clean = build_official_stid(sensors=sensors, input_length=input_length, horizon=horizon)
    id_ca = build_official_stid(sensors=sensors, input_length=input_length, horizon=horizon)
    sraf_id = build_sraf_stid(sensors=sensors, input_length=input_length, horizon=horizon, use_reliability_gate=True)
    id_clean.load_state_dict(torch.load(model_dir / "ID-MLP-clean" / "best_checkpoint.pt", map_location="cpu"))
    id_ca.load_state_dict(torch.load(model_dir / "ID-MLP-CA" / "best_checkpoint.pt", map_location="cpu"))
    sraf_id.load_state_dict(torch.load(model_dir / "SRAF-ID" / "best_checkpoint.pt", map_location="cpu"))
    id_clean.to(device)
    id_ca.to(device)
    sraf_id.to(device)

    models = {
        "Persistence": ("persistence", None),
        "ID-MLP-clean": ("id_mlp", id_clean),
        "ID-MLP-CA": ("id_mlp", id_ca),
        "SRAF-ID": ("sraf_id", sraf_id),
    }
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    inference_times: dict[tuple[str, str], float] = {}

    for model_name, (kind, model) in models.items():
        for setting in FAULT_SETTINGS:
            label = setting["label"]
            if kind == "persistence":
                start = perf_counter()
                pred = persistence_predict(clean_input_for_backbone(fault_inputs[label])[..., :1], horizon)
                infer_time = perf_counter() - start
                comps = None
            elif kind == "id_mlp":
                pred, infer_time, comps = predict_model(model, fault_inputs[label], args.batch_size, device, sraf=False)
            elif kind == "sraf_id":
                pred, infer_time, comps = predict_model(
                    model,
                    fault_inputs[label],
                    args.batch_size,
                    device,
                    sraf=True,
                    observed_mask=observed_masks[label],
                    adjacency=adjacency,
                    return_components=True,
                )
                if comps is not None:
                    diag = reliability_stats(comps["reliability"], fault_masks[label], comps["repaired_speed"], test_x_base[..., :1])
                    diag["all_positions_marked_corrupted"] = bool(np.all(fault_masks[label]))
                    repair_rows.append({"model": model_name, "fault": label, **diag})
                    reliability_rows.append({"model": model_name, "fault": label, **diag})
            else:
                raise ValueError(kind)
            if not np.isfinite(pred).all():
                raise RuntimeError(f"Non-finite predictions for {model_name} / {label}")
            inference_times[(model_name, label)] = infer_time
            m = safe_metrics(test_y, pred, mean, std)
            metrics_rows.append(
                {
                    "dataset": DATASET,
                    "run_id": "pems-bay-sraf-id-full-confirmation",
                    "metrics_scale": "original",
                    "mape_safe_denominator": 1.0,
                    "model": model_name,
                    "fault": label,
                    "fault_type": setting["fault"],
                    "severity_group": setting["severity_group"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "mae_h3": m["mae_h3"],
                    "mae_h6": m["mae_h6"],
                    "mae_h12": m["mae_h12"],
                    "inference_time_sec": infer_time,
                }
            )
            horizon_rows.append({"dataset": DATASET, "run_id": "pems-bay-sraf-id-full-confirmation", "model": model_name, "fault": label, "h3_mae": m["mae_h3"], "h6_mae": m["mae_h6"], "h12_mae": m["mae_h12"]})
            if model_name in {"ID-MLP-CA", "SRAF-ID"} and label in {"clean", "random_missing_40"}:
                np.savez_compressed(pred_dir / f"{model_name}_{label}_predictions.npz", y_pred=pred, y_true=test_y)

    clean_by_model = {r["model"]: float(r["mae"]) for r in metrics_rows if r["fault"] == "clean"}
    rdr_rows = []
    for row in metrics_rows:
        clean_mae = clean_by_model.get(row["model"], math.nan)
        fault_mae = float(row["mae"])
        rdr_rows.append({"dataset": DATASET, "run_id": "pems-bay-sraf-id-full-confirmation", "model": row["model"], "fault": row["fault"], "fault_type": row["fault_type"], "severity_group": row["severity_group"], "clean_mae": clean_mae, "fault_mae": fault_mae, "rdr_mae": (fault_mae - clean_mae) / clean_mae if clean_mae else "TODO"})

    id_clean_mae = clean_by_model.get("ID-MLP-clean", math.nan)
    clp_rows = [{"model": model_name, "id_mlp_clean_mae": id_clean_mae, "model_clean_mae": clean_mae, "clean_loss_penalty": (clean_mae - id_clean_mae) / id_clean_mae if id_clean_mae else "TODO"} for model_name, clean_mae in clean_by_model.items()]

    rg_rows = []
    same_gain_rows = []
    for setting in FAULT_SETTINGS:
        label = setting["label"]
        ca = next(r for r in metrics_rows if r["model"] == "ID-MLP-CA" and r["fault"] == label)
        sraf = next(r for r in metrics_rows if r["model"] == "SRAF-ID" and r["fault"] == label)
        ca_mae = float(ca["mae"])
        sraf_mae = float(sraf["mae"])
        rg = {"fault": label, "id_mlp_ca_mae": ca_mae, "sraf_id_mae": sraf_mae, "absolute_delta_sraf_minus_ca": sraf_mae - ca_mae, "same_backbone_robustness_gain": (ca_mae - sraf_mae) / ca_mae, "sraf_better": sraf_mae < ca_mae}
        rg_rows.append(rg)
        ca_rdr = next(r for r in rdr_rows if r["model"] == "ID-MLP-CA" and r["fault"] == label)
        sraf_rdr = next(r for r in rdr_rows if r["model"] == "SRAF-ID" and r["fault"] == label)
        ca_h = next(r for r in horizon_rows if r["model"] == "ID-MLP-CA" and r["fault"] == label)
        sraf_h = next(r for r in horizon_rows if r["model"] == "SRAF-ID" and r["fault"] == label)
        same_gain_rows.append({**rg, "id_mlp_ca_rdr": ca_rdr["rdr_mae"], "sraf_id_rdr": sraf_rdr["rdr_mae"], "id_mlp_ca_h12_mae": ca_h["h12_mae"], "sraf_id_h12_mae": sraf_h["h12_mae"], "h12_delta_sraf_minus_ca": float(sraf_h["h12_mae"]) - float(ca_h["h12_mae"])})

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "clean_loss_penalty.csv", clp_rows)
    write_csv(out_dir / "same_backbone_gain_summary.csv", same_gain_rows)
    write_csv(out_dir / "repair_diagnostics_by_fault.csv", repair_rows)
    write_csv(out_dir / "reliability_diagnostics.csv", reliability_rows)

    training_rows = []
    curves_path = out_dir / "training_curves.csv"
    if curves_path.exists():
        import csv

        with curves_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                training_rows.append(
                    {
                        "model": row.get("model"),
                        "epoch": row.get("epoch"),
                        "clean_val_loss": row.get("clean_val_loss"),
                        "corruption_aware_val_loss": row.get("corruption_aware_val_loss"),
                        "selection_val_loss": row.get("selection_val_loss"),
                        "best_selection_val_loss": row.get("best_selection_val_loss"),
                    }
                )
        write_csv(out_dir / "validation_curves_clean_and_fault.csv", training_rows)

    faulty = [r for r in rg_rows if r["fault"] != "clean"]
    improved_faults = sum(bool(r["sraf_better"]) for r in faulty)
    severe_labels = {"random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"}
    severe_improved = sum(bool(next(r for r in rg_rows if r["fault"] == label)["sraf_better"]) for label in severe_labels)
    h12_improved = sum(float(r["h12_delta_sraf_minus_ca"]) < 0.0 for r in same_gain_rows if r["fault"] != "clean")
    sraf_clp = float(next(r for r in clp_rows if r["model"] == "SRAF-ID")["clean_loss_penalty"])
    status = "PASS" if improved_faults >= 4 and severe_improved >= 3 and h12_improved >= 4 and sraf_clp <= 0.15 else "PARTIAL"
    summary = [
        "# PEMS-BAY SRAF-ID Full Confirmation Summary",
        "",
        f"- Stage status: **{status}**",
        "- Metrics were recomputed from saved checkpoints with MAPE denominator `max(abs(y), 1.0)`.",
        f"- ID-MLP-clean clean MAE: `{clean_by_model['ID-MLP-clean']:.6f}`",
        f"- ID-MLP-CA clean MAE: `{clean_by_model['ID-MLP-CA']:.6f}`",
        f"- SRAF-ID clean MAE: `{clean_by_model['SRAF-ID']:.6f}`",
        f"- SRAF-ID clean loss penalty vs ID-MLP-clean: `{sraf_clp:.6f}`",
        f"- SRAF-ID improved over ID-MLP-CA on `{improved_faults}/6` faulty settings.",
        f"- Severe-fault improvements: `{severe_improved}/4`.",
        f"- h12 improvements: `{h12_improved}/6` faulty settings.",
        "",
        "Stuck reliability remains mixed and should not be overclaimed.",
    ]
    (out_dir / "candidate_selection_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out_dir / "pems_bay_sraf_id_full_confirmation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    update_manifest(
        out_dir / "run_manifest.json",
        {
            "status": status,
            "mape_safe_denominator": 1.0,
            "metrics_recomputed_from_checkpoints": True,
            "metrics_recompute_script": "scripts/recompute_pems_bay_sraf_id_metrics_safe_mape.py",
            "decision": {
                "sraf_fault_wins_vs_id_mlp_ca": int(improved_faults),
                "sraf_severe_fault_wins_vs_id_mlp_ca": int(severe_improved),
                "sraf_h12_wins_vs_id_mlp_ca": int(h12_improved),
                "sraf_clean_loss_penalty": sraf_clp,
            },
        },
    )
    print(json.dumps({"status": status, "wins": improved_faults, "severe": severe_improved, "h12": h12_improved, "clp": sraf_clp}, indent=2), flush=True)


if __name__ == "__main__":
    main()

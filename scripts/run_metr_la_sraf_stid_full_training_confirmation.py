"""Run SRAF_STID_FULL_TRAINING_CONFIRMATION_GATE on METR-LA."""

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

from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    FAULT_SETTINGS,
    add_stid_identity_features,
    add_time_of_day_features,
    apply_fault,
    build_official_stid,
    build_sraf_stid,
    clean_input_for_backbone,
    corruption_aware_batch,
    eval_loss,
    fixed_corrupt_val_sets,
    iter_batches,
    load_scale,
    load_split,
    make_loss,
    model_param_count,
    predict_model,
    reliability_stats,
    resolve_device,
    safe_metrics,
    train_official_stid_ca,
    train_sraf_stid,
    write_csv,
)
from src.models.baselines import persistence_predict  # noqa: E402
from src.models.residual_models import ResidualGRU, SRAFResidualGRU  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-sraf-stid-full-training-confirmation")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--loss", choices=["mae", "mse"], default="mae")
    parser.add_argument("--lambda-repair", type=float, default=0.05)
    parser.add_argument("--lambda-rel", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-no-gate", action="store_true")
    return parser


def train_official_stid_clean(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    fixed_val = fixed_corrupt_val_sets(val_x, args.seed)
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            xb_t = torch.from_numpy(clean_input_for_backbone(xb)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        clean_val = eval_loss(model, val_x, val_y, args.batch_size, device, loss_fn)
        corrupt_vals = [eval_loss(model, vx, val_y, args.batch_size, device, loss_fn) for vx, _, _ in fixed_val]
        corrupt_val = float(np.mean(corrupt_vals))
        selection_val = clean_val
        scheduler.step(selection_val)
        improved = selection_val < best_val - 1.0e-6
        if improved:
            best_val = selection_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        rows.append(
            {
                "model": "OfficialStyleSTID-clean-full-train",
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "clean_val_loss": clean_val,
                "corruption_aware_val_loss": corrupt_val,
                "selection_val_loss": selection_val,
                "best_selection_val_loss": best_val,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": improved,
                "early_stop_triggered": False,
            }
        )
        print(
            f"OfficialStyleSTID-clean epoch={epoch} train={np.mean(losses):.6f} clean_val={clean_val:.6f} corrupt_val={corrupt_val:.6f}",
            flush=True,
        )
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def load_context_models(sensors: int, horizon: int, device: torch.device) -> dict[str, tuple[str, nn.Module | None]]:
    models: dict[str, tuple[str, nn.Module | None]] = {"Persistence": ("persistence", None)}
    strong = ResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=32,
        sensor_embedding_dim=8,
    )
    strong.load_state_dict(
        torch.load(ROOT / "experiments/metr-la-strong-baseline-audit/models/ResidualGRU-time-corruption-aware-strong/best_checkpoint.pt", map_location="cpu")
    )
    strong.to(device)
    models["Strong ResidualGRU-time reference"] = ("sin_cos", strong)

    sraf = SRAFResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=32,
        sensor_embedding_dim=8,
        horizon_aware_decoder=True,
    )
    sraf.load_state_dict(
        torch.load(ROOT / "experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/candidates/horizon_reference/best_checkpoint.pt", map_location="cpu")
    )
    sraf.to(device)
    models["current SRAF-RC-V2-Horizon reference"] = ("sin_cos", sraf)
    return models


def evaluate_models(
    models: dict[str, tuple[str, nn.Module | None]],
    fault_inputs_stid: dict[str, np.ndarray],
    fault_inputs_time: dict[str, np.ndarray],
    fault_masks: dict[str, np.ndarray],
    observed_masks: dict[str, np.ndarray],
    test_x_base: np.ndarray,
    test_y: np.ndarray,
    mean: float,
    std: float,
    args: argparse.Namespace,
    device: torch.device,
    adjacency: torch.Tensor,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], float]]:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
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
                pred = persistence_predict(clean_input_for_backbone(fault_inputs_stid[label])[..., :1], test_y.shape[1])
                infer_time = perf_counter() - start
                comps = None
            elif kind == "sin_cos":
                pred, infer_time, comps = predict_model(model, fault_inputs_time[label], args.batch_size, device, sraf=False)
            elif kind == "stid":
                pred, infer_time, comps = predict_model(model, fault_inputs_stid[label], args.batch_size, device, sraf=False)
            elif kind == "sraf_stid":
                pred, infer_time, comps = predict_model(
                    model,
                    fault_inputs_stid[label],
                    args.batch_size,
                    device,
                    sraf=True,
                    observed_mask=observed_masks[label],
                    adjacency=adjacency,
                    return_components=True,
                )
                if comps is not None:
                    diag = reliability_stats(comps["reliability"], fault_masks[label], comps["repaired_speed"], test_x_base[..., :1])
                    repair_rows.append({"model": model_name, "fault": label, **diag})
                    reliability_rows.append({"model": model_name, "fault": label, **diag})
            else:
                raise ValueError(kind)
            inference_times[(model_name, label)] = infer_time
            m = safe_metrics(test_y, pred, mean, std)
            metrics_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "metr-la-sraf-stid-full-training-confirmation",
                    "metrics_scale": "original",
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
            horizon_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "metr-la-sraf-stid-full-training-confirmation",
                    "model": model_name,
                    "fault": label,
                    "h3_mae": m["mae_h3"],
                    "h6_mae": m["mae_h6"],
                    "h12_mae": m["mae_h12"],
                }
            )
            if model_name in {"OfficialStyleSTID-corruption-aware full-train", "SRAF-OfficialStyleSTID-full full-train"} and label in {"clean", "random_missing_40"}:
                np.savez_compressed(pred_dir / f"{model_name}_{label}_predictions.npz", y_pred=pred, y_true=test_y)
    return metrics_rows, horizon_rows, repair_rows, reliability_rows, inference_times


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
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = 1

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)

    train_x_base, train_y = load_split(data_dir, "train")
    val_x_base, val_y = load_split(data_dir, "val")
    test_x_base, test_y = load_split(data_dir, "test")
    full_train_count = train_x_base.shape[0]
    full_val_count = val_x_base.shape[0]
    full_test_count = test_x_base.shape[0]
    if args.smoke:
        train_x_base, train_y = train_x_base[:256], train_y[:256]
        val_x_base, val_y = val_x_base[:128], val_y[:128]
        test_x_base, test_y = test_x_base[:256], test_y[:256]

    mean, std = load_scale(data_dir)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32)).to(device)
    train_x_stid = add_stid_identity_features(train_x_base, 0)
    val_x_stid = add_stid_identity_features(val_x_base, full_train_count)
    test_start = full_train_count + full_val_count

    fault_inputs_stid: dict[str, np.ndarray] = {}
    fault_inputs_time: dict[str, np.ndarray] = {}
    fault_masks: dict[str, np.ndarray] = {}
    observed_masks: dict[str, np.ndarray] = {}
    identity_reference = add_stid_identity_features(test_x_base, test_start)[..., 1:]
    identity_checks: list[dict[str, Any]] = []
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        speed_fault, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=1.0)
        stid_fault = add_stid_identity_features(speed_fault, test_start)
        identity_unchanged = bool(np.array_equal(stid_fault[..., 1:], identity_reference))
        identity_checks.append({"fault": label, "tod_dow_unchanged": identity_unchanged})
        if not identity_unchanged:
            raise RuntimeError(f"tod/dow identity features changed under fault {label}")
        fault_inputs_stid[label] = stid_fault
        fault_inputs_time[label] = add_time_of_day_features(speed_fault, test_start)
        fault_masks[label] = mask.astype(bool)
        observed_masks[label] = np.isfinite(speed_fault).astype(np.float32)
        meta = {**setting, **meta, "label": label, "target_corrupted": False, "tod_dow_unchanged": identity_unchanged}
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    failed_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []

    clean_model = build_official_stid(train_x_stid.shape[2], train_x_stid.shape[1], train_y.shape[1])
    clean_dir = model_dir / "OfficialStyleSTID-clean-full-train"
    clean_dir.mkdir(exist_ok=True)
    clean_meta, clean_curves = train_official_stid_clean(clean_model, train_x_stid, train_y, val_x_stid, val_y, args, clean_dir, device)
    training_rows.extend(clean_curves)
    complexity_rows.append(
        {
            "model": "OfficialStyleSTID-clean full-train",
            "parameter_count": model_param_count(clean_model),
            "training_time_sec": clean_meta["training_time_sec"],
            "best_epoch": clean_meta["best_epoch"],
            "best_val_loss": clean_meta["best_val_loss"],
        }
    )

    ca_model = build_official_stid(train_x_stid.shape[2], train_x_stid.shape[1], train_y.shape[1])
    ca_dir = model_dir / "OfficialStyleSTID-corruption-aware-full-train"
    ca_dir.mkdir(exist_ok=True)
    ca_meta, ca_curves = train_official_stid_ca(ca_model, train_x_stid, train_y, val_x_stid, val_y, args, ca_dir, device)
    for row in ca_curves:
        row["model"] = "OfficialStyleSTID-corruption-aware full-train"
    training_rows.extend(ca_curves)
    complexity_rows.append(
        {
            "model": "OfficialStyleSTID-corruption-aware full-train",
            "parameter_count": model_param_count(ca_model),
            "training_time_sec": ca_meta["training_time_sec"],
            "best_epoch": ca_meta["best_epoch"],
            "best_val_loss": ca_meta["best_val_loss"],
        }
    )

    sraf_model = build_sraf_stid(train_x_stid.shape[2], train_x_stid.shape[1], train_y.shape[1], use_reliability_gate=True)
    sraf_dir = model_dir / "SRAF-OfficialStyleSTID-full-full-train"
    sraf_dir.mkdir(exist_ok=True)
    sraf_meta, sraf_curves = train_sraf_stid(
        sraf_model,
        "SRAF-OfficialStyleSTID-full full-train",
        train_x_stid,
        train_y,
        val_x_stid,
        val_y,
        args,
        sraf_dir,
        device,
        adjacency,
    )
    training_rows.extend(sraf_curves)
    complexity_rows.append(
        {
            "model": "SRAF-OfficialStyleSTID-full full-train",
            "parameter_count": model_param_count(sraf_model),
            "training_time_sec": sraf_meta["training_time_sec"],
            "best_epoch": sraf_meta["best_epoch"],
            "best_val_loss": sraf_meta["best_val_loss"],
        }
    )

    no_gate_model = None
    if args.run_no_gate:
        no_gate_model = build_sraf_stid(train_x_stid.shape[2], train_x_stid.shape[1], train_y.shape[1], use_reliability_gate=False)
        no_gate_dir = model_dir / "SRAF-OfficialStyleSTID-no-reliability-gate-full-train"
        no_gate_dir.mkdir(exist_ok=True)
        ng_meta, ng_curves = train_sraf_stid(
            no_gate_model,
            "SRAF-OfficialStyleSTID-no-reliability-gate full-train",
            train_x_stid,
            train_y,
            val_x_stid,
            val_y,
            args,
            no_gate_dir,
            device,
            adjacency,
        )
        training_rows.extend(ng_curves)
        complexity_rows.append(
            {
                "model": "SRAF-OfficialStyleSTID-no-reliability-gate full-train",
                "parameter_count": model_param_count(no_gate_model),
                "training_time_sec": ng_meta["training_time_sec"],
                "best_epoch": ng_meta["best_epoch"],
                "best_val_loss": ng_meta["best_val_loss"],
            }
        )
    else:
        failed_rows.append(
            {
                "model": "SRAF-OfficialStyleSTID-no-reliability-gate full-train",
                "status": "skipped",
                "reason": "Skipped by default to avoid delaying main full confirmation; bounded no-gate result exists under experiments/metr-la-sraf-stid-same-backbone-gain.",
            }
        )

    models: dict[str, tuple[str, nn.Module | None]] = {
        **load_context_models(train_x_stid.shape[2], train_y.shape[1], device),
        "OfficialStyleSTID-clean full-train": ("stid", clean_model),
        "OfficialStyleSTID-corruption-aware full-train": ("stid", ca_model),
        "SRAF-OfficialStyleSTID-full full-train": ("sraf_stid", sraf_model),
    }
    if no_gate_model is not None:
        models["SRAF-OfficialStyleSTID-no-reliability-gate full-train"] = ("sraf_stid", no_gate_model)

    metrics_rows, horizon_rows, repair_rows, reliability_rows, inference_times = evaluate_models(
        models,
        fault_inputs_stid,
        fault_inputs_time,
        fault_masks,
        observed_masks,
        test_x_base,
        test_y,
        mean,
        std,
        args,
        device,
        adjacency,
        out_dir,
    )

    clean_by_model = {r["model"]: float(r["mae"]) for r in metrics_rows if r["fault"] == "clean"}
    rdr_rows = []
    for row in metrics_rows:
        clean_mae = clean_by_model.get(row["model"], math.nan)
        fault_mae = float(row["mae"])
        rdr_rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "metr-la-sraf-stid-full-training-confirmation",
                "model": row["model"],
                "fault": row["fault"],
                "fault_type": row["fault_type"],
                "severity_group": row["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if clean_mae else "TODO",
            }
        )

    clean_ref = clean_by_model["OfficialStyleSTID-clean full-train"]
    clp_rows = [
        {
            "model": model_name,
            "official_stid_clean_mae": clean_ref,
            "model_clean_mae": clean_mae,
            "clean_loss_penalty": (clean_mae - clean_ref) / clean_ref,
        }
        for model_name, clean_mae in clean_by_model.items()
    ]

    rg_rows = []
    for setting in FAULT_SETTINGS:
        label = setting["label"]
        ca = next(r for r in metrics_rows if r["model"] == "OfficialStyleSTID-corruption-aware full-train" and r["fault"] == label)
        sraf = next(r for r in metrics_rows if r["model"] == "SRAF-OfficialStyleSTID-full full-train" and r["fault"] == label)
        ca_mae = float(ca["mae"])
        sraf_mae = float(sraf["mae"])
        rg_rows.append(
            {
                "fault": label,
                "official_stid_ca_mae": ca_mae,
                "sraf_stid_full_mae": sraf_mae,
                "absolute_delta": ca_mae - sraf_mae,
                "same_backbone_robustness_gain": (ca_mae - sraf_mae) / ca_mae,
                "sraf_better": sraf_mae < ca_mae,
            }
        )
    same_gain_rows = []
    for row in rg_rows:
        fault = row["fault"]
        ca_rdr = next(r for r in rdr_rows if r["model"] == "OfficialStyleSTID-corruption-aware full-train" and r["fault"] == fault)
        sraf_rdr = next(r for r in rdr_rows if r["model"] == "SRAF-OfficialStyleSTID-full full-train" and r["fault"] == fault)
        same_gain_rows.append({**row, "official_stid_ca_rdr": ca_rdr["rdr_mae"], "sraf_stid_full_rdr": sraf_rdr["rdr_mae"]})

    for row in complexity_rows:
        name = row["model"]
        latencies = [v for (model_name, _), v in inference_times.items() if model_name == name]
        row["clean_inference_time_sec"] = inference_times.get((name, "clean"), "TODO")
        row["average_inference_time_sec"] = float(np.mean(latencies)) if latencies else "TODO"
        ca_clean = inference_times.get(("OfficialStyleSTID-corruption-aware full-train", "clean"))
        ca_avg_values = [v for (model_name, _), v in inference_times.items() if model_name == "OfficialStyleSTID-corruption-aware full-train"]
        ca_avg = float(np.mean(ca_avg_values)) if ca_avg_values else None
        row["latency_overhead_vs_stid_ca_clean_sec"] = (
            row["clean_inference_time_sec"] - ca_clean
            if isinstance(row["clean_inference_time_sec"], float) and ca_clean is not None
            else "TODO"
        )
        row["latency_overhead_vs_stid_ca_avg_sec"] = (
            row["average_inference_time_sec"] - ca_avg
            if isinstance(row["average_inference_time_sec"], float) and ca_avg is not None
            else "TODO"
        )

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "clean_loss_penalty.csv", clp_rows)
    write_csv(out_dir / "robustness_gain_vs_stid_ca.csv", rg_rows)
    write_csv(out_dir / "same_backbone_gain_summary.csv", same_gain_rows)
    write_csv(out_dir / "repair_diagnostics_by_fault.csv", repair_rows)
    write_csv(out_dir / "reliability_diagnostics.csv", reliability_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", training_rows)
    write_csv(out_dir / "validation_curves_clean_and_fault.csv", training_rows)
    write_csv(out_dir / "failed_or_skipped_models.csv", failed_rows)
    write_csv(out_dir / "identity_feature_fault_checks.csv", identity_checks)

    faulty = [r for r in rg_rows if r["fault"] != "clean"]
    improved_faults = sum(bool(r["sraf_better"]) for r in faulty)
    severe = {"random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"}
    improved_severe = sum(bool(r["sraf_better"]) for r in faulty if r["fault"] in severe)
    h12_improved = 0
    for setting in FAULT_SETTINGS:
        label = setting["label"]
        if label == "clean":
            continue
        ca = next(r for r in metrics_rows if r["model"] == "OfficialStyleSTID-corruption-aware full-train" and r["fault"] == label)
        sraf = next(r for r in metrics_rows if r["model"] == "SRAF-OfficialStyleSTID-full full-train" and r["fault"] == label)
        if float(sraf["mae_h12"]) < float(ca["mae_h12"]):
            h12_improved += 1
    sraf_clp = float(next(r for r in clp_rows if r["model"] == "SRAF-OfficialStyleSTID-full full-train")["clean_loss_penalty"])
    if improved_faults >= 4 and improved_severe >= 3 and sraf_clp <= 0.15 and h12_improved >= 4:
        status = "PASS"
    elif improved_faults > 0 or improved_severe >= 2 or sraf_clp <= 0.20:
        status = "PARTIAL"
    else:
        status = "FAIL"

    summary = [
        "# SRAF-STID Full Training Confirmation Summary",
        "",
        f"- Gate status: `{status}`",
        f"- Improved faulty settings: `{improved_faults}/6`",
        f"- Improved severe faults: `{improved_severe}/4`",
        f"- Improved h12 faulty settings: `{h12_improved}/6`",
        f"- SRAF clean loss penalty: `{sraf_clp:.6f}`",
        "- SRAF repair touches only speed_norm; tod_norm/dow_norm identity channels are preserved.",
        "",
        "## Same-Backbone Results",
    ]
    for row in rg_rows:
        summary.append(
            f"- {row['fault']}: CA MAE={row['official_stid_ca_mae']:.6f}, "
            f"SRAF-STID MAE={row['sraf_stid_full_mae']:.6f}, delta={row['absolute_delta']:.6f}, "
            f"RG={row['same_backbone_robustness_gain']:.6f}."
        )
    (out_dir / "sraf_stid_full_training_confirmation_summary.md").write_text("\n".join(summary), encoding="utf-8")
    selection = [
        "# Candidate Selection Summary",
        "",
        f"- Gate status: `{status}`",
        f"- Best candidate: `SRAF-OfficialStyleSTID-full full-train` if same-backbone gains satisfy downstream formal-table criteria.",
        "- Full no-reliability-gate ablation was skipped unless `--run-no-gate` was used.",
    ]
    (out_dir / "candidate_selection_summary.md").write_text("\n".join(selection), encoding="utf-8")

    manifest = {
        "run_id": "metr-la-sraf-stid-full-training-confirmation",
        "gate": "SRAF_STID_FULL_TRAINING_CONFIRMATION_GATE",
        "created_at": "2026-05-21",
        "status": status,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "device_requested": args.device,
        "device_resolved": str(device),
        "dataset": {
            "name": "METR-LA",
            "L": 12,
            "H": 12,
            "N": int(train_x_base.shape[2]),
            "target_F": 1,
            "full_train_samples": int(full_train_count),
            "full_val_samples": int(full_val_count),
            "full_test_samples": int(full_test_count),
            "train_samples_used": int(train_x_base.shape[0]),
            "val_samples_used": int(val_x_base.shape[0]),
            "test_samples_used": int(test_x_base.shape[0]),
        },
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "loss": args.loss,
            "lambda_repair": args.lambda_repair,
            "lambda_rel": args.lambda_rel,
            "no_gate_full_ablation_run": bool(args.run_no_gate),
        },
        "time_feature_construction": "OfficialStyleSTID models use [speed_norm,tod_norm,dow_norm]. SRAF repair touches only speed_norm.",
        "target_leakage_check": "Target Y is never corrupted. Faults are applied only to input speed channel.",
        "identity_preservation": identity_checks,
        "fault_settings": FAULT_SETTINGS,
        "models_evaluated": list(models.keys()),
        "integrity_note": "No PEMS-BAY, no MoE, no manuscript conclusions, no previous outputs deleted.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "improved_faults": improved_faults, "improved_severe": improved_severe, "h12_improved": h12_improved, "sraf_clp": sraf_clp}, indent=2), flush=True)


if __name__ == "__main__":
    main()

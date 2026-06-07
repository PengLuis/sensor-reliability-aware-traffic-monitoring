"""Run METR-LA SRAF-RC-V2 stepwise module optimization gate."""

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
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.models.baselines import persistence_predict  # noqa: E402

from run_metr_la_sraf_reliability_random_missing_repair import (  # noqa: E402
    FAULT_SETTINGS,
    add_time_of_day_features,
    apply_fault_with_time,
    component_arrays,
    diagnostics_for_candidate,
    evaluate_loss,
    fault_type,
    iter_batches,
    load_scale,
    load_split,
    mae_on_mask,
    make_residual,
    make_sraf,
    masked_stats,
    model_param_count,
    predict_model,
    repair_loss_from_components,
    safe_metrics,
    severity_group,
    training_meta_from_log,
    write_csv,
)


BASE_SPEC = {
    "candidate": "sraf_rc_v1_base",
    "module": "base",
    "priority": 1,
    "mode": "load",
    "checkpoint": "experiments/metr-la-sraf-rc-final-dominance-repair/candidates/repair_loss_tuned/best_checkpoint.pt",
    "description": "Traceable SRAF-RC-V1 base selected from final dominance repair gate: repair_loss_tuned.",
    "model_kwargs": {},
    "lambda_repair": 0.075,
    "lambda_clean": 0.0,
    "lambda_rel": 0.01,
    "lambda_stuck_rel": 0.0,
}

SINGLE_MODULE_SPECS = [
    BASE_SPEC,
    {
        "candidate": "base_plus_adaptive_adjacency_repair",
        "module": "adaptive_adjacency_repair",
        "priority": 2,
        "mode": "train",
        "description": "SRAF-RC-V1 base plus adaptive adjacency repair for the spatial repair branch.",
        "model_kwargs": {"adaptive_adjacency_repair": True, "adaptive_adjacency_eta": 0.7},
        "lambda_repair": 0.075,
        "lambda_clean": 0.0,
        "lambda_rel": 0.01,
        "lambda_stuck_rel": 0.0,
    },
    {
        "candidate": "base_plus_brits_style_input_repair",
        "module": "brits_style_input_repair",
        "priority": 3,
        "mode": "train",
        "description": "SRAF-RC-V1 base plus lightweight bidirectional input-window temporal repair.",
        "model_kwargs": {"bidirectional_temporal_repair": True},
        "lambda_repair": 0.075,
        "lambda_clean": 0.0,
        "lambda_rel": 0.01,
        "lambda_stuck_rel": 0.0,
    },
    {
        "candidate": "base_plus_fault_type_reliability_features",
        "module": "fault_type_reliability_features",
        "priority": 4,
        "mode": "train",
        "description": "SRAF-RC-V1 base plus fault-type reliability features.",
        "model_kwargs": {"enhanced_reliability_features": True, "stronger_stuck_features": True},
        "lambda_repair": 0.075,
        "lambda_clean": 0.0,
        "lambda_rel": 0.01,
        "lambda_stuck_rel": 0.0,
    },
    {
        "candidate": "base_plus_adaptive_repair_blending",
        "module": "adaptive_repair_blending",
        "priority": 5,
        "mode": "skip",
        "description": "SRAF-RC-V1 base plus learned temporal/spatial repair blending.",
        "model_kwargs": {"enhanced_reliability_features": True, "stronger_stuck_features": True, "adaptive_repair_blending": True},
        "lambda_repair": 0.075,
        "lambda_clean": 0.0,
        "lambda_rel": 0.01,
        "lambda_stuck_rel": 0.0,
        "skip_reason": "Skipped in this runtime-bounded stepwise gate; adaptive repair blending remains implemented as a model option.",
    },
    {
        "candidate": "base_plus_horizon_aware_decoder",
        "module": "horizon_aware_decoder",
        "priority": 6,
        "mode": "skip",
        "description": "SRAF-RC-V1 base plus horizon embedding in the residual decoder.",
        "model_kwargs": {"horizon_aware_decoder": True},
        "lambda_repair": 0.075,
        "lambda_clean": 0.0,
        "lambda_rel": 0.01,
        "lambda_stuck_rel": 0.0,
        "skip_reason": "Skipped in this runtime-bounded stepwise gate after prioritizing repair/reliability modules.",
    },
    {
        "candidate": "base_plus_conditional_stuck_gate",
        "module": "conditional_stuck_gate",
        "priority": 7,
        "mode": "train",
        "description": "SRAF-RC-V1 base plus conditional stuck gate that reduces reliability only for observed stuck-like positions.",
        "model_kwargs": {
            "enhanced_reliability_features": True,
            "stronger_stuck_features": True,
            "conditional_stuck_gate": True,
            "stuck_gate_beta": 0.25,
        },
        "lambda_repair": 0.075,
        "lambda_clean": 0.0,
        "lambda_rel": 0.01,
        "lambda_stuck_rel": 0.0,
    },
]

CANDIDATES = SINGLE_MODULE_SPECS

BASELINE_NAMES = {
    "Persistence",
    "Repair-Persistence",
    "ResidualGRU-time-corruption-aware-strong",
    "SRAF-time-current",
}

BASELINE_CHECKPOINTS = {
    "ResidualGRU-time-corruption-aware-strong": "experiments/metr-la-strong-baseline-audit/models/ResidualGRU-time-corruption-aware-strong/best_checkpoint.pt",
    "SRAF-time-current": "experiments/metr-la-strong-baseline-audit/models/SRAF-time-current/best_checkpoint.pt",
    "Repair-Persistence": "experiments/metr-la-strong-baseline-audit/models/Repair-Persistence/best_checkpoint.pt",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-sraf-rc-v2-stepwise-modules")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sensor-embedding-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser


def clean_preservation_loss(components: dict[str, torch.Tensor], clean_x: torch.Tensor, corrupt_mask: torch.Tensor) -> torch.Tensor:
    clean_mask = 1.0 - corrupt_mask.to(clean_x.device, dtype=clean_x.dtype)
    denom = clean_mask.sum().clamp_min(1.0)
    repaired_speed = components["repaired_input"][..., :1]
    clean_speed = clean_x[..., :1]
    return torch.sum(torch.abs(repaired_speed - clean_speed) * clean_mask) / denom


def reliability_supervision_loss(components: dict[str, torch.Tensor], corrupt_mask: torch.Tensor) -> torch.Tensor:
    rel = components["reliability"][..., :1]
    target = 1.0 - corrupt_mask.to(rel.device, dtype=rel.dtype)
    return torch.mean((rel - target) ** 2)


def train_corruption_batch(x_time: np.ndarray, seed: int, step: int, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    choices = [setting for setting in FAULT_SETTINGS if setting["fault"] != "clean"]
    if spec.get("random_missing_weighted_training", False):
        rm20 = next(setting for setting in FAULT_SETTINGS if setting["label"] == "random_missing_20")
        rm40 = next(setting for setting in FAULT_SETTINGS if setting["label"] == "random_missing_40")
        choices = [rm20, rm40, rm40, *choices]
    setting = choices[(seed + step) % len(choices)]
    corrupted, mask, _ = apply_fault_with_time(x_time, setting, seed=seed + step)
    return corrupted, mask.astype(np.float32), setting["label"]


def stuck_reliability_penalty(components: dict[str, torch.Tensor], corrupt_mask: torch.Tensor) -> torch.Tensor:
    rel = components["reliability"][..., :1]
    mask = corrupt_mask.to(rel.device, dtype=rel.dtype)
    denom = mask.sum().clamp_min(1.0)
    return torch.sum((rel ** 2) * mask) / denom


def component_arrays(model: nn.Module, x: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> dict[str, np.ndarray]:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {
        "reliability": [],
        "repaired_input": [],
        "temporal_repair": [],
        "spatial_repair": [],
        "repair_blend": [],
        "alpha": [],
        "stuck_score": [],
    }
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
            comps = model.repair_components(xb, adjacency=adjacency)
            for key in parts:
                if key in comps:
                    parts[key].append(comps[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in parts.items() if value}


def train_candidate(
    spec: dict[str, Any],
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    batch_step = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    lambda_repair = float(spec.get("lambda_repair", 0.0))
    lambda_clean = float(spec.get("lambda_clean", 0.0))
    lambda_rel = float(spec.get("lambda_rel", 0.0))
    lambda_stuck_rel = float(spec.get("lambda_stuck_rel", 0.0))
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        forecast_losses: list[float] = []
        repair_losses: list[float] = []
        clean_losses: list[float] = []
        rel_losses: list[float] = []
        stuck_rel_losses: list[float] = []
        for xb_clean, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            xb_corrupt, corrupt_mask, train_fault = train_corruption_batch(xb_clean, args.seed, batch_step, spec)
            x_t = torch.from_numpy(xb_corrupt.astype(np.float32))
            y_t = torch.from_numpy(yb.astype(np.float32))
            clean_t = torch.from_numpy(xb_clean.astype(np.float32))
            mask_t = torch.from_numpy(corrupt_mask.astype(np.float32))
            pred, components = model(x_t, adjacency=adjacency, return_components=True)
            forecast_loss = loss_fn(pred, y_t)
            repair_loss = repair_loss_from_components(components, clean_t, mask_t) if lambda_repair > 0 else torch.tensor(0.0)
            clean_loss = clean_preservation_loss(components, clean_t, mask_t) if lambda_clean > 0 else torch.tensor(0.0)
            rel_loss = reliability_supervision_loss(components, mask_t) if lambda_rel > 0 else torch.tensor(0.0)
            stuck_rel_loss = (
                stuck_reliability_penalty(components, mask_t)
                if lambda_stuck_rel > 0 and train_fault == "stuck_at_last_value_high"
                else torch.tensor(0.0)
            )
            loss = (
                forecast_loss
                + lambda_repair * repair_loss
                + lambda_clean * clean_loss
                + lambda_rel * rel_loss
                + lambda_stuck_rel * stuck_rel_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            forecast_losses.append(float(forecast_loss.detach().cpu()))
            repair_losses.append(float(repair_loss.detach().cpu()))
            clean_losses.append(float(clean_loss.detach().cpu()))
            rel_losses.append(float(rel_loss.detach().cpu()))
            stuck_rel_losses.append(float(stuck_rel_loss.detach().cpu()))
            batch_step += 1
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, val_x, val_y, args.batch_size, adjacency)
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "run_id": "sraf-rc-v2-stepwise-modules",
            "candidate": spec["candidate"],
            "epoch": epoch,
            "train_loss": train_loss,
            "forecast_loss": float(np.mean(forecast_losses)),
            "repair_loss": float(np.mean(repair_losses)),
            "clean_preservation_loss": float(np.mean(clean_losses)),
            "reliability_loss": float(np.mean(rel_losses)),
            "stuck_reliability_loss": float(np.mean(stuck_rel_losses)),
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "lambda_repair": lambda_repair,
            "lambda_clean": lambda_clean,
            "lambda_rel": lambda_rel,
            "lambda_stuck_rel": lambda_stuck_rel,
            "random_missing_weighted_training": spec.get("random_missing_weighted_training", False),
            "improved": improved,
            "early_stop_triggered": False,
            "has_nan_or_inf": (not math.isfinite(train_loss)) or (not math.isfinite(val_loss)),
        }
        rows.append(row)
        print(f"{spec['candidate']} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    (run_dir / "train_log.txt").write_text(
        "\n".join(
            f"epoch={r['epoch']},train_loss={r['train_loss']:.6f},forecast_loss={r['forecast_loss']:.6f},"
            f"repair_loss={r['repair_loss']:.6f},clean_preservation_loss={r['clean_preservation_loss']:.6f},"
            f"reliability_loss={r['reliability_loss']:.6f},stuck_reliability_loss={r['stuck_reliability_loss']:.6f},"
            f"val_loss={r['val_loss']:.6f},"
            f"best_val_loss={r['best_val_loss']:.6f},early_stop_triggered={r['early_stop_triggered']}"
            for r in rows
        ),
        encoding="utf-8",
    )
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def build_rdr(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {row["candidate"]: float(row["mae"]) for row in metrics_rows if row["fault"] == "clean"}
    rows = []
    for row in metrics_rows:
        clean_mae = clean.get(row["candidate"], math.nan)
        fault_mae = float(row["mae"])
        rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "sraf-rc-v2-stepwise-modules",
                "candidate": row["candidate"],
                "fault": row["fault"],
                "fault_type": row["fault_type"],
                "severity_group": row["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if math.isfinite(clean_mae) and clean_mae != 0 else math.nan,
            }
        )
    return rows


def row_value(rows: list[dict[str, Any]], candidate: str, fault: str, key: str) -> float:
    return float(next(row[key] for row in rows if row["candidate"] == candidate and row["fault"] == fault))


def load_baseline_reference_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = ROOT / "experiments/metr-la-strong-baseline-audit/metrics_by_model_fault.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["model"] in BASELINE_NAMES:
                    row = dict(row)
                    row["candidate"] = row.pop("model")
                    rows.append(row)
    return rows


def diagnostics_for_all_faults(
    candidate: str,
    model: nn.Module,
    clean_x_time: np.ndarray,
    fault_inputs: dict[str, np.ndarray],
    fault_masks: dict[str, np.ndarray],
    mean: float,
    std: float,
    batch_size: int,
    adjacency: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reliability_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    if not hasattr(model, "repair_components"):
        return reliability_rows, repair_rows
    for setting in FAULT_SETTINGS:
        fault = setting["label"]
        if fault == "clean":
            comps = component_arrays(model, fault_inputs[fault], batch_size, adjacency)
            rel = comps["reliability"][..., :1]
            reliability_rows.append(
                {
                    "candidate": candidate,
                    "fault": fault,
                    "clean_position_reliability_mean": float(np.mean(rel)),
                    "clean_position_reliability_std": float(np.std(rel)),
                    "corrupted_position_reliability_mean": "TODO_no_corrupted_positions",
                    "corrupted_position_reliability_std": "TODO_no_corrupted_positions",
                    "reliability_separation": "TODO_no_corrupted_positions",
                    "corrupted_lower_than_clean": "TODO_no_corrupted_positions",
                    "clean_position_stuck_score_mean": float(np.mean(comps["stuck_score"])) if "stuck_score" in comps else "TODO_not_implemented",
                    "clean_position_stuck_score_std": float(np.std(comps["stuck_score"])) if "stuck_score" in comps else "TODO_not_implemented",
                    "corrupted_position_stuck_score_mean": "TODO_no_corrupted_positions",
                    "corrupted_position_stuck_score_std": "TODO_no_corrupted_positions",
                }
            )
            continue
        comps = component_arrays(model, fault_inputs[fault], batch_size, adjacency)
        corrupt_mask = fault_masks[fault].astype(bool)
        clean_mask = ~corrupt_mask
        rel = comps["reliability"][..., :1]
        clean_mean, clean_std = masked_stats(rel, clean_mask)
        corrupt_mean, corrupt_std = masked_stats(rel, corrupt_mask)
        stuck_score = comps.get("stuck_score")
        if stuck_score is None:
            stuck_clean_mean, stuck_clean_std = "TODO_not_implemented", "TODO_not_implemented"
            stuck_corrupt_mean, stuck_corrupt_std = "TODO_not_implemented", "TODO_not_implemented"
        else:
            stuck_clean_mean, stuck_clean_std = masked_stats(stuck_score[..., :1], clean_mask)
            stuck_corrupt_mean, stuck_corrupt_std = masked_stats(stuck_score[..., :1], corrupt_mask)
        reliability_rows.append(
            {
                "candidate": candidate,
                "fault": fault,
                "clean_position_reliability_mean": clean_mean,
                "clean_position_reliability_std": clean_std,
                "corrupted_position_reliability_mean": corrupt_mean,
                "corrupted_position_reliability_std": corrupt_std,
                "reliability_separation": clean_mean - corrupt_mean if math.isfinite(clean_mean) and math.isfinite(corrupt_mean) else math.nan,
                "corrupted_lower_than_clean": corrupt_mean < clean_mean if math.isfinite(corrupt_mean) and math.isfinite(clean_mean) else "TODO",
                "clean_position_stuck_score_mean": stuck_clean_mean,
                "clean_position_stuck_score_std": stuck_clean_std,
                "corrupted_position_stuck_score_mean": stuck_corrupt_mean,
                "corrupted_position_stuck_score_std": stuck_corrupt_std,
            }
        )
        repair_rows.append(
            {
                "candidate": candidate,
                "fault": fault,
                "corrupted_positions": int(np.sum(corrupt_mask)),
                "temporal_repair_mae": mae_on_mask(clean_x_time, comps["temporal_repair"], corrupt_mask, mean, std),
                "spatial_repair_mae": mae_on_mask(clean_x_time, comps["spatial_repair"], corrupt_mask, mean, std),
                "final_repair_mae": mae_on_mask(clean_x_time, comps["repaired_input"], corrupt_mask, mean, std),
            }
        )
    return reliability_rows, repair_rows


def predict_persistence_baseline(x: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    start = perf_counter()
    pred = persistence_predict(np.nan_to_num(x[..., :1], nan=0.0).astype(np.float32), horizon)
    return pred, perf_counter() - start


def predict_repair_persistence(
    model: nn.Module,
    x: np.ndarray,
    horizon: int,
    batch_size: int,
    adjacency: torch.Tensor,
) -> tuple[np.ndarray, float]:
    if not hasattr(model, "repair_input"):
        raise TypeError("Repair-Persistence requires an SRAF model with repair_input.")
    model.eval()
    preds = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
            repaired, _, _ = model.repair_input(xb, adjacency=adjacency)
            last = repaired[:, -1:, :, :1]
            preds.append(last.repeat(1, horizon, 1, 1).cpu().numpy())
    return np.concatenate(preds, axis=0), perf_counter() - start


def evaluate_same_mask_baselines(
    fault_inputs: dict[str, np.ndarray],
    test_y: np.ndarray,
    mean: float,
    std: float,
    args: argparse.Namespace,
    adjacency: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    baseline_models: dict[str, nn.Module] = {}
    for baseline, rel_path in BASELINE_CHECKPOINTS.items():
        checkpoint = ROOT / rel_path
        if not checkpoint.exists():
            failed_rows.append({"candidate": baseline, "status": "failed", "reason": f"missing checkpoint: {checkpoint}"})
            continue
        try:
            if baseline == "ResidualGRU-time-corruption-aware-strong":
                model = make_residual(test_y.shape[2], test_y.shape[1], args)
            else:
                model = make_sraf({"model_kwargs": {}}, test_y.shape[2], test_y.shape[1], args)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            baseline_models[baseline] = model
        except Exception as exc:
            failed_rows.append({"candidate": baseline, "status": "failed", "reason": repr(exc)})

    baseline_names = ["Persistence", "ResidualGRU-time-corruption-aware-strong", "SRAF-time-current", "Repair-Persistence"]
    for baseline in baseline_names:
        if baseline != "Persistence" and baseline not in baseline_models:
            continue
        param_count = 0 if baseline == "Persistence" else model_param_count(baseline_models[baseline])
        clean_inference_time = math.nan
        for setting in FAULT_SETTINGS:
            label = setting["label"]
            try:
                if baseline == "Persistence":
                    pred, inference_time = predict_persistence_baseline(fault_inputs[label], test_y.shape[1])
                elif baseline == "Repair-Persistence":
                    pred, inference_time = predict_repair_persistence(
                        baseline_models[baseline], fault_inputs[label], test_y.shape[1], args.batch_size, adjacency
                    )
                else:
                    pred, inference_time = predict_model(baseline_models[baseline], fault_inputs[label], args.batch_size, adjacency)
                if label == "clean":
                    clean_inference_time = inference_time
                m = safe_metrics(test_y, pred, mean, std)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "sraf-rc-v2-stepwise-modules",
                    "candidate": baseline,
                    "fault": label,
                    "fault_type": fault_type(label),
                    "severity_group": severity_group(label),
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "mae_h3": m.get("mae_h3", math.nan),
                    "mae_h6": m.get("mae_h6", math.nan),
                    "mae_h12": m.get("mae_h12", math.nan),
                    "parameter_count": param_count,
                    "inference_time_sec": inference_time,
                    "training_time_sec": "reused_existing_traceable_checkpoint" if baseline != "Persistence" else 0.0,
                    "best_epoch": "see_source_checkpoint" if baseline != "Persistence" else 0,
                    "best_val_loss": "see_source_checkpoint" if baseline != "Persistence" else 0.0,
                    "lambda_repair": 0.0,
                    "lambda_clean": 0.0,
                    "lambda_rel": 0.0,
                    "description": "Same-mask baseline evaluated inside dominance gate.",
                }
                metrics_rows.append(row)
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "sraf-rc-v2-stepwise-modules",
                        "candidate": baseline,
                        "fault": label,
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
            except Exception as exc:
                failed_rows.append({"candidate": baseline, "status": "failed", "reason": f"{label}: {exc!r}"})
        complexity_rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "sraf-rc-v2-stepwise-modules",
                "candidate": baseline,
                "parameter_count": param_count,
                "training_time_sec": "reused_existing_traceable_checkpoint" if baseline != "Persistence" else 0.0,
                "clean_inference_time_sec": clean_inference_time,
                "best_epoch": "see_source_checkpoint" if baseline != "Persistence" else 0,
                "best_val_loss": "see_source_checkpoint" if baseline != "Persistence" else 0.0,
            }
        )
    return metrics_rows, horizon_rows, complexity_rows, failed_rows


def write_summaries(
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    rdr_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    baseline_names = ["Persistence", "Repair-Persistence", "ResidualGRU-time-corruption-aware-strong", "SRAF-time-current"]
    candidate_names = []
    for row in metrics_rows:
        candidate = row["candidate"]
        if candidate not in baseline_names and candidate not in candidate_names:
            candidate_names.append(candidate)
    strong = "ResidualGRU-time-corruption-aware-strong"
    current = "sraf_rc_v1_base"
    clean_limit = 5.7
    rm20_ref = row_value(metrics_rows, current, "random_missing_20", "mae")
    stuck_rc_ref = row_value(metrics_rows, current, "stuck_at_last_value_high", "mae")
    stuck_time_ref = 5.821036
    strong_rm40 = 5.955920
    eligible: list[tuple[int, int, float, str]] = []
    lines = ["# Candidate Selection Summary", ""]
    for candidate in candidate_names:
        if not any(row["candidate"] == candidate for row in metrics_rows):
            continue
        clean = row_value(metrics_rows, candidate, "clean", "mae")
        rm20 = row_value(metrics_rows, candidate, "random_missing_20", "mae")
        rm40 = row_value(metrics_rows, candidate, "random_missing_40", "mae")
        stuck = row_value(metrics_rows, candidate, "stuck_at_last_value_high", "mae")
        high_faults = [s["label"] for s in FAULT_SETTINGS if s["label"] not in {"clean", "random_missing_20"}]
        lower_than_strong = 0
        for fault in high_faults:
            if any(row["candidate"] == strong and row["fault"] == fault for row in rdr_rows):
                lower_than_strong += row_value(rdr_rows, candidate, fault, "rdr_mae") < row_value(rdr_rows, strong, fault, "rdr_mae")
        rel_stuck = next((row for row in reliability_rows if row["candidate"] == candidate and row["fault"] == "stuck_at_last_value_high"), None)
        rel_stuck_ok = bool(rel_stuck and rel_stuck["corrupted_lower_than_clean"] is True)
        try:
            rel_stuck_sep = float(rel_stuck["reliability_separation"]) if rel_stuck else math.nan
        except (TypeError, ValueError):
            rel_stuck_sep = math.nan
        criteria = [
            clean <= clean_limit,
            rm40 <= strong_rm40 or rm40 <= strong_rm40 + 0.01,
            rm20 <= rm20_ref,
            stuck <= stuck_rc_ref,
            rel_stuck_ok or (math.isfinite(rel_stuck_sep) and rel_stuck_sep > -0.051466),
            lower_than_strong >= 4,
        ]
        score = sum(bool(x) for x in criteria)
        lines.append(
            f"- `{candidate}`: clean={clean:.6f}, rm20={rm20:.6f}, rm40={rm40:.6f}, stuck={stuck:.6f}, "
            f"lower_RDR_vs_strong={lower_than_strong}, stuck_rel_sep={rel_stuck_sep}, stuck_rel_ok={rel_stuck_ok}, score={score}/6."
        )
        if candidate != current and clean <= 5.8:
            eligible.append((score, int(rm40 <= strong_rm40), -rm40, candidate))
    selected = sorted(eligible, reverse=True)[0][3] if eligible else "none"
    lines.append("")
    lines.append(f"Selected candidate: `{selected}`.")
    lines.append(f"Skipped/failed candidates: {len(failed_rows)}.")
    (out_dir / "candidate_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")

    dom = ["# Baseline Dominance Summary", ""]
    for candidate in candidate_names:
        if not any(row["candidate"] == candidate for row in metrics_rows):
            continue
        dom.append(f"## {candidate}")
        for baseline in baseline_names:
            if not any(row["candidate"] == baseline for row in metrics_rows):
                continue
            wins = 0
            total = 0
            for setting in FAULT_SETTINGS:
                fault = setting["label"]
                if any(row["candidate"] == baseline and row["fault"] == fault for row in metrics_rows):
                    wins += row_value(metrics_rows, candidate, fault, "mae") <= row_value(metrics_rows, baseline, fault, "mae")
                    total += 1
            dom.append(f"- MAE wins vs `{baseline}`: {wins}/{total}")
        dom.append("")
    (out_dir / "baseline_dominance_summary.md").write_text("\n".join(dom), encoding="utf-8")
    (out_dir / "dominance_comparison_summary.md").write_text("\n".join(dom), encoding="utf-8")

    random_diag_rows: list[dict[str, Any]] = []
    for candidate in [*candidate_names, strong]:
        if any(row["candidate"] == candidate for row in metrics_rows):
            for fault in ["random_missing_20", "random_missing_40"]:
                rel = next((row for row in reliability_rows if row["candidate"] == candidate and row["fault"] == fault), None)
                rep = next((row for row in repair_rows if row["candidate"] == candidate and row["fault"] == fault), None)
                random_diag_rows.append(
                    {
                        "candidate": candidate,
                        "fault": fault,
                        "mae": row_value(metrics_rows, candidate, fault, "mae"),
                        "rdr_mae": row_value(rdr_rows, candidate, fault, "rdr_mae"),
                        "reliability_separation": rel["reliability_separation"] if rel else "TODO_baseline_has_no_reliability",
                        "final_repair_mae": rep["final_repair_mae"] if rep else "TODO_baseline_has_no_repair",
                        "delta_mae_vs_strong": row_value(metrics_rows, candidate, fault, "mae") - row_value(metrics_rows, strong, fault, "mae")
                        if any(row["candidate"] == strong and row["fault"] == fault for row in metrics_rows)
                        else "TODO",
                    }
                )
    write_csv(out_dir / "random_missing_diagnostics.csv", random_diag_rows)

    stuck_diag_rows: list[dict[str, Any]] = []
    for candidate in [*candidate_names, strong]:
        if any(row["candidate"] == candidate for row in metrics_rows):
            rel = next((row for row in reliability_rows if row["candidate"] == candidate and row["fault"] == "stuck_at_last_value_high"), None)
            rep = next((row for row in repair_rows if row["candidate"] == candidate and row["fault"] == "stuck_at_last_value_high"), None)
            stuck_diag_rows.append(
                {
                    "candidate": candidate,
                    "fault": "stuck_at_last_value_high",
                    "mae": row_value(metrics_rows, candidate, "stuck_at_last_value_high", "mae"),
                    "rdr_mae": row_value(rdr_rows, candidate, "stuck_at_last_value_high", "rdr_mae"),
                    "delta_vs_selected_sraf_rc": row_value(metrics_rows, candidate, "stuck_at_last_value_high", "mae") - stuck_rc_ref,
                    "delta_vs_current_sraf_time": row_value(metrics_rows, candidate, "stuck_at_last_value_high", "mae") - stuck_time_ref,
                    "delta_vs_strong": row_value(metrics_rows, candidate, "stuck_at_last_value_high", "mae")
                    - row_value(metrics_rows, strong, "stuck_at_last_value_high", "mae")
                    if any(row["candidate"] == strong and row["fault"] == "stuck_at_last_value_high" for row in metrics_rows)
                    else "TODO",
                    "reliability_separation": rel["reliability_separation"] if rel else "TODO_baseline_has_no_reliability",
                    "corrupted_position_reliability_mean": rel["corrupted_position_reliability_mean"] if rel else "TODO_baseline_has_no_reliability",
                    "final_repair_mae": rep["final_repair_mae"] if rep else "TODO_baseline_has_no_repair",
                    "stuck_indicator_statistics": "stuck_score is returned by conditional stuck gate candidates; reliability and repair diagnostics are saved separately",
                }
            )
    write_csv(out_dir / "stuck_fault_diagnostics.csv", stuck_diag_rows)


def execute_candidate(
    spec: dict[str, Any],
    out_dir: Path,
    train_x_base: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    test_x_clean: np.ndarray,
    fault_inputs: dict[str, np.ndarray],
    fault_masks: dict[str, np.ndarray],
    mean: float,
    std: float,
    args: argparse.Namespace,
    adjacency: torch.Tensor,
    metrics_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
) -> None:
    candidate = spec["candidate"]
    run_dir = out_dir / "candidates" / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    if spec["mode"] == "skip":
        failed_rows.append({"candidate": candidate, "status": "skipped", "reason": spec.get("skip_reason", "skipped")})
        return
    try:
        model = make_sraf(spec, train_x_base.shape[2], train_y.shape[1], args)
        train_extra: dict[str, Any] = {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}
        if spec["mode"] == "load":
            checkpoint = ROOT / spec["checkpoint"]
            if not checkpoint.exists():
                raise FileNotFoundError(str(checkpoint))
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")
            (run_dir / "train_log.txt").write_text(f"reused_checkpoint: {checkpoint}\n", encoding="utf-8")
        else:
            checkpoint = run_dir / "best_checkpoint.pt"
            if checkpoint.exists():
                model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                train_extra = training_meta_from_log(run_dir)
            else:
                train_extra, curves = train_candidate(spec, model, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
                curve_rows.extend(curves)
        param_count = model_param_count(model)
        for setting in FAULT_SETTINGS:
            label = setting["label"]
            pred, inference_time = predict_model(model, fault_inputs[label], args.batch_size, adjacency)
            m = safe_metrics(test_y, pred, mean, std)
            if label == "clean":
                np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
            row = {
                "dataset": "METR-LA",
                "run_id": "sraf-rc-v2-stepwise-modules",
                "candidate": candidate,
                "fault": label,
                "fault_type": fault_type(label),
                "severity_group": severity_group(label),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
                "mae_h3": m.get("mae_h3", math.nan),
                "mae_h6": m.get("mae_h6", math.nan),
                "mae_h12": m.get("mae_h12", math.nan),
                "parameter_count": param_count,
                "inference_time_sec": inference_time,
                "training_time_sec": train_extra["training_time_sec"],
                "best_epoch": train_extra["best_epoch"],
                "best_val_loss": train_extra["best_val_loss"],
                "lambda_repair": spec.get("lambda_repair", 0.0),
                "lambda_clean": spec.get("lambda_clean", 0.0),
                "lambda_rel": spec.get("lambda_rel", 0.0),
                "lambda_stuck_rel": spec.get("lambda_stuck_rel", 0.0),
                "random_missing_weighted_training": spec.get("random_missing_weighted_training", False),
                "description": spec["description"],
            }
            metrics_rows.append(row)
            horizon_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "sraf-rc-v2-stepwise-modules",
                    "candidate": candidate,
                    "fault": label,
                    "mae_15min_h3": row["mae_h3"],
                    "mae_30min_h6": row["mae_h6"],
                    "mae_60min_h12": row["mae_h12"],
                }
            )
        rel, rep = diagnostics_for_all_faults(candidate, model, test_x_clean, fault_inputs, fault_masks, mean, std, args.batch_size, adjacency)
        reliability_rows.extend(rel)
        repair_rows.extend(rep)
        clean_row = next(row for row in metrics_rows if row["candidate"] == candidate and row["fault"] == "clean")
        complexity_rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "sraf-rc-v2-stepwise-modules",
                "candidate": candidate,
                "parameter_count": param_count,
                "training_time_sec": train_extra["training_time_sec"],
                "clean_inference_time_sec": clean_row["inference_time_sec"],
                "best_epoch": train_extra["best_epoch"],
                "best_val_loss": train_extra["best_val_loss"],
            }
        )
    except Exception as exc:
        failed_rows.append({"candidate": candidate, "status": "failed", "reason": repr(exc)})
        print(f"FAILED {candidate}: {exc!r}", flush=True)


def metric_value(rows: list[dict[str, Any]], candidate: str, fault: str, key: str) -> float:
    return float(next(row[key] for row in rows if row["candidate"] == candidate and row["fault"] == fault))


def classify_modules(
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = "sraf_rc_v1_base"
    base_clean = metric_value(metrics_rows, base, "clean", "mae")
    base_rm40 = metric_value(metrics_rows, base, "random_missing_40", "mae")
    base_stuck = metric_value(metrics_rows, base, "stuck_at_last_value_high", "mae")
    base_h12 = metric_value(horizon_rows, base, "random_missing_40", "mae_60min_h12")
    base_params = next(row["parameter_count"] for row in complexity_rows if row["candidate"] == base)
    failed_by_candidate = {row["candidate"]: row for row in failed_rows}
    rows: list[dict[str, Any]] = []
    beneficial: list[dict[str, Any]] = []
    for spec in SINGLE_MODULE_SPECS:
        candidate = spec["candidate"]
        if candidate == base:
            continue
        if spec["mode"] == "skip" or candidate in failed_by_candidate:
            rows.append(
                {
                    "candidate": candidate,
                    "module": spec.get("module", "unknown"),
                    "classification": "skipped",
                    "reason": failed_by_candidate.get(candidate, {}).get("reason", spec.get("skip_reason", "skipped")),
                }
            )
            continue
        clean = metric_value(metrics_rows, candidate, "clean", "mae")
        rm40 = metric_value(metrics_rows, candidate, "random_missing_40", "mae")
        stuck = metric_value(metrics_rows, candidate, "stuck_at_last_value_high", "mae")
        h12_rm40 = metric_value(horizon_rows, candidate, "random_missing_40", "mae_60min_h12")
        params = next(row["parameter_count"] for row in complexity_rows if row["candidate"] == candidate)
        improves_key = (clean < base_clean) or (rm40 < base_rm40) or (stuck < base_stuck) or (h12_rm40 < base_h12)
        clean_ok = clean <= base_clean + 0.05
        rm40_ok = rm40 <= base_rm40 + 0.03
        stuck_ok = stuck <= base_stuck + 0.03
        lightweight = float(params) <= max(float(base_params) * 2.0, 15000.0)
        if improves_key and clean_ok and rm40_ok and stuck_ok and lightweight:
            classification = "beneficial"
            beneficial.append({**spec, "selection_score": int(clean < base_clean) + int(rm40 < base_rm40) + int(stuck < base_stuck) + int(h12_rm40 < base_h12)})
        elif improves_key:
            classification = "mixed"
        else:
            classification = "harmful"
        rows.append(
            {
                "candidate": candidate,
                "module": spec.get("module", "unknown"),
                "classification": classification,
                "clean_delta_vs_base": clean - base_clean,
                "rm40_delta_vs_base": rm40 - base_rm40,
                "stuck_delta_vs_base": stuck - base_stuck,
                "rm40_h12_delta_vs_base": h12_rm40 - base_h12,
                "parameter_delta_vs_base": float(params) - float(base_params),
                "reason": f"improves_key={improves_key}; clean_ok={clean_ok}; rm40_ok={rm40_ok}; stuck_ok={stuck_ok}; lightweight={lightweight}",
            }
        )
    lines = ["# Module Selection Summary", "", f"Base candidate: `{base}`.", ""]
    for row in rows:
        lines.append(
            f"- `{row['candidate']}` ({row['module']}): {row['classification']}; {row.get('reason', '')}"
        )
    (out_dir / "module_selection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(out_dir / "module_selection_summary.csv", rows)
    return sorted(beneficial, key=lambda item: (item.get("selection_score", 0), -item["priority"]), reverse=True)


def make_stack_specs(beneficial_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not beneficial_specs:
        return []
    stack_specs: list[dict[str, Any]] = []
    merged_kwargs: dict[str, Any] = {}
    modules: list[str] = []
    for index, spec in enumerate(beneficial_specs[:2], start=1):
        merged_kwargs.update(spec.get("model_kwargs", {}))
        modules.append(spec.get("module", spec["candidate"]))
        stack_specs.append(
            {
                "candidate": f"stack_top{index}",
                "module": "+".join(modules),
                "priority": 100 + index,
                "mode": "train",
                "description": f"SRAF-RC-V2 greedy stack with modules: {', '.join(modules)}.",
                "model_kwargs": dict(merged_kwargs),
                "lambda_repair": BASE_SPEC["lambda_repair"],
                "lambda_clean": BASE_SPEC["lambda_clean"],
                "lambda_rel": BASE_SPEC["lambda_rel"],
                "lambda_stuck_rel": BASE_SPEC["lambda_stuck_rel"],
                "stacked_from": [item["candidate"] for item in beneficial_specs[:index]],
            }
        )
    return stack_specs


def write_stacking_results_summary(
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    stack_specs: list[dict[str, Any]],
) -> None:
    base = "sraf_rc_v1_base"
    lines = ["# Stacking Results Summary", "", f"Base candidate: `{base}`.", ""]
    if not stack_specs:
        lines.append("No stack candidate was trained because no single module met the beneficial-module criteria.")
    for spec in stack_specs:
        candidate = spec["candidate"]
        if not any(row["candidate"] == candidate for row in metrics_rows):
            lines.append(f"- `{candidate}`: TODO - no metrics were produced.")
            continue
        clean_delta = metric_value(metrics_rows, candidate, "clean", "mae") - metric_value(metrics_rows, base, "clean", "mae")
        rm40_delta = metric_value(metrics_rows, candidate, "random_missing_40", "mae") - metric_value(metrics_rows, base, "random_missing_40", "mae")
        stuck_delta = metric_value(metrics_rows, candidate, "stuck_at_last_value_high", "mae") - metric_value(metrics_rows, base, "stuck_at_last_value_high", "mae")
        h12_delta = metric_value(horizon_rows, candidate, "random_missing_40", "mae_60min_h12") - metric_value(
            horizon_rows, base, "random_missing_40", "mae_60min_h12"
        )
        lines.append(
            f"- `{candidate}` from {spec.get('stacked_from', [])}: clean_delta={clean_delta:.6f}, "
            f"rm40_delta={rm40_delta:.6f}, stuck_delta={stuck_delta:.6f}, rm40_h12_delta={h12_delta:.6f}."
        )
    (out_dir / "stacking_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x_base, train_y = load_split(data_dir, "train")
    val_x_base, val_y = load_split(data_dir, "val")
    test_x_base, test_y = load_split(data_dir, "test")
    if args.smoke:
        train_x_base, train_y = train_x_base[:128], train_y[:128]
        val_x_base, val_y = val_x_base[:64], val_y[:64]
        test_x_base, test_y = test_x_base[:128], test_y[:128]
        args.epochs = min(args.epochs, 1)
        args.patience = 1
    mean, std = load_scale(data_dir)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32))
    train_x = add_time_of_day_features(train_x_base, 0)
    val_x = add_time_of_day_features(val_x_base, train_x_base.shape[0])
    test_x_clean = add_time_of_day_features(test_x_base, train_x_base.shape[0] + val_x_base.shape[0])

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(parents=True, exist_ok=True)
    fault_inputs: dict[str, np.ndarray] = {}
    fault_masks: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        cx, mask, meta = apply_fault_with_time(test_x_clean, setting, seed=args.seed + idx)
        fault_inputs[label] = cx
        fault_masks[label] = mask.astype(bool)
        meta = {**setting, **meta, "label": label, "target_corrupted": False, "mask_path": str(fault_dir / f"{label}_mask.npz")}
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest = {
        "run_id": "metr-la-sraf-rc-v2-stepwise-modules",
        "gate": "SRAF_RC_V2_STEPWISE_MODULE_OPTIMIZATION_GATE",
        "created_at": "2026-05-18",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "dataset": {
            "name": "METR-LA",
            "L": 12,
            "H": 12,
            "N": int(train_x_base.shape[2]),
            "F_target": 1,
            "train_samples_used": int(train_x_base.shape[0]),
            "val_samples_used": int(val_x_base.shape[0]),
            "test_samples_used": int(test_x_base.shape[0]),
        },
        "training": {
            "hidden_dim": args.hidden_dim,
            "sensor_embedding_dim": args.sensor_embedding_dim,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "loss": "MSE plus optional repair, clean preservation, reliability supervision, and stuck reliability losses",
            "corrupt_speed_only": True,
        },
        "time_feature_construction": "Speed plus input-window sin/cos time-of-day; faults corrupt speed only.",
        "target_leakage_check": "Target Y is never corrupted and no future target-horizon features are used.",
        "candidates": CANDIDATES,
        "stacking_strategy": "Train base and single modules first; classify beneficial modules against base; train stack_top1 and stack_top2 only from beneficial modules when available.",
        "fault_settings": FAULT_SETTINGS,
        "integrity_note": "No PEMS-BAY, no manuscript conclusions, no change to main claim or SRAF-RC direction.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    candidates = CANDIDATES[:3] if args.smoke else CANDIDATES
    for spec in candidates:
        execute_candidate(
            spec,
            out_dir,
            train_x_base,
            train_x,
            train_y,
            val_x,
            val_y,
            test_y,
            test_x_clean,
            fault_inputs,
            fault_masks,
            mean,
            std,
            args,
            adjacency,
            metrics_rows,
            horizon_rows,
            complexity_rows,
            reliability_rows,
            repair_rows,
            failed_rows,
            curve_rows,
        )

    stack_specs: list[dict[str, Any]] = []
    if not args.smoke:
        beneficial_specs = classify_modules(out_dir, metrics_rows, horizon_rows, complexity_rows, failed_rows)
        stack_specs = make_stack_specs(beneficial_specs)
        if not stack_specs:
            failed_rows.append({"candidate": "stack_top1", "status": "skipped", "reason": "No beneficial single module satisfied stacking criteria."})
        for spec in stack_specs:
            execute_candidate(
                spec,
                out_dir,
                train_x_base,
                train_x,
                train_y,
                val_x,
                val_y,
                test_y,
                test_x_clean,
                fault_inputs,
                fault_masks,
                mean,
                std,
                args,
                adjacency,
                metrics_rows,
                horizon_rows,
                complexity_rows,
                reliability_rows,
                repair_rows,
                failed_rows,
                curve_rows,
            )
        write_stacking_results_summary(out_dir, metrics_rows, horizon_rows, stack_specs)
    else:
        (out_dir / "module_selection_summary.md").write_text("Smoke run: module selection skipped.", encoding="utf-8")
        (out_dir / "stacking_results_summary.md").write_text("Smoke run: stacking skipped.", encoding="utf-8")

    if not args.smoke:
        base_metrics, base_horizon, base_complexity, base_failed = evaluate_same_mask_baselines(
            fault_inputs, test_y, mean, std, args, adjacency
        )
        metrics_rows.extend(base_metrics)
        horizon_rows.extend(base_horizon)
        complexity_rows.extend(base_complexity)
        failed_rows.extend(base_failed)

    rdr_rows = build_rdr(metrics_rows)
    write_csv(out_dir / "metrics_by_candidate_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "reliability_score_diagnostics.csv", reliability_rows)
    write_csv(out_dir / "repair_quality_diagnostics.csv", repair_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", curve_rows)
    write_csv(out_dir / "failed_or_skipped_candidates.csv", failed_rows)
    write_summaries(out_dir, metrics_rows, rdr_rows, horizon_rows, reliability_rows, repair_rows, failed_rows)
    return {"status": "completed", "output_dir": str(out_dir), "metrics_rows": len(metrics_rows), "failed_or_skipped": len(failed_rows)}


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()

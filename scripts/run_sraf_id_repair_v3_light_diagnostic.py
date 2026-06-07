"""Diagnostic runner for SRAF-ID repair-v3-light.

This is not a formal manuscript experiment. It trains/evaluates a small set of
repair-layer variants under capped splits, compares against a same-budget
current SRAF-ID reference, and records complexity/latency and repair statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    clean_input_for_backbone,
    corruption_aware_batch,
    eval_loss,
    fixed_corrupt_val_sets,
    iter_batches,
    make_loss,
    model_param_count,
)
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault, resolve_device  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import (  # noqa: E402
    FAULT_SPECS,
    load_payload,
    safe_metrics,
)
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import (  # noqa: E402
    build_v2,
    build_official_stid,
    predict_v2,
    train_v2,
)
from src.models.strong_backbones_v3 import SRAFOfficialStyleSTIDWrapperV3Light  # noqa: E402


STAGE = "SRAF_ID_REPAIR_V3_LIGHT_DIAGNOSTIC_GATE"
FAULTS = [
    "continuous_outage_24",
    "random_missing_40",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
]
SEEDS = [42, 43, 44]
DATASETS = ["METR-LA", "PEMS-BAY"]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tod_profile(train_x: np.ndarray) -> torch.Tensor:
    speed = train_x[..., 0]
    tod_idx = np.floor(train_x[..., 1] * 288.0).astype(np.int64).clip(0, 287)
    samples, length, sensors = speed.shape
    sums = np.zeros((288, sensors), dtype=np.float64)
    counts = np.zeros((288, sensors), dtype=np.float64)
    flat_speed = speed.reshape(samples * length, sensors)
    flat_tod = tod_idx.reshape(samples * length, sensors)
    for n in range(sensors):
        np.add.at(sums[:, n], flat_tod[:, n], flat_speed[:, n])
        np.add.at(counts[:, n], flat_tod[:, n], 1.0)
    sensor_mean = np.nanmean(speed, axis=(0, 1))
    profile = np.where(counts > 0, sums / np.maximum(counts, 1.0), sensor_mean.reshape(1, sensors))
    profile = np.nan_to_num(profile, nan=0.0).astype(np.float32)
    return torch.from_numpy(profile[..., None])


def build_v3_model(payload: dict[str, Any], cfg: dict[str, Any]) -> SRAFOfficialStyleSTIDWrapperV3Light:
    sensors = payload["train_x"].shape[2]
    input_length = payload["train_x"].shape[1]
    horizon = payload["train_y"].shape[1]
    return SRAFOfficialStyleSTIDWrapperV3Light(
        sensors=sensors,
        backbone=build_official_stid(sensors, input_length, horizon),
        tod_profile=build_tod_profile(payload["train_x"]),
        topk=int(cfg.get("topk", 5)),
        fusion_hidden_dim=int(cfg.get("fusion_hidden", 32)),
        observed_input_blend=float(cfg.get("observed_blend", 0.5)),
    )


def make_current_sraf_id(payload: dict[str, Any]) -> nn.Module:
    sensors = payload["train_x"].shape[2]
    input_length = payload["train_x"].shape[1]
    horizon = payload["train_y"].shape[1]
    cfg = {
        "rel_hidden": 64,
        "alpha_hidden": 16,
        "adaptive_alpha": True,
        "stuck_features": True,
        "flatness": False,
        "second_delta": True,
        "repair_disagreement": True,
        "base_rel_only": False,
    }
    model = build_v2(sensors, input_length, horizon, cfg)
    model.repairer.use_reliability_gate = False
    return model


def get_faulted(x: np.ndarray, fault_label: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = FAULT_SPECS[fault_label]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    x_fault = x.copy()
    x_fault[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return x_fault.astype(np.float32), mask.astype(np.float32), observed


def predict_sraf(model: nn.Module, x: np.ndarray, observed: np.ndarray, batch_size: int, device: torch.device, adjacency: torch.Tensor, return_components: bool = False) -> tuple[np.ndarray, float, dict[str, np.ndarray] | None]:
    model.eval()
    preds: list[np.ndarray] = []
    comp_rows: dict[str, list[np.ndarray]] = {"weights": [], "repair_disp": []}
    st = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(clean_input_for_backbone(x[i : i + batch_size])).to(device)
            obs = torch.from_numpy(observed[i : i + batch_size].astype(np.float32)).to(device)
            if return_components:
                pred, comps = model(xb, adjacency=adjacency, observed_mask=obs, return_components=True)
                weights = comps.get("candidate_weights")
                if weights is not None:
                    comp_rows["weights"].append(weights.detach().cpu().numpy())
                disp = torch.abs(comps["repaired_input_speed"] - comps["x_filled"])
                comp_rows["repair_disp"].append(disp.detach().cpu().numpy())
            else:
                pred = model(xb, adjacency=adjacency, observed_mask=obs)
            preds.append(pred.detach().cpu().numpy())
    comps_out = None
    if return_components:
        comps_out = {}
        if comp_rows["weights"]:
            comps_out["weights"] = np.concatenate(comp_rows["weights"], axis=0)
        if comp_rows["repair_disp"]:
            comps_out["repair_disp"] = np.concatenate(comp_rows["repair_disp"], axis=0)
    return np.concatenate(preds, axis=0), perf_counter() - st, comps_out


def train_v3(
    model: SRAFOfficialStyleSTIDWrapperV3Light,
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    adjacency: torch.Tensor,
    lambda_delta: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    fixed_val = fixed_corrupt_val_sets(val_x, args.seed)
    settings = [FAULT_SPECS[f] for f in FAULTS]
    best_state = None
    best_val = math.inf
    best_epoch = 0
    no_imp = 0
    rows: list[dict[str, Any]] = []
    step = 0
    start = perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        forecast_losses: list[float] = []
        repair_losses: list[float] = []
        delta_losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=ep):
            setting = settings[step % len(settings)]
            x_corrupt, mask, observed = corruption_aware_batch(xb, setting, args.seed + step)
            xb_t = torch.from_numpy(clean_input_for_backbone(x_corrupt)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            observed_t = torch.from_numpy(observed.astype(np.float32)).to(device)
            clean_speed = torch.from_numpy(xb[..., :1].astype(np.float32)).to(device)
            pred, comps = model(xb_t, adjacency=adjacency, observed_mask=observed_t, return_components=True)
            forecast = loss_fn(pred, yb_t)
            repair = torch.sum(torch.abs(comps["repaired_input_speed"] - clean_speed) * mask_t) / mask_t.sum().clamp_min(1.0)
            if lambda_delta > 0:
                delta_rep = comps["repaired_input_speed"][:, 1:] - comps["repaired_input_speed"][:, :-1]
                delta_clean = clean_speed[:, 1:] - clean_speed[:, :-1]
                delta_mask = torch.maximum(mask_t[:, 1:], mask_t[:, :-1])
                delta_loss = torch.sum(torch.abs(delta_rep - delta_clean) * delta_mask) / delta_mask.sum().clamp_min(1.0)
            else:
                delta_loss = torch.tensor(0.0, device=device)
            total = forecast + args.lambda_repair * repair + lambda_delta * delta_loss
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(total.detach().cpu()))
            forecast_losses.append(float(forecast.detach().cpu()))
            repair_losses.append(float(repair.detach().cpu()))
            delta_losses.append(float(delta_loss.detach().cpu()))
            step += 1
        clean_val = eval_loss_sraf_adj(model, val_x, val_y, args.batch_size, device, loss_fn, adjacency)
        corrupt_vals = [eval_loss_sraf_adj(model, vx, val_y, args.batch_size, device, loss_fn, adjacency, observed_mask=obs) for vx, obs, _ in fixed_val]
        selection = 0.5 * clean_val + 0.5 * float(np.mean(corrupt_vals))
        sch.step(selection)
        if selection < best_val - 1.0e-6:
            best_val = selection
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        rows.append(
            {
                "model": model_name,
                "epoch": ep,
                "train_loss": float(np.mean(losses)),
                "forecast_loss": float(np.mean(forecast_losses)),
                "repair_loss": float(np.mean(repair_losses)),
                "delta_loss": float(np.mean(delta_losses)),
                "selection_val_loss": selection,
                "best_selection_val_loss": best_val,
            }
        )
        if no_imp >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def eval_loss_sraf_adj(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    loss_fn: nn.Module,
    adjacency: torch.Tensor,
    observed_mask: np.ndarray | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(clean_input_for_backbone(x[i : i + batch_size])).to(device)
            yb = torch.from_numpy(y[i : i + batch_size].astype(np.float32)).to(device)
            om = None if observed_mask is None else torch.from_numpy(observed_mask[i : i + batch_size].astype(np.float32)).to(device)
            pred = model(xb, adjacency=adjacency, observed_mask=om)
            losses.append(float(loss_fn(pred, yb).detach().cpu()))
    return float(np.mean(losses))


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"name": "C0_current_sraf_id_budget", "family": "current", "topk": 5, "fusion_hidden": 32, "observed_blend": 0.5, "lambda_delta": 0.0},
        {"name": "R1_v3_bidir_topk5_profile_softmax", "family": "v3", "topk": 5, "fusion_hidden": 32, "observed_blend": 0.5, "lambda_delta": 0.0},
        {"name": "R2_v3_topk8_profile_stronger_repair", "family": "v3", "topk": 8, "fusion_hidden": 32, "observed_blend": 0.25, "lambda_delta": 0.0},
        {"name": "R3_v3_topk5_profile_delta_loss", "family": "v3", "topk": 5, "fusion_hidden": 32, "observed_blend": 0.5, "lambda_delta": 0.01},
    ]


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**job["args"])
    dataset = job["dataset"]
    seed = int(job["seed"])
    args.seed = seed
    cfg = job["candidate"]
    out = Path(args.output_dir)
    device = resolve_device(args.device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    payload = load_payload(dataset, args.train_limit, args.val_limit, args.test_limit)
    adj_t = torch.from_numpy(payload["adj"]).to(device)
    run_dir = out / "runs" / dataset.lower().replace("-", "_") / f"seed_{seed}" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_snapshot.json"
    cfg_path.write_text(json.dumps({"stage": STAGE, "dataset": dataset, "seed": seed, "candidate": cfg, "args": job["args"]}, indent=2), encoding="utf-8")
    st = perf_counter()
    if cfg["family"] == "current":
        model = make_current_sraf_id(payload)
        meta, curves = train_v2(model, cfg["name"], payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj_t)
        predict_fn = "v2"
    else:
        model = build_v3_model(payload, cfg)
        meta, curves = train_v3(model, cfg["name"], payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj_t, lambda_delta=float(cfg.get("lambda_delta", 0.0)))
        predict_fn = "v3"
    write_csv(run_dir / "training_curves.csv", curves)
    metric_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for idx, fault in enumerate(FAULTS):
        x_fault, mask, observed = get_faulted(payload["test_x"], fault, seed + idx)
        if predict_fn == "v2":
            pred, latency = predict_v2(model, x_fault, observed, args.batch_size, device, adj_t)
            comps = None
        else:
            pred, latency, comps = predict_sraf(model, x_fault, observed, args.batch_size, device, adj_t, return_components=True)
        met = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        row = {
            "dataset": dataset,
            "seed": seed,
            "candidate": cfg["name"],
            "fault": fault,
            "mae": float(met["mae"]),
            "rmse": float(met["rmse"]),
            "h3_mae": float(met["mae_h3"]),
            "h6_mae": float(met["mae_h6"]),
            "h12_mae": float(met["mae_h12"]),
            "latency_sec": float(latency),
            "params": int(model_param_count(model)),
            "training_time_sec": float(meta["training_time_sec"]),
            "best_epoch": int(meta["best_epoch"]),
            "status": "completed",
        }
        metric_rows.append(row)
        if comps is not None:
            stats = {
                "dataset": dataset,
                "seed": seed,
                "candidate": cfg["name"],
                "fault": fault,
                "repair_displacement_mean": float(np.mean(comps.get("repair_disp", np.array([np.nan])))),
            }
            if "weights" in comps:
                weights = comps["weights"]
                for i, name in enumerate(["temporal", "spatial", "profile"]):
                    stats[f"weight_{name}_mean"] = float(np.mean(weights[..., i]))
            comp_rows.append(stats)
    write_csv(run_dir / "metrics.csv", metric_rows)
    if comp_rows:
        write_csv(run_dir / "repair_component_stats.csv", comp_rows)
    manifest = {
        "stage": STAGE,
        "dataset": dataset,
        "seed": seed,
        "candidate": cfg["name"],
        "status": "completed",
        "runtime_sec": perf_counter() - st,
        "output_path": str(run_dir),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "metrics": metric_rows, "components": comp_rows}


def aggregate(out: Path, expected_jobs: int) -> dict[str, Any]:
    metric_files = list((out / "runs").glob("**/metrics.csv"))
    all_metrics = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    all_metrics.to_csv(out / "diagnostic_per_seed_metrics.csv", index=False)
    comp_files = list((out / "runs").glob("**/repair_component_stats.csv"))
    if comp_files:
        pd.concat([pd.read_csv(p) for p in comp_files], ignore_index=True).to_csv(out / "repair_component_statistics.csv", index=False)
    if all_metrics.empty:
        return {"status": "FAIL", "completed_jobs": 0}
    agg = all_metrics.groupby(["dataset", "candidate", "fault"], as_index=False).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
        h3_mae_mean=("h3_mae", "mean"),
        h6_mae_mean=("h6_mae", "mean"),
        h12_mae_mean=("h12_mae", "mean"),
        latency_mean_sec=("latency_sec", "mean"),
        params=("params", "first"),
        training_time_mean_sec=("training_time_sec", "mean"),
        completed_seeds=("seed", "nunique"),
    )
    agg.to_csv(out / "diagnostic_aggregate_metrics.csv", index=False)
    cur = agg[agg.candidate == "C0_current_sraf_id_budget"][["dataset", "fault", "mae_mean", "h12_mae_mean"]].rename(columns={"mae_mean": "current_mae", "h12_mae_mean": "current_h12"})
    cmp = agg.merge(cur, on=["dataset", "fault"], how="left")
    cmp["gain_vs_current_pct"] = (cmp["current_mae"] - cmp["mae_mean"]) / cmp["current_mae"] * 100.0
    cmp["h12_gain_vs_current_pct"] = (cmp["current_h12"] - cmp["h12_mae_mean"]) / cmp["current_h12"] * 100.0
    cmp.to_csv(out / "comparison_vs_current_sraf_id.csv", index=False)
    avg = cmp[cmp.candidate != "C0_current_sraf_id_budget"].groupby("candidate", as_index=False).agg(
        avg_gain_vs_current_pct=("gain_vs_current_pct", "mean"),
        avg_mae=("mae_mean", "mean"),
        avg_latency_sec=("latency_mean_sec", "mean"),
        params=("params", "first"),
    )
    avg = avg.sort_values(["avg_gain_vs_current_pct", "avg_mae"], ascending=[False, True])
    avg.to_csv(out / "candidate_ranking.csv", index=False)
    make_report(out, all_metrics, agg, cmp, avg, expected_jobs)
    return {"status": "PASS" if len(metric_files) == expected_jobs else "PARTIAL", "completed_jobs": len(metric_files)}


def make_report(out: Path, per_seed: pd.DataFrame, agg: pd.DataFrame, cmp: pd.DataFrame, ranking: pd.DataFrame, expected_jobs: int) -> None:
    status = "PASS" if per_seed[["dataset", "seed", "candidate"]].drop_duplicates().shape[0] == expected_jobs else "PARTIAL"
    best = ranking.iloc[0].candidate if not ranking.empty else "NONE"
    lines = [
        "# SRAF_ID_REPAIR_V3_LIGHT_DIAGNOSTIC_REPORT",
        "",
        "## 1. Stage Metadata",
        f"- stage: `{STAGE}`",
        f"- status: `{status}`",
        f"- timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        "- formal/manuscript results modified: `NO`",
        "- model path: diagnostic-only v3-light classes added separately from frozen v1/v2",
        f"- expected training jobs: `{expected_jobs}`",
        f"- completed training jobs: `{per_seed[['dataset', 'seed', 'candidate']].drop_duplicates().shape[0]}`",
        f"- output directory: `{out}`",
        "",
        "## 2. Implementation Summary",
        "- temporal optimization: bidirectional temporal repair candidate.",
        "- spatial optimization: observed-aware top-k adjacency repair candidate.",
        "- MLP optimization: softmax fusion MLP over temporal/spatial/train-only time-of-day profile candidates.",
        "- profile source: train split only, grouped by time-of-day and sensor.",
        "- identity preservation: tod/dow bypass repair and are concatenated unchanged before ID-MLP.",
        "",
        "## 3. Code Check",
        "- Static compile and shape-smoke checks are recorded in `code_check_report.md`.",
        "",
        "## 4. Diagnostic Results",
        "- Per-seed metrics: `diagnostic_per_seed_metrics.csv`.",
        "- Aggregates: `diagnostic_aggregate_metrics.csv`.",
        "- Comparison vs current SRAF-ID same-budget reference: `comparison_vs_current_sraf_id.csv`.",
        "",
        "## 5. Candidate Ranking",
        f"- best candidate by average gain vs current same-budget SRAF-ID: `{best}`.",
    ]
    if not ranking.empty:
        for row in ranking.itertuples():
            lines.append(f"- `{row.candidate}`: avg_gain_vs_current=`{row.avg_gain_vs_current_pct:.3f}%`, avg_mae=`{row.avg_mae:.6f}`, latency=`{row.avg_latency_sec:.4f}s`, params=`{int(row.params)}`")
    pems_outage = cmp[(cmp.dataset == "PEMS-BAY") & (cmp.fault == "continuous_outage_24") & (cmp.candidate != "C0_current_sraf_id_budget")]
    lines.extend(["", "## 6. PEMS-BAY Continuous Outage Focus"])
    for row in pems_outage.sort_values("gain_vs_current_pct", ascending=False).itertuples():
        lines.append(f"- `{row.candidate}`: MAE=`{row.mae_mean:.6f}`, gain_vs_current=`{row.gain_vs_current_pct:.3f}%`, h12_gain=`{row.h12_gain_vs_current_pct:.3f}%`.")
    lines.extend(
        [
            "",
            "## 7. Complexity and Latency",
            "- Complexity/latency columns are included in `diagnostic_aggregate_metrics.csv`.",
            "- This is capped diagnostic training, not formal 10-seed evidence.",
            "",
            "## 8. Decision",
            "- SHOULD_UPDATE_MANUSCRIPT_NOW: `NO`",
            "- SHOULD_CONSIDER_FORMAL_RERUN: `YES` if the top candidate has positive average gain and no broad regressions after mentor review.",
            "- NEXT_ACTION: review candidate ranking and decide whether to authorize a formal rerun for the selected repair-v3-light candidate.",
        ]
    )
    (out / "SRAF_ID_REPAIR_V3_LIGHT_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/sraf_id_repair_v3_light_diagnostic")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--train-limit", type=int, default=4096)
    p.add_argument("--val-limit", type=int, default=1024)
    p.add_argument("--test-limit", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    candidates = candidate_grid()
    jobs = [{"dataset": ds, "seed": seed, "candidate": cand, "args": vars(args)} for ds in DATASETS for seed in SEEDS for cand in candidates]
    plan = {
        "stage": STAGE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": DATASETS,
        "faults": FAULTS,
        "seeds": SEEDS,
        "candidates": [c["name"] for c in candidates],
        "expected_training_jobs": len(jobs),
        "max_workers": args.max_workers,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "diagnostic_only_not_formal": True,
    }
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    manifest_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(run_job, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                manifest_rows.append(result["manifest"])
                print(f"completed {result['manifest']['dataset']} seed={result['manifest']['seed']} {result['manifest']['candidate']}", flush=True)
            except Exception as exc:
                manifest_rows.append({"status": "failed", "error": repr(exc), "runtime_sec": math.nan})
                print(f"failed {repr(exc)}", flush=True)
    write_csv(out / "run_manifest.csv", manifest_rows)
    summary = aggregate(out, len(jobs))
    print("TERMINAL SUMMARY", flush=True)
    print(f"expected training jobs: {len(jobs)}", flush=True)
    print(f"completed training jobs: {summary['completed_jobs']}", flush=True)
    print(f"status: {summary['status']}", flush=True)


if __name__ == "__main__":
    main()

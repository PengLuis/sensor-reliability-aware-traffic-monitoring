"""Seed-42 quick smoke for current SRAF-ID vs softmax-fusion SRAF-ID."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import model_param_count  # noqa: E402
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault, resolve_device  # noqa: E402
from scripts.run_sraf_id_repair_factor_ablation import build_factor_model  # noqa: E402
from scripts.run_sraf_id_repair_v3_light_diagnostic import make_current_sraf_id, predict_sraf, train_v3, write_csv  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload, safe_metrics  # noqa: E402
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import predict_v2, train_v2  # noqa: E402


STAGE = "SRAF_ID_SOFTMAX_FUSION_SEED42_SMOKE"
DATASETS = ["METR-LA", "PEMS-BAY"]
SEED = 42
FAULT_SPECS = {
    "clean": {"fault": "clean", "label": "clean"},
    "random_missing_20": {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20"},
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "continuous_outage_24": {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
}
FAULTS = list(FAULT_SPECS)


def get_faulted(x: np.ndarray, fault: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fault == "clean":
        mask = np.zeros_like(x[..., :1], dtype=np.float32)
        observed = np.ones_like(mask, dtype=np.float32)
        return x.astype(np.float32), mask, observed
    spec = FAULT_SPECS[fault]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    x_fault = x.copy()
    x_fault[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return x_fault.astype(np.float32), mask.astype(np.float32), observed


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**job["args"])
    args.seed = SEED
    dataset = job["dataset"]
    model_name = job["model"]
    out = Path(args.output_dir)
    device = resolve_device(args.device)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    payload = load_payload(dataset, args.train_limit, args.val_limit, args.test_limit)
    adj_t = torch.from_numpy(payload["adj"]).to(device)
    run_dir = out / "runs" / dataset.lower().replace("-", "_") / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    if model_name == "SRAF-ID-current":
        model = make_current_sraf_id(payload)
        meta, curves = train_v2(model, model_name, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj_t)
        kind = "current"
    else:
        cfg = {"name": "A3_mlp_only_softmax_no_profile", "family": "factor", "temporal_mode": "basic", "spatial_mode": "adjacency", "fusion_mode": "softmax", "use_profile": False, "topk": 5, "fixed_profile_weight": 0.0}
        model = build_factor_model(payload, cfg)
        meta, curves = train_v3(model, model_name, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj_t)
        kind = "softmax"
    write_csv(run_dir / "training_curves.csv", curves)
    rows = []
    comp_rows = []
    for idx, fault in enumerate(FAULTS):
        x_fault, mask, observed = get_faulted(payload["test_x"], fault, SEED + idx)
        if kind == "current":
            pred, latency = predict_v2(model, x_fault, observed, args.batch_size, device, adj_t)
            comps = None
        else:
            pred, latency, comps = predict_sraf(model, x_fault, observed, args.batch_size, device, adj_t, return_components=True)
        met = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        rows.append(
            {
                "dataset": dataset,
                "seed": SEED,
                "model": model_name,
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
            }
        )
        if comps and "weights" in comps:
            comp_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "fault": fault,
                    "weight_temporal_mean": float(np.mean(comps["weights"][..., 0])),
                    "weight_spatial_mean": float(np.mean(comps["weights"][..., 1])),
                    "repair_displacement_mean": float(np.mean(comps["repair_disp"])),
                }
            )
    write_csv(run_dir / "metrics.csv", rows)
    if comp_rows:
        write_csv(run_dir / "fusion_stats.csv", comp_rows)
    manifest = {"stage": STAGE, "dataset": dataset, "model": model_name, "seed": SEED, "status": "completed", "runtime_sec": perf_counter() - start}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "metrics": rows, "fusion": comp_rows}


def aggregate(out: Path) -> None:
    metrics = pd.concat([pd.read_csv(p) for p in (out / "runs").glob("**/metrics.csv")], ignore_index=True)
    metrics.to_csv(out / "seed42_per_fault_metrics.csv", index=False)
    cur = metrics[metrics.model == "SRAF-ID-current"][["dataset", "fault", "mae", "h12_mae"]].rename(columns={"mae": "current_mae", "h12_mae": "current_h12"})
    cmp = metrics.merge(cur, on=["dataset", "fault"], how="left")
    cmp["gain_vs_current_pct"] = (cmp["current_mae"] - cmp["mae"]) / cmp["current_mae"] * 100.0
    cmp["h12_gain_vs_current_pct"] = (cmp["current_h12"] - cmp["h12_mae"]) / cmp["current_h12"] * 100.0
    cmp.to_csv(out / "seed42_comparison_vs_current.csv", index=False)
    faulty = cmp[cmp.fault != "clean"].groupby(["dataset", "model"], as_index=False).agg(avg_faulty_mae=("mae", "mean"), avg_faulty_gain_vs_current_pct=("gain_vs_current_pct", "mean"), latency_mean_sec=("latency_sec", "mean"), params=("params", "first"))
    faulty.to_csv(out / "seed42_avg_faulty_summary.csv", index=False)
    fusion_files = list((out / "runs").glob("**/fusion_stats.csv"))
    if fusion_files:
        pd.concat([pd.read_csv(p) for p in fusion_files], ignore_index=True).to_csv(out / "seed42_fusion_stats.csv", index=False)
    soft = cmp[cmp.model == "SRAF-ID-softmax-fusion"]
    lines = [
        "# SRAF_ID_SOFTMAX_FUSION_SEED42_SMOKE_REPORT",
        "",
        f"- stage: `{STAGE}`",
        "- status: `PASS`",
        f"- timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        "- seed: `42`",
        "- diagnostic only: `YES`",
        "- formal/manuscript results modified: `NO`",
        "",
        "## Average Faulty Summary",
    ]
    for r in faulty.itertuples():
        lines.append(f"- `{r.dataset}` / `{r.model}`: avg_faulty_mae=`{r.avg_faulty_mae:.6f}`, avg_gain_vs_current=`{r.avg_faulty_gain_vs_current_pct:.3f}%`, latency=`{r.latency_mean_sec:.4f}s`, params=`{int(r.params)}`.")
    lines.extend(["", "## Per-Fault Softmax Gain vs Current"])
    for r in soft.itertuples():
        lines.append(f"- `{r.dataset}` `{r.fault}`: MAE=`{r.mae:.6f}`, gain=`{r.gain_vs_current_pct:.3f}%`, h12_gain=`{r.h12_gain_vs_current_pct:.3f}%`.")
    (out / "SRAF_ID_SOFTMAX_FUSION_SEED42_SMOKE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/sraf_id_softmax_fusion_seed42_smoke")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"dataset": ds, "model": model, "args": vars(args)} for ds in DATASETS for model in ["SRAF-ID-current", "SRAF-ID-softmax-fusion"]]
    (out / "run_plan.json").write_text(json.dumps({"stage": STAGE, "seed": SEED, "jobs": jobs, "max_workers": args.max_workers}, indent=2), encoding="utf-8")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(run_job, job) for job in jobs]
        for fut in as_completed(futures):
            result = fut.result()
            rows.append(result["manifest"])
            print(f"completed {result['manifest']['dataset']} {result['manifest']['model']}", flush=True)
    write_csv(out / "run_manifest.csv", rows)
    aggregate(out)
    print("TERMINAL SUMMARY", flush=True)
    print(f"completed jobs: {len(rows)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()

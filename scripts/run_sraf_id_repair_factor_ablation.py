"""Strict temporal/spatial/MLP/profile factor ablation for SRAF-ID repair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sraf_id_repair_v3_light_diagnostic import (  # noqa: E402
    DATASETS,
    FAULTS,
    SEEDS,
    STAGE,
    aggregate,
    build_tod_profile,
    make_current_sraf_id,
    run_job as _unused_run_job,
    train_v3,
    write_csv,
    predict_sraf,
    get_faulted,
)
from scripts.run_metr_la_strong_clean_backbone_integration import resolve_device  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload, safe_metrics  # noqa: E402
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import build_official_stid, predict_v2, train_v2  # noqa: E402
from scripts.run_metr_la_sraf_stid_same_backbone_gain import model_param_count  # noqa: E402
from src.models.strong_backbones_v3 import SRAFOfficialStyleSTIDWrapperFactorAblation  # noqa: E402


FACTOR_STAGE = "SRAF_ID_REPAIR_FACTOR_ABLATION_DIAGNOSTIC_GATE"


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"name": "C0_current_sraf_id_budget", "family": "current", "description": "Current SRAF-ID same-budget reference."},
        {"name": "A1_temporal_only_bidir", "family": "factor", "temporal_mode": "bidir", "spatial_mode": "adjacency", "fusion_mode": "alpha", "use_profile": False, "topk": 5, "fixed_profile_weight": 0.0, "description": "Only temporal repair changes to bidirectional; spatial/fusion/profile held conservative."},
        {"name": "A2_spatial_only_observed_topk", "family": "factor", "temporal_mode": "basic", "spatial_mode": "topk", "fusion_mode": "alpha", "use_profile": False, "topk": 5, "fixed_profile_weight": 0.0, "description": "Only spatial repair changes to observed-aware top-k; temporal/fusion/profile held conservative."},
        {"name": "A3_mlp_only_softmax_no_profile", "family": "factor", "temporal_mode": "basic", "spatial_mode": "adjacency", "fusion_mode": "softmax", "use_profile": False, "topk": 5, "fixed_profile_weight": 0.0, "description": "Only fusion changes from sigmoid alpha to 2-way softmax MLP; no profile."},
        {"name": "A4_profile_only_fixed10", "family": "factor", "temporal_mode": "basic", "spatial_mode": "adjacency", "fusion_mode": "alpha", "use_profile": True, "topk": 5, "fixed_profile_weight": 0.10, "description": "Only adds train-only TOD profile with fixed 10% conservative weight; no softmax profile selection."},
        {"name": "A5_full_bidir_topk_profile_softmax", "family": "factor", "temporal_mode": "bidir", "spatial_mode": "topk", "fusion_mode": "softmax", "use_profile": True, "topk": 5, "fixed_profile_weight": 0.0, "description": "Full combination: bidirectional temporal + observed top-k spatial + profile + softmax MLP."},
    ]


def build_factor_model(payload: dict[str, Any], cfg: dict[str, Any]) -> nn.Module:
    sensors = payload["train_x"].shape[2]
    input_length = payload["train_x"].shape[1]
    horizon = payload["train_y"].shape[1]
    return SRAFOfficialStyleSTIDWrapperFactorAblation(
        sensors=sensors,
        backbone=build_official_stid(sensors, input_length, horizon),
        tod_profile=build_tod_profile(payload["train_x"]),
        temporal_mode=cfg.get("temporal_mode", "basic"),
        spatial_mode=cfg.get("spatial_mode", "adjacency"),
        fusion_mode=cfg.get("fusion_mode", "alpha"),
        use_profile=bool(cfg.get("use_profile", False)),
        topk=int(cfg.get("topk", 5)),
        fusion_hidden_dim=int(cfg.get("fusion_hidden", 16)),
        fixed_profile_weight=float(cfg.get("fixed_profile_weight", 0.10)),
        observed_input_blend=float(cfg.get("observed_blend", 0.5)),
    )


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    from time import perf_counter
    import numpy as np

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
    (run_dir / "config_snapshot.json").write_text(json.dumps({"stage": FACTOR_STAGE, "dataset": dataset, "seed": seed, "candidate": cfg, "args": job["args"]}, indent=2), encoding="utf-8")
    st = perf_counter()
    if cfg["family"] == "current":
        model = make_current_sraf_id(payload)
        meta, curves = train_v2(model, cfg["name"], payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj_t)
        predict_kind = "v2"
    else:
        model = build_factor_model(payload, cfg)
        meta, curves = train_v3(model, cfg["name"], payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj_t, lambda_delta=0.0)
        predict_kind = "factor"
    write_csv(run_dir / "training_curves.csv", curves)
    metric_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for idx, fault in enumerate(FAULTS):
        x_fault, mask, observed = get_faulted(payload["test_x"], fault, seed + idx)
        if predict_kind == "v2":
            pred, latency = predict_v2(model, x_fault, observed, args.batch_size, device, adj_t)
            comps = None
        else:
            pred, latency, comps = predict_sraf(model, x_fault, observed, args.batch_size, device, adj_t, return_components=True)
        met = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        metric_rows.append(
            {
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
        )
        if comps is not None and "weights" in comps:
            weights = comps["weights"]
            row = {
                "dataset": dataset,
                "seed": seed,
                "candidate": cfg["name"],
                "fault": fault,
                "repair_displacement_mean": float(np.mean(comps.get("repair_disp", np.array([np.nan])))),
                "weight_temporal_mean": float(np.mean(weights[..., 0])),
                "weight_spatial_mean": float(np.mean(weights[..., 1])),
            }
            if weights.shape[-1] > 2:
                row["weight_profile_mean"] = float(np.mean(weights[..., 2]))
            comp_rows.append(row)
    write_csv(run_dir / "metrics.csv", metric_rows)
    if comp_rows:
        write_csv(run_dir / "repair_component_stats.csv", comp_rows)
    manifest = {"stage": FACTOR_STAGE, "dataset": dataset, "seed": seed, "candidate": cfg["name"], "status": "completed", "runtime_sec": perf_counter() - st, "output_path": str(run_dir)}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest}


def write_factor_report(out: Path) -> None:
    cmp = pd.read_csv(out / "comparison_vs_current_sraf_id.csv")
    rank = pd.read_csv(out / "candidate_ranking.csv")
    stat_path = out / "repair_component_statistics.csv"
    stats = pd.read_csv(stat_path) if stat_path.exists() else pd.DataFrame()
    neg = cmp[(cmp.candidate != "C0_current_sraf_id_budget") & (cmp.gain_vs_current_pct < 0)]
    neg.to_csv(out / "negative_cases_vs_current_sraf_id.csv", index=False)
    lines = [
        "# SRAF_ID_REPAIR_FACTOR_ABLATION_DIAGNOSTIC_REPORT",
        "",
        "## 1. Stage Metadata",
        f"- stage: `{FACTOR_STAGE}`",
        f"- timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        "- diagnostic only: `YES`",
        "- formal/manuscript results modified: `NO`",
        "- purpose: isolate temporal, spatial, MLP fusion, and profile contributions.",
        "",
        "## 2. Candidate Definitions",
    ]
    for c in candidate_grid():
        lines.append(f"- `{c['name']}`: {c['description']}")
    lines.extend(["", "## 3. Ranking"])
    for r in rank.itertuples():
        lines.append(f"- `{r.candidate}`: avg_gain_vs_current=`{r.avg_gain_vs_current_pct:.3f}%`, avg_mae=`{r.avg_mae:.6f}`, latency=`{r.avg_latency_sec:.4f}s`, params=`{int(r.params)}`.")
    lines.extend(["", "## 4. Factor Attribution"])
    for cand in [c["name"] for c in candidate_grid() if c["name"] != "C0_current_sraf_id_budget"]:
        sub = cmp[cmp.candidate == cand]
        lines.append(f"- `{cand}`: mean gain=`{sub.gain_vs_current_pct.mean():.3f}%`, negative pairs=`{int((sub.gain_vs_current_pct < 0).sum())}/10`.")
    if not stats.empty:
        lines.extend(["", "## 5. Fusion Weight Diagnostics"])
        for cand, sub in stats.groupby("candidate"):
            vals = sub.mean(numeric_only=True)
            profile = vals.get("weight_profile_mean", float("nan"))
            lines.append(f"- `{cand}`: temporal=`{vals.get('weight_temporal_mean', float('nan')):.3f}`, spatial=`{vals.get('weight_spatial_mean', float('nan')):.3f}`, profile=`{profile:.3f}`.")
    lines.extend(
        [
            "",
            "## 6. Decision",
            "- SHOULD_REPLACE_CURRENT_SRAF_ID: `NO`",
            "- SHOULD_UPDATE_MANUSCRIPT_NOW: `NO`",
            "- interpretation: use this diagnostic only for repair design. Formal results remain unchanged.",
            "- outputs: `diagnostic_per_seed_metrics.csv`, `diagnostic_aggregate_metrics.csv`, `comparison_vs_current_sraf_id.csv`, `candidate_ranking.csv`, `repair_component_statistics.csv`.",
        ]
    )
    (out / "SRAF_ID_REPAIR_FACTOR_ABLATION_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/sraf_id_repair_factor_ablation_diagnostic")
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
    plan = {"stage": FACTOR_STAGE, "created_at": datetime.now().isoformat(timespec="seconds"), "datasets": DATASETS, "faults": FAULTS, "seeds": SEEDS, "candidates": [c["name"] for c in candidates], "expected_training_jobs": len(jobs), "max_workers": args.max_workers, "diagnostic_only_not_formal": True}
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(run_job, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                rows.append(result["manifest"])
                print(f"completed {result['manifest']['dataset']} seed={result['manifest']['seed']} {result['manifest']['candidate']}", flush=True)
            except Exception as exc:
                rows.append({"status": "failed", "error": repr(exc)})
                print(f"failed {repr(exc)}", flush=True)
    write_csv(out / "run_manifest.csv", rows)
    summary = aggregate(out, len(jobs))
    write_factor_report(out)
    print("TERMINAL SUMMARY", flush=True)
    print(f"expected training jobs: {len(jobs)}", flush=True)
    print(f"completed training jobs: {summary['completed_jobs']}", flush=True)
    print(f"status: {summary['status']}", flush=True)


if __name__ == "__main__":
    main()

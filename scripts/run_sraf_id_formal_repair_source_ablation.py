"""Formal 10-seed repair-source ablations for final SRAF-ID.

Runs temporal-only, spatial-only, and fixed-fusion variants under the same
formal protocol as the final SRAF-ID softmax-fusion mainline. This script does
not rerun baselines or alter the final SRAF-ID implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from scripts.run_sraf_id_final_figure_table_package import (  # noqa: E402
    DATASETS,
    FAULTS,
    LOCAL_FAULT_SPECS,
    SEEDS,
    combined_frames,
    load_new_metrics,
)
from scripts.run_sraf_id_repair_factor_ablation import build_factor_model  # noqa: E402
from scripts.run_sraf_id_repair_v3_light_diagnostic import predict_sraf, train_v3, write_csv  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload, safe_metrics  # noqa: E402


STAGE = "SRAF_ID_FORMAL_REPAIR_SOURCE_ABLATION_GATE"
FINAL_PACKAGE = ROOT / "experiments" / "sraf_id_final_figure_table_package"
OUT_DEFAULT = "experiments/sraf_id_formal_repair_source_ablation"
VARIANTS = [
    {
        "model": "SRAF-ID-temporal-only",
        "fusion_mode": "temporal_only",
        "description": "Use only the temporal repair candidate; current spatial candidate is computed but not used.",
    },
    {
        "model": "SRAF-ID-spatial-only",
        "fusion_mode": "spatial_only",
        "description": "Use only the adjacency spatial repair candidate; current temporal candidate is computed but not used.",
    },
    {
        "model": "SRAF-ID-fixed-fusion",
        "fusion_mode": "fixed",
        "description": "Use fixed 0.5 temporal + 0.5 spatial repair fusion; no learned fusion weights.",
    },
]


def get_faulted(x: np.ndarray, fault: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fault == "clean":
        mask = np.zeros_like(x[..., :1], dtype=np.float32)
        observed = np.ones_like(mask, dtype=np.float32)
        return x.astype(np.float32), mask, observed
    spec = LOCAL_FAULT_SPECS[fault]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    x_fault = x.copy()
    x_fault[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return x_fault.astype(np.float32), mask.astype(np.float32), observed


def variant_cfg(model_name: str) -> dict[str, Any]:
    row = next(v for v in VARIANTS if v["model"] == model_name)
    return {
        "name": model_name,
        "family": "factor",
        "temporal_mode": "basic",
        "spatial_mode": "adjacency",
        "fusion_mode": row["fusion_mode"],
        "use_profile": False,
        "topk": 5,
        "fixed_profile_weight": 0.0,
        "description": row["description"],
    }


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**job["args"])
    dataset = job["dataset"]
    seed = int(job["seed"])
    model_name = job["model"]
    args.seed = seed
    out = Path(args.output_dir)
    run_dir = out / "per_run" / dataset.lower().replace("-", "_") / f"seed_{seed}" / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    manifest_path = run_dir / "run_manifest.json"
    if args.skip_existing and metrics_path.exists() and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("status") == "completed":
                return {"dataset": dataset, "seed": seed, "model": model_name, "status": "skipped_existing", "output_path": str(run_dir)}
        except Exception:
            pass
    start = perf_counter()
    device = resolve_device(args.device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    payload = load_payload(dataset, 10**12 if args.train_limit == 0 else args.train_limit, 10**12 if args.val_limit == 0 else args.val_limit, None if args.test_limit == 0 else args.test_limit)
    adj_t = torch.from_numpy(payload["adj"]).to(device)
    cfg = variant_cfg(model_name)
    model = build_factor_model(payload, cfg)
    model.to(device)
    (run_dir / "config_snapshot.json").write_text(json.dumps({"stage": STAGE, "dataset": dataset, "seed": seed, "model": model_name, "variant": cfg, "args": vars(args)}, indent=2), encoding="utf-8")
    checkpoint = run_dir / "best_checkpoint.pt"
    curves_path = run_dir / "training_curves.csv"
    if args.skip_existing and checkpoint.exists() and curves_path.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        curves_df = pd.read_csv(curves_path)
        meta = {
            "training_time_sec": float("nan"),
            "best_epoch": int(curves_df.loc[curves_df["best_selection_val_loss"].idxmin(), "epoch"]) if not curves_df.empty else -1,
            "best_val_loss": float(curves_df["best_selection_val_loss"].min()) if not curves_df.empty else float("nan"),
            "reused_checkpoint": True,
        }
    else:
        meta, curves = train_v3(model, model_name, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj_t, lambda_delta=0.0)
        meta["reused_checkpoint"] = False
        write_csv(curves_path, curves)
    rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for idx, fault in enumerate(FAULTS):
        x_fault, _mask, observed = get_faulted(payload["test_x"], fault, seed + idx)
        pred, latency, comps = predict_sraf(model, x_fault, observed, args.batch_size, device, adj_t, return_components=True)
        met = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                "fault": fault,
                "mae": float(met["mae"]),
                "rmse": float(met["rmse"]),
                "h3_mae": float(met["mae_h3"]),
                "h6_mae": float(met["mae_h6"]),
                "h12_mae": float(met["mae_h12"]),
                "latency_sec": float(latency),
                "parameter_count": int(model_param_count(model)),
                "training_time_sec": float(meta["training_time_sec"]),
                "best_epoch": int(meta["best_epoch"]),
                "reused_checkpoint": bool(meta.get("reused_checkpoint", False)),
                "status": "completed",
            }
        )
        if comps is not None and "weights" in comps:
            w = comps["weights"]
            comp_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": model_name,
                    "fault": fault,
                    "weight_temporal_mean": float(np.mean(w[..., 0])),
                    "weight_spatial_mean": float(np.mean(w[..., 1])),
                    "repair_displacement_mean": float(np.mean(comps.get("repair_disp", np.array([np.nan])))),
                }
            )
    write_csv(metrics_path, rows)
    if comp_rows:
        write_csv(run_dir / "repair_component_stats.csv", comp_rows)
    manifest = {
        "stage": STAGE,
        "dataset": dataset,
        "seed": seed,
        "model": model_name,
        "status": "completed",
        "runtime_sec": perf_counter() - start,
        "output_path": str(run_dir),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def aggregate(out: Path, run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_files = sorted((out / "per_run").glob("**/metrics.csv"))
    rows = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    rows.to_csv(out / "ablation_per_seed_metrics.csv", index=False)
    if rows.empty:
        return {"status": "FAIL", "completed_jobs": 0}
    agg = (
        rows.groupby(["dataset", "model", "fault"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
            h3_mae_mean=("h3_mae", "mean"),
            h3_mae_std=("h3_mae", lambda x: float(np.std(x, ddof=0))),
            h6_mae_mean=("h6_mae", "mean"),
            h6_mae_std=("h6_mae", lambda x: float(np.std(x, ddof=0))),
            h12_mae_mean=("h12_mae", "mean"),
            h12_mae_std=("h12_mae", lambda x: float(np.std(x, ddof=0))),
            seeds=("seed", "nunique"),
            latency_mean_sec=("latency_sec", "mean"),
            parameter_count_mean=("parameter_count", "mean"),
            training_time_mean_sec=("training_time_sec", "mean"),
        )
    )
    agg.to_csv(out / "ablation_aggregate_by_fault.csv", index=False)
    faulty = rows[rows["fault"] != "clean"]
    avg = faulty.groupby(["dataset", "model"], as_index=False).agg(avg_faulty_mae_mean=("mae", "mean"), avg_faulty_mae_std=("mae", lambda x: float(np.std(x, ddof=0))), seeds=("seed", "nunique"))
    clean = rows[rows["fault"] == "clean"].groupby(["dataset", "model"], as_index=False).agg(clean_mae_mean=("mae", "mean"), clean_mae_std=("mae", lambda x: float(np.std(x, ddof=0))))
    avg = avg.merge(clean, on=["dataset", "model"], how="left")
    final_avg = pd.read_csv(FINAL_PACKAGE / "sraf_id_softmax_formal_avg_faulty_summary.csv")
    ref = final_avg[["dataset", "model", "avg_faulty_mae_mean", "avg_faulty_mae_std", "clean_mae_mean"]].copy()
    ref = ref[ref["model"] == "SRAF-ID"]
    avg_with_ref = avg.merge(ref[["dataset", "avg_faulty_mae_mean"]].rename(columns={"avg_faulty_mae_mean": "sraf_id_avg_faulty_mae"}), on="dataset", how="left")
    avg_with_ref["difference_vs_sraf_id_pct"] = (avg_with_ref["avg_faulty_mae_mean"] - avg_with_ref["sraf_id_avg_faulty_mae"]) / avg_with_ref["sraf_id_avg_faulty_mae"] * 100.0
    avg_with_ref.to_csv(out / "ablation_avg_faulty_summary.csv", index=False)
    table_rows = []
    gated = pd.read_csv(FINAL_PACKAGE / "combined_avg_faulty_summary.csv")
    gated = gated[gated["model"].isin(["SRAF-ID", "SRAF-ID-gated"])][["dataset", "model", "avg_faulty_mae_mean", "avg_faulty_mae_std", "clean_mae_mean"]]
    combo = pd.concat([gated, avg], ignore_index=True, sort=False)
    s_map = combo[combo.model == "SRAF-ID"].set_index("dataset")["avg_faulty_mae_mean"].to_dict()
    for r in combo.itertuples():
        diff = 0.0 if r.model == "SRAF-ID" else (r.avg_faulty_mae_mean - s_map[r.dataset]) / s_map[r.dataset] * 100.0
        table_rows.append(
            {
                "Dataset": r.dataset,
                "Variant": r.model,
                "Average faulty MAE mean ± std": f"{r.avg_faulty_mae_mean:.4f} ± {r.avg_faulty_mae_std:.4f}",
                "Difference vs SRAF-ID": f"{diff:.3f}%",
                "Interpretation": "final method" if r.model == "SRAF-ID" else "repair-source ablation" if r.model.startswith("SRAF-ID-") and r.model != "SRAF-ID-gated" else "gated ablation/supplementary variant",
            }
        )
    pd.DataFrame(table_rows).to_csv(out / "table8_ablation_study_revised.csv", index=False)
    status = "PASS" if len([r for r in run_rows if r.get("status") in {"completed", "skipped_existing"}]) == len(DATASETS) * len(SEEDS) * len(VARIANTS) else "PARTIAL"
    report = [
        "# SRAF_ID_FORMAL_REPAIR_SOURCE_ABLATION_REPORT",
        "",
        f"- stage: `{STAGE}`",
        f"- status: `{status}`",
        f"- timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- expected training jobs: `{len(DATASETS) * len(SEEDS) * len(VARIANTS)}`",
        f"- completed/skipped jobs: `{len([r for r in run_rows if r.get('status') in {'completed', 'skipped_existing'}])}`",
        "- baselines rerun: `NO`",
        "- final SRAF-ID modified: `NO`",
        "",
        "## Average Faulty Ablation Summary",
    ]
    for r in avg_with_ref.itertuples():
        report.append(f"- `{r.dataset}` / `{r.model}`: MAE=`{r.avg_faulty_mae_mean:.6f} ± {r.avg_faulty_mae_std:.6f}`, diff_vs_SRAF-ID=`{r.difference_vs_sraf_id_pct:.3f}%`.")
    report.extend(["", "## Outputs", "- `ablation_per_seed_metrics.csv`", "- `ablation_aggregate_by_fault.csv`", "- `ablation_avg_faulty_summary.csv`", "- `table8_ablation_study_revised.csv`"])
    (out / "SRAF_ID_FORMAL_REPAIR_SOURCE_ABLATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return {"status": status, "completed_jobs": len([r for r in run_rows if r.get("status") in {"completed", "skipped_existing"}])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=OUT_DEFAULT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--test-limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"dataset": ds, "seed": seed, "model": v["model"], "args": vars(args)} for ds in DATASETS for seed in SEEDS for v in VARIANTS]
    plan = {
        "stage": STAGE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variants": [v["model"] for v in VARIANTS],
        "datasets": DATASETS,
        "faults": FAULTS,
        "seeds": SEEDS,
        "expected_training_jobs": len(jobs),
        "expected_metric_rows": len(jobs) * len(FAULTS),
        "max_workers": args.max_workers,
        "dry_run": args.dry_run,
    }
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(run_job, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                rows.append(result)
                print(f"{result['status']} {result['dataset']} seed={result['seed']} {result['model']}", flush=True)
            except Exception as exc:
                fail = {"status": "failed", "error_message": repr(exc)}
                rows.append(fail)
                print(f"failed {exc!r}", flush=True)
    write_csv(out / "run_manifest.csv", rows)
    summary = aggregate(out, rows)
    print("TERMINAL SUMMARY", flush=True)
    print(f"expected training jobs: {len(jobs)}", flush=True)
    print(f"completed/skipped jobs: {summary['completed_jobs']}", flush=True)
    print(f"status: {summary['status']}", flush=True)


if __name__ == "__main__":
    main()

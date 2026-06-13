"""Run a fault-distribution-matched ID-MLP-CA formal 10-seed baseline.

The final SRAF-ID mainline is trained with five rotating fault protocols. This
script trains the same-backbone ID-MLP-CA with exactly those five protocols and
evaluates each checkpoint on clean input plus the six manuscript test faults.
"""

from __future__ import annotations

import argparse
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

import scripts.run_metr_la_sraf_stid_same_backbone_gain as ca_module  # noqa: E402
from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    model_param_count,
    predict_model,
    train_official_stid_ca,
)
from scripts.run_metr_la_strong_clean_backbone_integration import resolve_device  # noqa: E402
from scripts.run_sraf_id_final_figure_table_package import get_faulted  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import (  # noqa: E402
    FAULT_SPECS,
    load_payload,
    safe_metrics,
)
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import build_official_stid  # noqa: E402
from src.protocols.matched_protocol import (  # noqa: E402
    DATASETS,
    DEFAULT_CONFIG_PATH,
    MATCHED_TRAIN_FAULTS,
    SEEDS,
    TEST_FAULTS,
)


STAGE = "MATCHED_BASELINE_FORMAL_10SEED_GATE"
TRAIN_FAULT_LABELS = list(MATCHED_TRAIN_FAULTS)
FAULTS = list(TEST_FAULTS)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**job["args"])
    dataset = str(job["dataset"])
    seed = int(job["seed"])
    args.seed = seed
    out = Path(args.output_dir)
    run_dir = out / "per_run" / f"{dataset.lower().replace('-', '_')}__id_mlp_ca_matched__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.csv"
    if args.skip_existing and manifest_path.exists() and metrics_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            return {"dataset": dataset, "seed": seed, "status": "skipped_existing", "output_path": str(run_dir)}

    started = perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    payload = load_payload(
        dataset,
        10**12 if args.train_limit == 0 else args.train_limit,
        10**12 if args.val_limit == 0 else args.val_limit,
        None if args.test_limit == 0 else args.test_limit,
    )
    model = build_official_stid(payload["train_x"].shape[2], payload["train_x"].shape[1], payload["train_y"].shape[1])

    # train_official_stid_ca resolves this module global at runtime. Assigning
    # the final SRAF-ID fault list makes the baseline training distribution
    # exactly matched while leaving the original source and artifacts intact.
    ca_module.TRAIN_FAULTS = [FAULT_SPECS[label] for label in TRAIN_FAULT_LABELS]
    meta, curves = train_official_stid_ca(
        model,
        payload["train_x"],
        payload["train_y"],
        payload["val_x"],
        payload["val_y"],
        args,
        run_dir,
        device,
    )
    write_csv(run_dir / "training_curves.csv", curves)

    rows: list[dict[str, Any]] = []
    for idx, fault in enumerate(FAULTS):
        x_fault, _, _ = get_faulted(payload["test_x"], fault, seed + idx)
        pred, latency, _ = predict_model(model, x_fault, args.batch_size, device, sraf=False)
        metrics = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model": "ID-MLP-CA",
                "fault": fault,
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "h3_mae": float(metrics["mae_h3"]),
                "h6_mae": float(metrics["mae_h6"]),
                "h12_mae": float(metrics["mae_h12"]),
                "latency_sec": float(latency),
                "parameter_count": int(model_param_count(model)),
                "training_time_sec": float(meta["training_time_sec"]),
                "best_epoch": int(meta["best_epoch"]),
                "status": "completed",
                "source_result_file": str(metrics_path),
            }
        )
    write_csv(metrics_path, rows)
    config = {
        "stage": STAGE,
        "dataset": dataset,
        "seed": seed,
        "model": "ID-MLP-CA",
        "train_fault_labels": TRAIN_FAULT_LABELS,
        "evaluation_faults": FAULTS,
        "fault_seed_rule": "seed + evaluation_fault_index",
        "protocol_config": str(DEFAULT_CONFIG_PATH),
        "args": vars(args),
    }
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        **config,
        "status": "completed",
        "runtime_sec": perf_counter() - started,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"dataset": dataset, "seed": seed, "status": "completed", "output_path": str(run_dir)}


def aggregate(out: Path, datasets: list[str], seeds: list[int]) -> None:
    files = sorted((out / "per_run").glob("*/metrics.csv"))
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    expected = len(datasets) * len(seeds) * len(FAULTS)
    if len(rows) != expected or rows["seed"].nunique() != len(seeds):
        raise RuntimeError(f"Incomplete matched baseline: rows={len(rows)}, expected={expected}")
    agg_dir = out / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(agg_dir / "formal_10seed_per_seed_metrics.csv", index=False)
    by_fault = (
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
    by_fault.to_csv(agg_dir / "formal_10seed_metrics_by_model_fault.csv", index=False)
    faulty = rows[rows["fault"] != "clean"]
    avg = faulty.groupby(["dataset", "model"], as_index=False).agg(
        avg_faulty_mae_mean=("mae", "mean"),
        avg_faulty_mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
        avg_faulty_rmse_mean=("rmse", "mean"),
        avg_faulty_rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
        seeds=("seed", "nunique"),
    )
    clean = rows[rows["fault"] == "clean"].groupby(["dataset", "model"], as_index=False).agg(
        clean_mae_mean=("mae", "mean"),
        clean_mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
    )
    severe = rows[rows["fault"].isin(["random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"])].groupby(
        ["dataset", "model"], as_index=False
    ).agg(severe_fault_mae_mean=("mae", "mean"))
    avg = avg.merge(severe, on=["dataset", "model"], how="left").merge(clean, on=["dataset", "model"], how="left")
    avg.to_csv(agg_dir / "formal_10seed_avg_faulty_summary.csv", index=False)
    comp = rows.groupby(["dataset", "model"], as_index=False).agg(
        parameter_count=("parameter_count", "mean"),
        latency_mean_sec=("latency_sec", "mean"),
        training_time_sec=("training_time_sec", "mean"),
    )
    comp.to_csv(agg_dir / "formal_10seed_complexity_latency.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/id_mlp_ca_matched_fault_distribution_10seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--loss", choices=["mae", "mse"], default="mae")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    jobs = [{"dataset": dataset, "seed": seed, "args": vars(args)} for dataset in datasets for seed in seeds]
    plan = {
        "stage": STAGE,
        "protocol_config": str(DEFAULT_CONFIG_PATH),
        "datasets": datasets,
        "seeds": seeds,
        "train_fault_labels": TRAIN_FAULT_LABELS,
        "evaluation_faults": FAULTS,
        "expected_training_jobs": len(jobs),
        "dry_run": bool(args.dry_run),
    }
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
        return
    results: list[dict[str, Any]] = []
    if args.max_workers == 1:
        for job in jobs:
            results.append(run_one(job))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
    write_csv(out / "run_status.csv", results)
    aggregate(out, datasets, seeds)
    manifest = {
        "stage": STAGE,
        "status": "PASS",
        "datasets": datasets,
        "seeds": seeds,
        "train_fault_labels": TRAIN_FAULT_LABELS,
        "evaluation_faults": FAULTS,
        "expected_training_jobs": len(jobs),
        "completed_or_skipped_jobs": len(results),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

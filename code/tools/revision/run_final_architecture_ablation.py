"""Run only the forecast-only architecture ablations required by the final audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_strong_clean_backbone_integration import resolve_device
from scripts.run_sraf_id_repair_factor_ablation import build_factor_model
from scripts.run_sraf_id_repair_v3_light_diagnostic import predict_sraf, train_v3
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload, safe_metrics
from tools.revision.run_reviewer_experiments import FAULTS, FAULT_LABELS, get_faulted, set_seeds, write_csv, write_json

STAGE = "SRAF_ID_FINAL_REVISION_EVIDENCE_FREEZE_20260730"
OUT = ROOT / "artifacts" / "revision_final_20260730"
DATASETS = ["METR-LA", "PEMS-BAY"]
SEEDS = list(range(42, 52))
VARIANTS = {
    "temporal_only_forecast_only": "temporal_only",
    "spatial_only_forecast_only": "spatial_only",
    "fixed_fusion_forecast_only": "fixed",
    "gated_fusion_forecast_only": "alpha",
}


def config_for(dataset: str, variant: str, seed: int) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "run_id": f"final_ablation__{dataset.lower().replace('-', '_')}__{variant}__seed{seed}",
        "dataset": dataset.lower().replace("-", "_"),
        "variant": variant,
        "seed": seed,
        "history_length": 12,
        "prediction_horizon": 12,
        "optimizer": "Adam",
        "maximum_epochs": 60,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.002,
        "weight_decay": 0.0001,
        "gradient_clipping": 5.0,
        "loss": "MAE",
        "repair_loss_weight": 0.0,
        "total_loss": "forecast_loss",
        "training_faults": ["CO24", "RM40", "GN-high", "LD-high", "SV-high"],
        "evaluation_faults": ["clean", "RM20", "RM40", "CO24", "GN-high", "LD-high", "SV-high"],
        "observed_input_blend": 0.5,
        "clean_future_target": True,
        "fault_location_mask_used_in_loss": False,
        "fault_location_mask_used_as_inference_input": False,
        "forecasting_backbone": "ID-MLP",
        "fusion_mode": VARIANTS[variant],
    }


def run_dir(dataset: str, variant: str, seed: int) -> Path:
    return OUT / "architecture_ablation" / dataset.lower().replace("-", "_") / variant / f"seed_{seed}"


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def run_one(dataset: str, variant: str, seed: int, device: torch.device) -> dict[str, Any]:
    target = run_dir(dataset, variant, seed)
    target.mkdir(parents=True, exist_ok=True)
    cfg = config_for(dataset, variant, seed)
    cfg_hash = digest(cfg)
    status_path = target / "run_status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "completed" and old.get("config_hash") == cfg_hash:
            return old

    write_json(target / "config.json", cfg)
    (target / "config_hash.txt").write_text(cfg_hash + "\n", encoding="utf-8")
    set_seeds(seed)
    payload = load_payload(dataset, 10**12, 10**12, None)
    adjacency = torch.from_numpy(payload["adj"]).to(device)
    factor_cfg = {
        "name": variant,
        "family": "factor",
        "temporal_mode": "basic",
        "spatial_mode": "adjacency",
        "fusion_mode": VARIANTS[variant],
        "use_profile": False,
        "topk": 5,
        "fusion_hidden": 16,
        "fixed_profile_weight": 0.0,
        "observed_blend": 0.5,
    }
    model = build_factor_model(payload, factor_cfg).to(device)
    train_args = argparse.Namespace(
        seed=seed,
        epochs=60,
        patience=10,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.0001,
        grad_clip=5.0,
        loss="mae",
        lambda_repair=0.0,
        lambda_rel=0.0,
    )
    started = perf_counter()
    meta, curves = train_v3(
        model, variant, payload["train_x"], payload["train_y"], payload["val_x"],
        payload["val_y"], train_args, target, device, adjacency, lambda_delta=0.0,
    )
    pd.DataFrame(curves).to_csv(target / "epoch_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(target / "validation_metrics.csv", index=False)
    (target / "train.log").write_text(
        "status=completed\n"
        f"training_time_sec={meta['training_time_sec']}\n"
        f"best_epoch={meta['best_epoch']}\n"
        f"best_validation_mae={meta['best_val_loss']}\n"
        "total_loss=forecast_loss\nrepair_loss_weight=0.0\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for index, fault in enumerate(FAULTS):
        x_fault, _fault_mask, observed = get_faulted(payload["test_x"], fault, seed + index)
        pred, _latency, _components = predict_sraf(
            model, x_fault, observed, 64, device, adjacency, return_components=False
        )
        metrics = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        setting, label = FAULT_LABELS[fault]
        rows.append({
            "dataset": dataset.lower().replace("-", "_"),
            "variant": variant,
            "seed": seed,
            "input_setting": setting,
            "fault": label,
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "horizon_3_mae": float(metrics["mae_h3"]),
            "horizon_6_mae": float(metrics["mae_h6"]),
            "horizon_12_mae": float(metrics["mae_h12"]),
            "checkpoint_epoch": int(meta["best_epoch"]),
            "validation_mae": float(meta["best_val_loss"]),
        })
    write_csv(target / "evaluation_by_fault.csv", rows)
    write_json(target / "evaluation_summary.json", {
        "dataset": dataset.lower().replace("-", "_"),
        "variant": variant,
        "seed": seed,
        "clean_mae": rows[0]["mae"],
        "faulty_average_mae": float(np.mean([r["mae"] for r in rows[1:]])),
        "test_results_descriptive_only": True,
    })
    status = {
        "run_id": cfg["run_id"],
        "status": "completed",
        "config_hash": cfg_hash,
        "attempts": 1,
        "runtime_sec": perf_counter() - started,
        "checkpoint_path": str(target / "best_checkpoint.pt"),
        "metrics_path": str(target / "evaluation_by_fault.csv"),
    }
    write_json(status_path, status)
    return status


def jobs() -> list[tuple[str, str, int]]:
    return [(dataset, variant, seed) for dataset in DATASETS for variant in VARIANTS for seed in SEEDS]


def run_worker(index: int, count: int) -> None:
    device = resolve_device("cuda")
    selected = [job for position, job in enumerate(jobs()) if position % count == index]
    for position, (dataset, variant, seed) in enumerate(selected, 1):
        print(f"[worker {index} {position}/{len(selected)}] {dataset} {variant} seed={seed}", flush=True)
        for attempt in (1, 2):
            try:
                result = run_one(dataset, variant, seed, device)
                if attempt != result.get("attempts"):
                    result["attempts"] = attempt
                    write_json(run_dir(dataset, variant, seed) / "run_status.json", result)
                break
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                log = OUT / "logs" / f"final_ablation__{dataset.lower().replace('-', '_')}__{variant}__seed{seed}__attempt{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(message + "\n", encoding="utf-8")
                if attempt == 2:
                    write_json(run_dir(dataset, variant, seed) / "run_status.json", {
                        "run_id": config_for(dataset, variant, seed)["run_id"],
                        "status": "FAILED",
                        "attempts": 2,
                        "error": message,
                    })
                    print(message, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["one", "worker"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "one":
        if args.dataset is None or args.variant is None or args.seed is None:
            parser.error("one requires --dataset, --variant, and --seed")
        print(json.dumps(run_one(args.dataset, args.variant, args.seed, resolve_device("cuda"))), flush=True)
    else:
        run_worker(args.worker_index, args.worker_count)


if __name__ == "__main__":
    main()

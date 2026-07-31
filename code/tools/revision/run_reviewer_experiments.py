"""Audit and run the reviewer-requested SRAF-ID repair-loss experiments.

All new artifacts are written below ``artifacts/revision_20260728``.  The
script deliberately reuses the frozen model/data helpers without changing the
paper's model implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import model_param_count
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault, resolve_device
from scripts.run_sraf_id_repair_factor_ablation import build_factor_model
from scripts.run_sraf_id_repair_v3_light_diagnostic import predict_sraf, train_v3
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload, safe_metrics

STAGE = "SRAF_ID_REVIEWER_REVISION_EXPERIMENTS_20260728"
OUT = ROOT / "artifacts" / "revision_20260728"
SEEDS = list(range(42, 52))
DATASETS = ["METR-LA", "PEMS-BAY"]
FAULTS = [
    "clean", "random_missing_20", "random_missing_40", "continuous_outage_24",
    "gaussian_noise_high", "linear_drift_high", "stuck_at_last_value_high",
]
FAULT_SPECS = {
    "random_missing_20": {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20"},
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "continuous_outage_24": {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
}
FAULT_LABELS = {
    "clean": ("clean", "clean"),
    "random_missing_20": ("faulty", "RM20"),
    "random_missing_40": ("faulty", "RM40"),
    "continuous_outage_24": ("faulty", "CO24"),
    "gaussian_noise_high": ("faulty", "GN-high"),
    "linear_drift_high": ("faulty", "LD-high"),
    "stuck_at_last_value_high": ("faulty", "SV-high"),
}
VARIANTS = {
    "sraf_id_lambda000": 0.00,
    "sraf_id_lambda001": 0.01,
    "sraf_id_lambda005": 0.05,
    "sraf_id_lambda010": 0.10,
}


def get_faulted(x: np.ndarray, fault: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fault == "clean":
        mask = np.zeros_like(x[..., :1], dtype=np.float32)
        return x.astype(np.float32), mask, np.ones_like(mask, dtype=np.float32)
    speed, mask, _ = apply_fault(x[..., :1], FAULT_SPECS[fault], seed=seed, train_std=1.0)
    x_fault = x.copy()
    x_fault[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return x_fault.astype(np.float32), mask.astype(np.float32), observed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_layout() -> None:
    for name in [
        "audit", "configs", "loss_ablation", "lambda_sensitivity",
        "candidate_diagnostics", "temporal_only_audit", "latency_benchmark",
        "statistics", "figures", "tables", "logs", "summary",
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "revision").mkdir(parents=True, exist_ok=True)


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"NOT_AVAILABLE: {type(exc).__name__}"


def environment_audit() -> None:
    ensure_layout()
    git_status = git_value("status", "--short")
    (OUT / "audit" / "git_status.txt").write_text(git_status + "\n", encoding="utf-8")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    env = {
        "stage": STAGE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repository_root": str(ROOT),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_status": git_status,
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "cpu": platform.processor(),
        "os": platform.platform(),
        "formal_config_paths": [str(ROOT / "configs" / "default.yaml"), str(ROOT / "configs" / "final_submission_fault_protocol.yaml")],
        "dataset_paths": [str(ROOT / "data" / "processed" / "metr-la"), str(ROOT / "data" / "processed" / "pems-bay")],
        "seed_controls": ["python_random", "numpy", "torch_cpu", "torch_cuda", "deterministic_epoch_sampler", "fault_seed"],
        "dataloader_generator": "not applicable: runner uses a deterministic NumPy epoch permutation rather than torch DataLoader",
    }
    write_json(OUT / "audit" / "environment.json", env)


def normalized_faults(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "fault" not in df:
        return set()
    mapping = {
        "clean": "clean", "random_missing_20": "RM20", "random_missing_40": "RM40",
        "continuous_outage_24": "CO24", "gaussian_noise_high": "GN-high",
        "linear_drift_high": "LD-high", "stuck_at_last_value_high": "SV-high",
    }
    return {mapping.get(str(x), str(x)) for x in df["fault"]}


def audit_candidate(dataset: str, variant: str, seed: int, base: Path, config_name: str = "config_snapshot.json") -> dict[str, Any]:
    config = base / config_name
    checkpoint = base / "best_checkpoint.pt"
    metrics = base / "metrics.csv"
    curves = base / "training_curves.csv"
    required = {"clean", "RM20", "RM40", "CO24", "GN-high", "LD-high", "SV-high"}
    faults = normalized_faults(metrics)
    complete = faults == required
    config_obj = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
    args = config_obj.get("args", {})
    protocol = (
        args.get("epochs") == 60 and args.get("patience") == 10 and
        args.get("batch_size") == 64 and args.get("learning_rate") == 0.002 and
        args.get("weight_decay") == 0.0001 and args.get("grad_clip") == 5.0
    )
    allowed = all([config.exists(), checkpoint.exists(), metrics.exists(), curves.exists(), complete, protocol])
    reason = "complete formal per-seed/per-fault artifacts" if allowed else "; ".join([
        *( ["missing config"] if not config.exists() else []),
        *( ["missing checkpoint"] if not checkpoint.exists() else []),
        *( ["missing metrics"] if not metrics.exists() else []),
        *( ["missing validation history"] if not curves.exists() else []),
        *( [f"fault coverage={sorted(faults)}"] if not complete else []),
        *( ["formal protocol fields do not match or are absent"] if not protocol else []),
    ])
    return {
        "dataset": dataset.lower().replace("-", "_"), "variant": variant, "seed": seed,
        "git_commit": "NOT_AVAILABLE", "config_hash": sha256_json(config_obj) if config.exists() else "",
        "config_path": str(config), "checkpoint_path": str(checkpoint), "metrics_path": str(metrics),
        "protocol_match": protocol, "raw_results_complete": complete and curves.exists(),
        "reuse_allowed": allowed, "reason": reason,
    }


def artifact_audit() -> None:
    environment_audit()
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        key = dataset.lower().replace("-", "_")
        for seed in SEEDS:
            rows.append(audit_candidate(dataset, "sraf_id_lambda005", seed, ROOT / "experiments" / "sraf_id_final_figure_table_package" / "per_run" / f"{key}__sraf_id_softmax_fusion__seed{seed}"))
            rows.append(audit_candidate(dataset, "id_mlp_ca", seed, ROOT / "experiments" / "id_mlp_ca_matched_fault_distribution_10seed" / "per_run" / f"{key}__id_mlp_ca_matched__seed{seed}"))
            rows.append(audit_candidate(dataset, "sraf_id_temporal_only", seed, ROOT / "experiments" / "sraf_id_formal_repair_source_ablation" / "per_run" / key / f"seed_{seed}" / "SRAF-ID-temporal-only"))
    write_csv(OUT / "audit" / "reuse_manifest.csv", rows)
    inventory: list[dict[str, Any]] = []
    for ckpt in (ROOT / "experiments").rglob("best_checkpoint.pt"):
        parent = ckpt.parent
        inventory.append({
            "checkpoint_path": str(ckpt), "bytes": ckpt.stat().st_size,
            "config_path": str(parent / "config_snapshot.json") if (parent / "config_snapshot.json").exists() else "",
            "metrics_path": str(parent / "metrics.csv") if (parent / "metrics.csv").exists() else "",
            "validation_path": str(parent / "training_curves.csv") if (parent / "training_curves.csv").exists() else "",
        })
    write_csv(OUT / "audit" / "existing_artifact_inventory.csv", inventory)
    write_json(OUT / "summary" / "AUDIT_SUMMARY.json", {
        "inventory_checkpoints": len(inventory),
        "reuse_rows": len(rows),
        "reuse_allowed": sum(bool(r["reuse_allowed"]) for r in rows),
    })


def model_config(variant: str, dataset: str, seed: int) -> dict[str, Any]:
    return {
        "stage": STAGE, "dataset": dataset, "variant": variant, "seed": seed,
        "history_length": 12, "prediction_horizon": 12,
        "training_faults": ["CO24", "RM40", "GN-high", "LD-high", "SV-high"],
        "evaluation_faults": ["clean", "RM20", "RM40", "CO24", "GN-high", "LD-high", "SV-high"],
        "backbone": "ID-MLP", "optimizer": "Adam", "epochs": 60, "patience": 10,
        "batch_size": 64, "learning_rate": 0.002, "weight_decay": 0.0001,
        "gradient_clipping": 5.0, "repair_loss_weight": VARIANTS[variant],
        "observed_input_blend": 0.5, "target_corrupted": False,
        "finite_fault_mask_used_as_model_input": False,
        "seed_policy": {"python": seed, "numpy": seed, "torch_cpu": seed, "torch_cuda": seed, "sampler": seed, "fault": "seed+fault_index / seed+training_step"},
    }


def run_dir_for(stage_name: str, dataset: str, variant: str, seed: int) -> Path:
    return OUT / stage_name / dataset.lower().replace("-", "_") / variant / f"seed_{seed}"


def train_one(stage_name: str, dataset: str, variant: str, seed: int, device: torch.device) -> dict[str, Any]:
    run_dir = run_dir_for(stage_name, dataset, variant, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_config(variant, dataset, seed)
    cfg_hash = sha256_json(cfg)
    status_path = run_dir / "run_status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "completed" and old.get("config_hash") == cfg_hash:
            return old
    write_json(run_dir / "config.json", cfg)
    (run_dir / "config_hash.txt").write_text(cfg_hash + "\n", encoding="utf-8")
    set_seeds(seed)
    payload = load_payload(dataset, 10**12, 10**12, None)
    adj = torch.from_numpy(payload["adj"]).to(device)
    factor_cfg = {
        "name": variant, "family": "factor", "temporal_mode": "basic",
        "spatial_mode": "adjacency", "fusion_mode": "softmax", "use_profile": False,
        "topk": 5, "fixed_profile_weight": 0.0, "description": "reviewer repair-loss experiment",
    }
    model = build_factor_model(payload, factor_cfg).to(device)
    args = argparse.Namespace(
        seed=seed, epochs=60, patience=10, batch_size=64, learning_rate=0.002,
        weight_decay=0.0001, grad_clip=5.0, loss="mae",
        lambda_repair=VARIANTS[variant], lambda_rel=0.01,
    )
    started = perf_counter()
    meta, curves = train_v3(model, variant, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, run_dir, device, adj, lambda_delta=0.0)
    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(run_dir / "epoch_metrics.csv", index=False)
    curve_df.to_csv(run_dir / "validation_metrics.csv", index=False)
    (run_dir / "train.log").write_text(
        f"status=completed\ntraining_time_sec={meta['training_time_sec']}\nbest_epoch={meta['best_epoch']}\nbest_validation_mae={meta['best_val_loss']}\n",
        encoding="utf-8",
    )
    eval_rows: list[dict[str, Any]] = []
    for idx, fault in enumerate(FAULTS):
        x_fault, _, observed = get_faulted(payload["test_x"], fault, seed + idx)
        pred, _, _ = predict_sraf(model, x_fault, observed, 64, device, adj, return_components=False)
        met = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        setting, label = FAULT_LABELS[fault]
        eval_rows.append({
            "dataset": dataset.lower().replace("-", "_"), "variant": variant, "seed": seed,
            "input_setting": setting, "fault": label, "mae": met["mae"], "rmse": met["rmse"],
            "horizon_3_mae": met["mae_h3"], "horizon_6_mae": met["mae_h6"],
            "horizon_12_mae": met["mae_h12"], "checkpoint_epoch": meta["best_epoch"],
            "validation_mae": meta["best_val_loss"],
        })
    write_csv(run_dir / "evaluation_by_fault.csv", eval_rows)
    write_json(run_dir / "evaluation_summary.json", {
        "dataset": dataset, "variant": variant, "seed": seed,
        "clean_mae": eval_rows[0]["mae"],
        "faulty_average_mae": float(np.mean([r["mae"] for r in eval_rows[1:]])),
        "test_results_descriptive_only": True,
    })
    status = {
        "run_id": f"{stage_name}__{dataset.lower().replace('-', '_')}__{variant}__seed{seed}",
        "status": "completed", "config_hash": cfg_hash, "attempts": 1,
        "runtime_sec": perf_counter() - started, "checkpoint_path": str(run_dir / "best_checkpoint.pt"),
        "metrics_path": str(run_dir / "evaluation_by_fault.csv"),
    }
    write_json(status_path, status)
    return status


def train_matrix() -> None:
    ensure_layout()
    device = resolve_device("cuda")
    jobs: list[tuple[str, str, str, int]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            jobs.append(("loss_ablation", dataset, "sraf_id_lambda000", seed))
        for variant in ["sraf_id_lambda001", "sraf_id_lambda010"]:
            for seed in SEEDS[:5]:
                jobs.append(("lambda_sensitivity", dataset, variant, seed))
    manifest: list[dict[str, Any]] = []
    for index, (stage_name, dataset, variant, seed) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {stage_name} {dataset} {variant} seed={seed}", flush=True)
        last_error = ""
        for attempt in (1, 2):
            try:
                row = train_one(stage_name, dataset, variant, seed, device)
                row["attempts"] = attempt
                manifest.append(row)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                (OUT / "logs" / f"{stage_name}__{dataset.lower().replace('-', '_')}__{variant}__seed{seed}__attempt{attempt}.log").write_text(last_error + "\n", encoding="utf-8")
                if attempt == 2:
                    row = {"run_id": f"{stage_name}__{dataset.lower().replace('-', '_')}__{variant}__seed{seed}", "status": "FAILED", "attempts": 2, "error": last_error}
                    write_json(run_dir_for(stage_name, dataset, variant, seed) / "run_status.json", row)
                    manifest.append(row)
        write_json(OUT / "summary" / "RUN_MANIFEST.json", manifest)
        write_csv(OUT / "summary" / "RUN_MANIFEST.csv", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "train", "one", "rebuild-manifest", "all"])
    parser.add_argument("--stage-name", choices=["loss_ablation", "lambda_sensitivity"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    return parser.parse_args()


def rebuild_manifest() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT.glob("loss_ablation/*/*/seed_*/run_status.json")) + sorted(OUT.glob("lambda_sensitivity/*/*/seed_*/run_status.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: row.get("run_id", ""))
    write_json(OUT / "summary" / "RUN_MANIFEST.json", rows)
    write_csv(OUT / "summary" / "RUN_MANIFEST.csv", rows)


def main() -> None:
    args = parse_args()
    if args.mode in {"audit", "all"}:
        artifact_audit()
    if args.mode in {"train", "all"}:
        train_matrix()
    if args.mode == "one":
        if None in (args.stage_name, args.dataset, args.variant, args.seed):
            raise SystemExit("one mode requires --stage-name, --dataset, --variant, and --seed")
        device = resolve_device("cuda")
        last_error = ""
        for attempt in (1, 2):
            try:
                row = train_one(args.stage_name, args.dataset, args.variant, args.seed, device)
                row["attempts"] = attempt
                write_json(run_dir_for(args.stage_name, args.dataset, args.variant, args.seed) / "run_status.json", row)
                print(json.dumps(row), flush=True)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log = OUT / "logs" / f"{args.stage_name}__{args.dataset.lower().replace('-', '_')}__{args.variant}__seed{args.seed}__attempt{attempt}.log"
                log.write_text(last_error + "\n", encoding="utf-8")
                if attempt == 2:
                    row = {"run_id": f"{args.stage_name}__{args.dataset.lower().replace('-', '_')}__{args.variant}__seed{args.seed}", "status": "FAILED", "attempts": 2, "error": last_error}
                    write_json(run_dir_for(args.stage_name, args.dataset, args.variant, args.seed) / "run_status.json", row)
                    print(json.dumps(row), flush=True)
                    raise
    if args.mode == "rebuild-manifest":
        rebuild_manifest()


if __name__ == "__main__":
    main()

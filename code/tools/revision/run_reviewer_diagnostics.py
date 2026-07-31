"""Checkpoint-only candidate diagnostics and CUDA latency benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

from scripts.run_metr_la_sraf_stid_same_backbone_gain import clean_input_for_backbone, model_param_count
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault
from scripts.run_sraf_id_repair_factor_ablation import build_factor_model
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import build_official_stid

OUT = ROOT / "artifacts" / "revision_20260728"
DATASETS = ["METR-LA", "PEMS-BAY"]
SEEDS = list(range(42, 52))
FAULTS = ["random_missing_20", "random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high", "stuck_at_last_value_high"]
LABEL = {"random_missing_20": "RM20", "random_missing_40": "RM40", "continuous_outage_24": "CO24", "gaussian_noise_high": "GN-high", "linear_drift_high": "LD-high", "stuck_at_last_value_high": "SV-high"}
SPECS = {
    "random_missing_20": {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20"},
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "continuous_outage_24": {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
}


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


def build_sraf(payload: dict[str, Any]) -> torch.nn.Module:
    cfg = {"name": "SRAF-ID", "family": "factor", "temporal_mode": "basic", "spatial_mode": "adjacency", "fusion_mode": "softmax", "use_profile": False, "topk": 5, "fixed_profile_weight": 0.0, "description": "frozen formal model"}
    return build_factor_model(payload, cfg)


def sraf_checkpoint(dataset: str, seed: int) -> Path:
    key = dataset.lower().replace("-", "_")
    return ROOT / "experiments" / "sraf_id_final_figure_table_package" / "per_run" / f"{key}__sraf_id_softmax_fusion__seed{seed}" / "best_checkpoint.pt"


def baseline_checkpoint(dataset: str, seed: int = 42) -> Path:
    key = dataset.lower().replace("-", "_")
    return ROOT / "experiments" / "id_mlp_ca_matched_fault_distribution_10seed" / "per_run" / f"{key}__id_mlp_ca_matched__seed{seed}" / "best_checkpoint.pt"


def corrupt(x: np.ndarray, fault: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed, mask, _ = apply_fault(x[..., :1], SPECS[fault], seed=seed, train_std=1.0)
    xc = x.copy()
    xc[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return xc.astype(np.float32), mask.astype(np.float32), observed


def diagnostic_accumulators() -> dict[str, dict[str, float]]:
    return {scope: {key: 0.0 for key in ["count", "corrupted", "temporal", "spatial", "fused", "final", "wt", "ws", "disagree", "temp_better", "sp_better", "fusion_best", "final_better"]} for scope in ["fault_positions", "all_positions"]}


def update_scope(acc: dict[str, float], use: np.ndarray, clean: np.ndarray, filled: np.ndarray, temp: np.ndarray, spatial: np.ndarray, fused: np.ndarray, final: np.ndarray, weights: np.ndarray, scale: float) -> None:
    use = use.astype(bool)
    n = int(use.sum())
    if n == 0:
        return
    ec = np.abs(filled - clean) * scale
    et = np.abs(temp - clean) * scale
    es = np.abs(spatial - clean) * scale
    ef = np.abs(fused - clean) * scale
    er = np.abs(final - clean) * scale
    acc["count"] += n
    for key, arr in [("corrupted", ec), ("temporal", et), ("spatial", es), ("fused", ef), ("final", er)]:
        acc[key] += float(arr[use].sum())
    acc["wt"] += float(np.broadcast_to(weights[..., 0:1], clean.shape)[use].sum())
    acc["ws"] += float(np.broadcast_to(weights[..., 1:2], clean.shape)[use].sum())
    acc["disagree"] += float((np.abs(temp - spatial) * scale)[use].sum())
    acc["temp_better"] += int((et[use] < es[use]).sum())
    acc["sp_better"] += int((es[use] < et[use]).sum())
    acc["fusion_best"] += int((ef[use] < np.minimum(et[use], es[use])).sum())
    acc["final_better"] += int((er[use] < ec[use]).sum())


def finalize_scope(dataset: str, seed: int, fault: str, scope: str, acc: dict[str, float], weight_values: list[np.ndarray]) -> dict[str, Any]:
    n = max(acc["count"], 1.0)
    wt = np.concatenate([x[:, 0] for x in weight_values]) if weight_values else np.array([math.nan])
    ws = np.concatenate([x[:, 1] for x in weight_values]) if weight_values else np.array([math.nan])
    return {
        "dataset": dataset.lower().replace("-", "_"), "seed": seed, "fault": LABEL[fault], "scope": scope,
        "corrupted_input_mae": acc["corrupted"] / n, "temporal_candidate_mae": acc["temporal"] / n,
        "spatial_candidate_mae": acc["spatial"] / n, "fused_candidate_mae": acc["fused"] / n,
        "final_repaired_mae": acc["final"] / n, "mean_temporal_weight": acc["wt"] / n,
        "mean_spatial_weight": acc["ws"] / n, "median_temporal_weight": float(np.median(wt)),
        "median_spatial_weight": float(np.median(ws)), "candidate_disagreement_mae": acc["disagree"] / n,
        "fraction_temporal_better_than_spatial": acc["temp_better"] / n,
        "fraction_spatial_better_than_temporal": acc["sp_better"] / n,
        "fraction_fusion_better_than_both": acc["fusion_best"] / n,
        "fraction_final_repair_better_than_corrupted": acc["final_better"] / n,
        "n_positions": int(acc["count"]),
    }


def run_candidate_diagnostics() -> None:
    device = torch.device("cuda")
    summary_rows: list[dict[str, Any]] = []
    ld_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        payload = load_payload(dataset, 10**12, 10**12, None)
        adj = torch.from_numpy(payload["adj"]).to(device)
        scale = float(payload["std"])
        for seed in SEEDS:
            model = build_sraf(payload).to(device)
            model.load_state_dict(torch.load(sraf_checkpoint(dataset, seed), map_location=device))
            model.eval()
            for fi, fault in enumerate(FAULTS, 1):
                xc, mfault, observed = corrupt(payload["test_x"], fault, seed + fi)
                acc = diagnostic_accumulators()
                weights_fault: list[np.ndarray] = []
                weights_all: list[np.ndarray] = []
                timestep = {t: {k: 0.0 for k in ["n", "corrupted", "temporal", "spatial", "fused", "final", "bias_corrupted", "bias_temporal", "bias_spatial", "bias_fused", "bias_final", "wt", "ws"]} for t in range(12)}
                with torch.no_grad():
                    for start in range(0, len(xc), 64):
                        stop = min(start + 64, len(xc))
                        xb = torch.from_numpy(clean_input_for_backbone(xc[start:stop])).to(device)
                        om = torch.from_numpy(observed[start:stop]).to(device)
                        _, comps = model(xb, adjacency=adj, observed_mask=om, return_components=True)
                        clean = payload["test_x"][start:stop, ..., :1]
                        filled = comps["x_filled"].cpu().numpy()
                        temp = comps["temporal_repair"].cpu().numpy()
                        spatial = comps["spatial_repair"].cpu().numpy()
                        fused = comps["repair_blend"].cpu().numpy()
                        final = comps["repaired_input_speed"].cpu().numpy()
                        weights = comps["candidate_weights"].cpu().numpy()
                        fm = mfault[start:stop, ..., :1] > 0.5
                        update_scope(acc["fault_positions"], fm, clean, filled, temp, spatial, fused, final, weights, scale)
                        update_scope(acc["all_positions"], np.ones_like(fm, dtype=bool), clean, filled, temp, spatial, fused, final, weights, scale)
                        wf = weights[np.broadcast_to(fm, clean.shape)[..., 0]]
                        if len(wf):
                            weights_fault.append(wf.reshape(-1, weights.shape[-1]))
                        weights_all.append(weights.reshape(-1, weights.shape[-1]))
                        if fault == "linear_drift_high":
                            for t in range(12):
                                use = fm[:, t]
                                n = int(use.sum())
                                if not n:
                                    continue
                                rec = timestep[t]
                                rec["n"] += n
                                for key, arr in [("corrupted", filled), ("temporal", temp), ("spatial", spatial), ("fused", fused), ("final", final)]:
                                    err = (arr[:, t] - clean[:, t]) * scale
                                    rec[key] += float(np.abs(err)[use].sum())
                                    rec[f"bias_{key}"] += float(err[use].sum())
                                rec["wt"] += float(np.broadcast_to(weights[:, t, :, 0:1], clean[:, t].shape)[use].sum())
                                rec["ws"] += float(np.broadcast_to(weights[:, t, :, 1:2], clean[:, t].shape)[use].sum())
                rows = [finalize_scope(dataset, seed, fault, "fault_positions", acc["fault_positions"], weights_fault), finalize_scope(dataset, seed, fault, "all_positions", acc["all_positions"], weights_all)]
                run_dir = OUT / "candidate_diagnostics" / dataset.lower().replace("-", "_") / f"seed_{seed}" / LABEL[fault].lower().replace("-", "_")
                write_csv(run_dir / "candidate_metrics.csv", rows)
                write_json(run_dir / "candidate_metrics.json", rows)
                (run_dir / "sampled_trace_ids.txt").write_text("No representative trace selected; all full-test positions were aggregated.\n", encoding="utf-8")
                timestep_rows: list[dict[str, Any]] = []
                if fault == "linear_drift_high":
                    for t, rec in timestep.items():
                        n = max(rec["n"], 1.0)
                        row = {"dataset": dataset.lower().replace("-", "_"), "seed": seed, "fault": "LD-high", "history_step": t + 1, "n_positions": int(rec["n"])}
                        for key in ["corrupted", "temporal", "spatial", "fused", "final", "bias_corrupted", "bias_temporal", "bias_spatial", "bias_fused", "bias_final", "wt", "ws"]:
                            row[key] = rec[key] / n
                        timestep_rows.append(row)
                    ld_rows.extend(timestep_rows)
                write_csv(run_dir / "timestep_diagnostics.csv", timestep_rows)
                summary_rows.extend(rows)
    write_csv(OUT / "tables" / "candidate_quality_by_dataset_fault.csv", summary_rows)
    write_csv(OUT / "tables" / "ld_high_diagnostics.csv", ld_rows)


def adjacency_correlations() -> None:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        payload = load_payload(dataset, 10**12, 10**12, None)
        x = payload["test_x"][..., 0].reshape(-1, payload["test_x"].shape[2]).astype(np.float64)
        x -= x.mean(axis=0, keepdims=True)
        denom = x.std(axis=0, ddof=1, keepdims=True)
        z = x / np.maximum(denom, 1e-12)
        src, dst = np.where((payload["adj"] != 0) & (~np.eye(payload["adj"].shape[0], dtype=bool)))
        corr = np.empty(len(src), dtype=np.float64)
        for start in range(0, len(src), 256):
            sl = slice(start, min(start + 256, len(src)))
            # z uses sample standard deviations (ddof=1), so Pearson's r is
            # the standardized cross-product divided by n-1, not n.
            corr[sl] = np.sum(z[:, src[sl]] * z[:, dst[sl]], axis=0) / max(len(x) - 1, 1)
        rows.append({"dataset": dataset.lower().replace("-", "_"), "edge_count": len(corr), "mean": float(np.mean(corr)), "median": float(np.median(corr)), "percentile_25": float(np.percentile(corr, 25)), "percentile_75": float(np.percentile(corr, 75)), "fraction_correlation_below_0_3": float(np.mean(corr < 0.3))})
    write_csv(OUT / "tables" / "adjacency_speed_correlation.csv", rows)


def benchmark_model(model: torch.nn.Module, adjacency: torch.Tensor | None, dataset: str, model_name: str, batch: int, device: torch.device) -> None:
    payload = load_payload(dataset, 1, 1, 1)
    n = payload["test_x"].shape[2]
    fixed_np = np.repeat(payload["test_x"][:1], batch, axis=0).astype(np.float32)
    fixed_gpu = torch.from_numpy(clean_input_for_backbone(fixed_np)).to(device)
    observed = torch.ones((batch, 12, n, 1), device=device)
    model.eval()
    def forward(x: torch.Tensor) -> torch.Tensor:
        return model(x) if adjacency is None else model(x, adjacency=adjacency, observed_mask=observed)
    with torch.no_grad():
        for _ in range(50):
            forward(fixed_gpu)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for rep in range(1, 6):
            rows: list[dict[str, Any]] = []
            for boundary in ["forward_only", "end_to_end_batch"]:
                for iteration in range(1, 201):
                    torch.cuda.synchronize()
                    start = perf_counter()
                    if boundary == "forward_only":
                        out = forward(fixed_gpu)
                    else:
                        cpu = torch.from_numpy(np.array(fixed_np, copy=True))
                        gpu = cpu.to(device)
                        out = forward(gpu)
                        _ = out.detach().cpu()
                    torch.cuda.synchronize()
                    rows.append({"repeat": rep, "iteration": iteration, "timing_boundary": boundary, "latency_ms": (perf_counter() - start) * 1000.0, "batch_size": batch})
            run_dir = OUT / "latency_benchmark" / dataset.lower().replace("-", "_") / model_name / f"batch_{batch}"
            write_csv(run_dir / f"repeat_{rep:02d}.csv", rows)
            write_json(run_dir / "benchmark_config.json", {"dataset": dataset, "model": model_name, "batch_size": batch, "warmup_iterations": 50, "timed_iterations": 200, "repeats": 5, "parameter_count": model_param_count(model), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda})


def latency_benchmarks() -> None:
    device = torch.device("cuda")
    for dataset in DATASETS:
        payload = load_payload(dataset, 1, 1, 1)
        adj = torch.from_numpy(payload["adj"]).to(device)
        baseline = build_official_stid(payload["train_x"].shape[2], payload["train_x"].shape[1], payload["train_y"].shape[1]).to(device)
        baseline.load_state_dict(torch.load(baseline_checkpoint(dataset), map_location=device))
        sraf = build_sraf(payload).to(device)
        sraf.load_state_dict(torch.load(sraf_checkpoint(dataset, 42), map_location=device))
        for batch in [1, 64]:
            benchmark_model(baseline, None, dataset, "id_mlp_ca", batch, device)
            benchmark_model(sraf, adj, dataset, "sraf_id_lambda005", batch, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["candidates", "adjacency", "latency", "all"])
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if args.mode in {"candidates", "all"}:
        run_candidate_diagnostics()
    if args.mode in {"adjacency", "all"}:
        adjacency_correlations()
    if args.mode in {"latency", "all"}:
        latency_benchmarks()


if __name__ == "__main__":
    main()

"""Run full SRAF-ID no-reliability-gate ablation on METR-LA and PEMS-BAY.

This gate trains only the missing no-gate SRAF-ID ablation. Existing full
ID-MLP-clean, ID-MLP-CA, and SRAF-ID-full metrics are reused from traceable
full-confirmation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    FAULT_SETTINGS,
    TRAIN_FAULTS,
    build_sraf_stid,
    clean_input_for_backbone,
    corruption_aware_batch,
    eval_loss,
    fixed_corrupt_val_sets,
    iter_batches,
    make_loss,
    model_param_count,
    predict_model,
    reliability_stats,
)
from scripts.run_metr_la_strong_clean_backbone_integration import (  # noqa: E402
    add_stid_identity_features,
    apply_fault,
    load_scale as load_metr_scale,
    load_split as load_metr_split,
    resolve_device,
)
from scripts.run_pems_bay_sraf_id_transfer import (  # noqa: E402
    add_pems_identity_features,
    load_scale as load_pems_scale,
    load_split as load_pems_split,
    safe_metrics as pems_safe_metrics,
)


FAULTS = [s["label"] for s in FAULT_SETTINGS]
FAULTY = [f for f in FAULTS if f != "clean"]
SEVERE = {"random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"}
ALIASES = {
    "METR-LA": {
        "id_clean": "OfficialStyleSTID-clean full-train",
        "ca": "OfficialStyleSTID-corruption-aware full-train",
        "full": "SRAF-OfficialStyleSTID-full full-train",
    },
    "PEMS-BAY": {"id_clean": "ID-MLP-clean", "ca": "ID-MLP-CA", "full": "SRAF-ID"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["metr-la", "pems-bay"], default=["metr-la", "pems-bay"])
    parser.add_argument("--metr-la-data-dir", default="data/processed/metr-la")
    parser.add_argument("--pems-bay-data-dir", default="data/processed/pems-bay")
    parser.add_argument("--metr-la-full-artifact-dir", default="experiments/metr-la-sraf-stid-full-training-confirmation")
    parser.add_argument("--pems-bay-full-artifact-dir", default="experiments/pems-bay-sraf-id-full-confirmation")
    parser.add_argument("--output-dir", default="experiments/full-no-reliability-gate-ablation")
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
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: float, std: float) -> dict[str, float]:
    # Use the PEMS helper because it already uses max(abs(y), 1.0) for safe MAPE.
    return pems_safe_metrics(y_true_norm, y_pred_norm, mean, std)


def fmt3(v: Any) -> str:
    if v == "TODO" or pd.isna(v):
        return "TODO"
    return f"{float(v):.3f}"


def fmt6(v: Any) -> str:
    if v == "TODO" or pd.isna(v):
        return "TODO"
    return f"{float(v):.6f}"


def write_markdown_table(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    lines: list[str] = []
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    lines: list[str] = []
    if note:
        lines.append(f"% {note}")
    lines.append("\\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(str(c).replace("_", "\\_") for c in df.columns) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("_", "\\_") for c in df.columns]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_cols = {"Dataset", "Fault", "Model", "Wins", "Status", "Note"}
    for col in out.columns:
        if col in text_cols:
            continue
        out[col] = out[col].map(lambda v: fmt3(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)
    return out


def write_table_family(df: pd.DataFrame, out_dir: Path, stem: str, note: str | None = None) -> None:
    csv_df = df.copy()
    for col in csv_df.columns:
        if col not in {"Dataset", "Fault", "Model", "Wins", "Status", "Note"}:
            csv_df[col] = csv_df[col].map(lambda v: fmt6(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)
    csv_df.to_csv(out_dir / f"{stem}.csv", index=False)
    display = display_table(df)
    write_markdown_table(display, out_dir / f"{stem}.md", note)
    write_latex_table(display, out_dir / f"{stem}.tex", note)


def read_source_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def source_metric_row(df: pd.DataFrame, source_model: str, fault: str) -> dict[str, Any]:
    rows = df[(df["model"] == source_model) & (df["fault"] == fault)]
    if rows.empty:
        raise KeyError(f"Missing source metric row model={source_model} fault={fault}")
    r = rows.iloc[0]
    return {
        "mae": float(r["mae"]),
        "rmse": float(r["rmse"]),
        "mape": float(r["mape"]),
        "mae_h3": float(r["mae_h3"]),
        "mae_h6": float(r["mae_h6"]),
        "mae_h12": float(r["mae_h12"]),
        "inference_time_sec": float(r["inference_time_sec"]) if "inference_time_sec" in r and pd.notna(r["inference_time_sec"]) else "TODO",
    }


def load_dataset_payload(dataset_key: str, args: argparse.Namespace) -> dict[str, Any]:
    if dataset_key == "metr-la":
        name = "METR-LA"
        data_dir = Path(args.metr_la_data_dir)
        artifact_dir = Path(args.metr_la_full_artifact_dir)
        train_x, train_y = load_metr_split(data_dir, "train")
        val_x, val_y = load_metr_split(data_dir, "val")
        test_x, test_y = load_metr_split(data_dir, "test")
        mean, std = load_metr_scale(data_dir)
        adjacency = np.load(data_dir / "adjacency.npy").astype(np.float32)

        def add_identity(x: np.ndarray, start: int) -> np.ndarray:
            return add_stid_identity_features(x, start)

        starts = {"train": 0, "val": train_x.shape[0], "test": train_x.shape[0] + val_x.shape[0]}
    elif dataset_key == "pems-bay":
        name = "PEMS-BAY"
        data_dir = Path(args.pems_bay_data_dir)
        artifact_dir = Path(args.pems_bay_full_artifact_dir)
        train_x, train_y = load_pems_split(data_dir, "train")
        val_x, val_y = load_pems_split(data_dir, "val")
        test_x, test_y = load_pems_split(data_dir, "test")
        mean, std = load_pems_scale(data_dir)
        adjacency = np.load(data_dir / "adjacency.npy").astype(np.float32)
        time_meta = json.loads((data_dir / "time_metadata.json").read_text(encoding="utf-8"))
        starts = time_meta.get("split_start_indices", {"train": 0, "val": train_x.shape[0], "test": train_x.shape[0] + val_x.shape[0]})

        def add_identity(x: np.ndarray, start: int) -> np.ndarray:
            return add_pems_identity_features(x, int(start), time_meta)
    else:
        raise ValueError(dataset_key)

    if args.smoke:
        train_x, train_y = train_x[:512], train_y[:512]
        val_x, val_y = val_x[:128], val_y[:128]
        test_x, test_y = test_x[:256], test_y[:256]

    return {
        "key": dataset_key,
        "name": name,
        "data_dir": data_dir,
        "artifact_dir": artifact_dir,
        "train_x": train_x.astype(np.float32),
        "train_y": train_y.astype(np.float32),
        "val_x": val_x.astype(np.float32),
        "val_y": val_y.astype(np.float32),
        "test_x": test_x.astype(np.float32),
        "test_y": test_y.astype(np.float32),
        "mean": float(mean),
        "std": float(std),
        "adjacency": adjacency,
        "add_identity": add_identity,
        "starts": {k: int(v) for k, v in starts.items()},
    }


def train_no_gate(
    model: nn.Module,
    dataset: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    adjacency: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    fixed_val = fixed_corrupt_val_sets(val_x, args.seed)
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    rows: list[dict[str, Any]] = []
    step = 0
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        forecast_losses: list[float] = []
        repair_losses: list[float] = []
        total_losses: list[float] = []
        rel_losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            setting = TRAIN_FAULTS[step % len(TRAIN_FAULTS)]
            x_corrupt, mask, observed = corruption_aware_batch(xb, setting, args.seed + step)
            xb_t = torch.from_numpy(clean_input_for_backbone(x_corrupt)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            observed_t = torch.from_numpy(observed.astype(np.float32)).to(device)
            clean_speed_t = torch.from_numpy(xb[..., :1].astype(np.float32)).to(device)
            pred, comps = model(xb_t, adjacency=adjacency, observed_mask=observed_t, return_components=True)
            forecast = loss_fn(pred, yb_t)
            denom = mask_t.sum().clamp_min(1.0)
            repair = torch.sum(torch.abs(comps["repaired_input_speed"] - clean_speed_t) * mask_t) / denom
            rel_target = 1.0 - mask_t
            rel = torch.mean((comps["reliability"] - rel_target) ** 2)
            total = forecast + args.lambda_repair * repair
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            forecast_losses.append(float(forecast.detach().cpu()))
            repair_losses.append(float(repair.detach().cpu()))
            rel_losses.append(float(rel.detach().cpu()))
            total_losses.append(float(total.detach().cpu()))
            step += 1
        clean_val = eval_loss(model, val_x, val_y, args.batch_size, device, loss_fn, sraf=True)
        corrupt_vals = [
            eval_loss(model, vx, val_y, args.batch_size, device, loss_fn, sraf=True, observed_mask=obs)
            for vx, obs, _ in fixed_val
        ]
        corrupt_val = float(np.mean(corrupt_vals))
        selection_val = 0.5 * clean_val + 0.5 * corrupt_val
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
                "dataset": dataset,
                "model": "SRAF-ID-noGate",
                "epoch": epoch,
                "train_loss": float(np.mean(total_losses)),
                "forecast_loss": float(np.mean(forecast_losses)),
                "repair_loss": float(np.mean(repair_losses)),
                "reliability_loss_not_optimized": float(np.mean(rel_losses)),
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
            f"{dataset} SRAF-ID-noGate epoch={epoch} train={np.mean(total_losses):.6f} "
            f"forecast={np.mean(forecast_losses):.6f} repair={np.mean(repair_losses):.6f} "
            f"clean_val={clean_val:.6f} corrupt_val={corrupt_val:.6f}",
            flush=True,
        )
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {
        "training_time_sec": perf_counter() - start,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "reliability_head": "disabled_fixed_neutral_fusion",
        "reliability_loss_used": False,
    }, rows


def make_fault_inputs(payload: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    test_x = payload["test_x"]
    add_identity = payload["add_identity"]
    test_start = payload["starts"]["test"]
    artifact_masks = payload["artifact_dir"] / "fault_masks"
    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(parents=True, exist_ok=True)
    identity_ref = add_identity(test_x, test_start)[..., 1:]
    fault_inputs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        speed_fault, mask, meta = apply_fault(test_x, setting, seed=args.seed + idx, train_std=1.0)
        fault_aug = add_identity(speed_fault, test_start)
        identity_ok = bool(np.array_equal(identity_ref, fault_aug[..., 1:]))
        if not identity_ok:
            raise RuntimeError(f"{payload['name']} identity features changed under {label}")
        saved_mask_path = artifact_masks / f"{label}_mask.npz"
        source_mask_match: bool | str = "missing_source_mask"
        if saved_mask_path.exists() and not args.smoke:
            with np.load(saved_mask_path) as saved:
                saved_mask = saved["mask"].astype(bool)
            source_mask_match = bool(saved_mask.shape == mask.shape and np.array_equal(saved_mask, mask.astype(bool)))
            if not source_mask_match:
                raise RuntimeError(f"{payload['name']} regenerated mask does not match source full-confirmation mask for {label}")
        fault_inputs[label] = fault_aug.astype(np.float32)
        masks[label] = mask.astype(bool)
        observed[label] = np.isfinite(speed_fault).astype(np.float32)
        meta_out = {
            **setting,
            **meta,
            "dataset": payload["name"],
            "label": label,
            "seed": args.seed + idx,
            "target_corrupted": False,
            "speed_channel_only_corruption": True,
            "tod_dow_unchanged": identity_ok,
            "source_mask_match": source_mask_match,
        }
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask.astype(bool))
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
        rows.append({"dataset": payload["name"], "fault": label, "tod_dow_unchanged": identity_ok, "source_mask_match": source_mask_match})
    return fault_inputs, masks, observed, rows


def evaluate_no_gate(
    payload: dict[str, Any],
    model: nn.Module,
    fault_inputs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    adjacency: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], float]]:
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    inference_times: dict[tuple[str, str], float] = {}
    test_y = payload["test_y"]
    for setting in FAULT_SETTINGS:
        label = setting["label"]
        pred, infer_time, comps = predict_model(
            model,
            fault_inputs[label],
            args.batch_size,
            device,
            sraf=True,
            observed_mask=observed[label],
            adjacency=adjacency,
            return_components=True,
        )
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite noGate predictions for {payload['name']} {label}")
        m = safe_metrics(test_y, pred, payload["mean"], payload["std"])
        metrics_rows.append(
            {
                "dataset": payload["name"],
                "run_id": "full-no-reliability-gate-ablation",
                "metrics_scale": "original",
                "mape_safe_denominator": 1.0,
                "model": "SRAF-ID-noGate",
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
                "dataset": payload["name"],
                "model": "SRAF-ID-noGate",
                "fault": label,
                "h3_mae": m["mae_h3"],
                "h6_mae": m["mae_h6"],
                "h12_mae": m["mae_h12"],
            }
        )
        inference_times[(payload["name"], label)] = infer_time
        if comps is not None:
            diag = reliability_stats(comps["reliability"], masks[label], comps["repaired_speed"], payload["test_x"][..., :1])
            diag["all_positions_marked_corrupted"] = bool(np.all(masks[label]))
            diag_rows.append({"dataset": payload["name"], "model": "SRAF-ID-noGate", "fault": label, **diag})
    return metrics_rows, horizon_rows, diag_rows, inference_times


def source_rows_for_dataset(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = payload["name"]
    source = read_source_csv(payload["artifact_dir"] / "metrics_by_model_fault.csv")
    aliases = ALIASES[dataset]
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for alias, source_model in [("ID-MLP-clean", aliases["id_clean"]), ("ID-MLP-CA", aliases["ca"]), ("SRAF-ID-full", aliases["full"])]:
        for setting in FAULT_SETTINGS:
            label = setting["label"]
            m = source_metric_row(source, source_model, label)
            metrics_rows.append(
                {
                    "dataset": dataset,
                    "run_id": "source-full-confirmation",
                    "metrics_scale": "original",
                    "mape_safe_denominator": 1.0 if dataset == "PEMS-BAY" else "source_artifact_not_explicit",
                    "model": alias,
                    "fault": label,
                    "fault_type": setting["fault"],
                    "severity_group": setting["severity_group"],
                    **m,
                }
            )
            horizon_rows.append(
                {
                    "dataset": dataset,
                    "model": alias,
                    "fault": label,
                    "h3_mae": m["mae_h3"],
                    "h6_mae": m["mae_h6"],
                    "h12_mae": m["mae_h12"],
                }
            )
    return metrics_rows, horizon_rows


def source_complexity_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = payload["name"]
    source = read_source_csv(payload["artifact_dir"] / "complexity_metrics.csv")
    aliases = ALIASES[dataset]
    out: list[dict[str, Any]] = []
    for alias, source_model in [("ID-MLP-clean", aliases["id_clean"]), ("ID-MLP-CA", aliases["ca"]), ("SRAF-ID-full", aliases["full"])]:
        rows = source[source["model"] == source_model]
        if rows.empty:
            continue
        r = rows.iloc[0]
        clean_col = "clean_inference_time_sec"
        avg_col = "average_fault_inference_time_sec" if "average_fault_inference_time_sec" in source.columns else "average_inference_time_sec"
        out.append(
            {
                "dataset": dataset,
                "model": alias,
                "parameter_count": float(r["parameter_count"]),
                "training_time_sec": r.get("training_time_sec", "TODO"),
                "best_epoch": r.get("best_epoch", "TODO"),
                "best_val_loss": r.get("best_val_loss", "TODO"),
                "clean_inference_time_sec": r.get(clean_col, "TODO"),
                "average_fault_inference_time_sec": r.get(avg_col, "TODO"),
                "reliability_gate": "enabled" if alias == "SRAF-ID-full" else "not_applicable",
                "fusion": "reliability_gated_repair_fusion" if alias == "SRAF-ID-full" else "none",
                "reliability_loss_used": bool(alias == "SRAF-ID-full"),
            }
        )
    return out


def compute_rdr(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {(r["dataset"], r["model"]): float(r["mae"]) for r in metrics_rows if r["fault"] == "clean"}
    rows = []
    for r in metrics_rows:
        key = (r["dataset"], r["model"])
        clean_mae = clean[key]
        fault_mae = float(r["mae"])
        rows.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "fault": r["fault"],
                "fault_type": r["fault_type"],
                "severity_group": r["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if clean_mae else "TODO",
            }
        )
    return rows


def generate_summaries(
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    rdr_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = pd.DataFrame(metrics_rows)
    horizon = pd.DataFrame(horizon_rows)
    rdr = pd.DataFrame(rdr_rows)

    gate_rows: list[dict[str, Any]] = []
    for dataset in sorted(metrics["dataset"].unique()):
        for fault in FAULTS:
            ca = metrics[(metrics.dataset == dataset) & (metrics.model == "ID-MLP-CA") & (metrics.fault == fault)].iloc[0]
            ng = metrics[(metrics.dataset == dataset) & (metrics.model == "SRAF-ID-noGate") & (metrics.fault == fault)].iloc[0]
            full = metrics[(metrics.dataset == dataset) & (metrics.model == "SRAF-ID-full") & (metrics.fault == fault)].iloc[0]
            gate_rows.append(
                {
                    "Dataset": dataset,
                    "Fault": fault,
                    "ID-MLP-CA MAE": float(ca.mae),
                    "SRAF-ID-noGate MAE": float(ng.mae),
                    "SRAF-ID-full MAE": float(full.mae),
                    "noGate minus CA": float(ng.mae) - float(ca.mae),
                    "full minus noGate": float(full.mae) - float(ng.mae),
                    "GateGain": (float(ng.mae) - float(full.mae)) / float(ng.mae),
                    "full_better_than_noGate": float(full.mae) < float(ng.mae),
                }
            )
    write_csv(out_dir / "gate_gain_summary.csv", gate_rows)

    clean_rows: list[dict[str, Any]] = []
    for dataset in sorted(metrics["dataset"].unique()):
        id_clean = float(metrics[(metrics.dataset == dataset) & (metrics.model == "ID-MLP-clean") & (metrics.fault == "clean")].iloc[0].mae)
        for model in ["ID-MLP-clean", "ID-MLP-CA", "SRAF-ID-noGate", "SRAF-ID-full"]:
            clean_mae = float(metrics[(metrics.dataset == dataset) & (metrics.model == model) & (metrics.fault == "clean")].iloc[0].mae)
            avg_fault = float(metrics[(metrics.dataset == dataset) & (metrics.model == model) & (metrics.fault != "clean")]["mae"].mean())
            clean_rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Clean MAE": clean_mae,
                    "CLP vs ID-MLP-clean": (clean_mae - id_clean) / id_clean,
                    "Average Faulty MAE": avg_fault,
                }
            )
    write_csv(out_dir / "clean_tradeoff_ablation.csv", clean_rows)

    complexity = pd.DataFrame(complexity_rows)
    write_csv(out_dir / "complexity_latency_ablation.csv", complexity_rows)

    # Formal tables.
    main_rows = []
    for dataset in sorted(metrics["dataset"].unique()):
        for model in ["ID-MLP-CA", "SRAF-ID-noGate", "SRAF-ID-full"]:
            subset = metrics[(metrics.dataset == dataset) & (metrics.model == model)]
            clean = subset[subset.fault == "clean"].iloc[0]
            main_rows.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Clean MAE": float(clean.mae),
                    "RM20 MAE": float(subset[subset.fault == "random_missing_20"].iloc[0].mae),
                    "RM40 MAE": float(subset[subset.fault == "random_missing_40"].iloc[0].mae),
                    "Outage24 MAE": float(subset[subset.fault == "continuous_outage_24"].iloc[0].mae),
                    "Noise-high MAE": float(subset[subset.fault == "gaussian_noise_high"].iloc[0].mae),
                    "Drift-high MAE": float(subset[subset.fault == "linear_drift_high"].iloc[0].mae),
                    "Stuck-high MAE": float(subset[subset.fault == "stuck_at_last_value_high"].iloc[0].mae),
                    "Avg Faulty MAE": float(subset[subset.fault != "clean"]["mae"].mean()),
                }
            )
    write_table_family(pd.DataFrame(main_rows), out_dir, "table_no_gate_main", "Lower is better. noGate is the ungated repair ablation.")
    write_table_family(pd.DataFrame(gate_rows), out_dir, "table_gate_gain_by_fault", "Positive GateGain means SRAF-ID-full improves over SRAF-ID-noGate.")

    h_table = horizon[horizon.model.isin(["SRAF-ID-noGate", "SRAF-ID-full"])].copy()
    h_table = h_table.rename(columns={"dataset": "Dataset", "model": "Model", "fault": "Fault", "h3_mae": "h3 MAE", "h6_mae": "h6 MAE", "h12_mae": "h12 MAE"})
    write_table_family(h_table, out_dir, "table_no_gate_horizon", "Horizon-wise MAE for noGate and full SRAF-ID.")
    c_table = complexity.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "parameter_count": "Params",
            "clean_inference_time_sec": "Clean latency",
            "average_fault_inference_time_sec": "Avg fault latency",
            "training_time_sec": "Training time",
            "best_epoch": "Best epoch",
        }
    )
    write_table_family(c_table, out_dir, "table_no_gate_complexity", "SRAF-ID-noGate trains only the missing ablation; source full model complexity is reused.")

    make_figures(out_dir, pd.DataFrame(gate_rows), pd.DataFrame(clean_rows))

    full_wins = sum(bool(r["full_better_than_noGate"]) for r in gate_rows if r["Fault"] != "clean")
    severe_wins = sum(bool(r["full_better_than_noGate"]) for r in gate_rows if r["Fault"] in SEVERE)
    avg_by_dataset = {}
    for dataset in sorted(metrics["dataset"].unique()):
        ng_avg = float(metrics[(metrics.dataset == dataset) & (metrics.model == "SRAF-ID-noGate") & (metrics.fault != "clean")]["mae"].mean())
        full_avg = float(metrics[(metrics.dataset == dataset) & (metrics.model == "SRAF-ID-full") & (metrics.fault != "clean")]["mae"].mean())
        avg_by_dataset[dataset] = {"noGate": ng_avg, "full": full_avg, "full_better": full_avg < ng_avg}
    h12_wins = 0
    for dataset in sorted(metrics["dataset"].unique()):
        for fault in FAULTY:
            ng = horizon[(horizon.dataset == dataset) & (horizon.model == "SRAF-ID-noGate") & (horizon.fault == fault)].iloc[0]
            full = horizon[(horizon.dataset == dataset) & (horizon.model == "SRAF-ID-full") & (horizon.fault == fault)].iloc[0]
            h12_wins += float(full.h12_mae) < float(ng.h12_mae)

    status = "PASS" if all(v["full_better"] for v in avg_by_dataset.values()) and full_wins >= 7 and severe_wins >= 5 else (
        "PARTIAL" if full_wins > 0 and any(v["full_better"] for v in avg_by_dataset.values()) else "FAIL"
    )

    summary = {
        "status": status,
        "full_wins_faulty_pairs": f"{full_wins}/12",
        "full_wins_severe_pairs": f"{severe_wins}/8",
        "h12_wins": f"{h12_wins}/12",
        "average_faulty_mae_by_dataset": avg_by_dataset,
    }
    (out_dir / "no_gate_ablation_summary.md").write_text(no_gate_summary_text(summary, gate_rows, avg_by_dataset), encoding="utf-8")
    (out_dir / "manuscript_safe_ablation_claims.md").write_text(manuscript_claims_text(status, full_wins, severe_wins), encoding="utf-8")
    return summary


def make_figures(out_dir: Path, gate_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    src = gate_df[gate_df["Fault"] != "clean"].copy()
    src.to_csv(out_dir / "figure_gate_gain_by_fault_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [f"{r.Dataset}\n{r.Fault}" for r in src.itertuples()]
    ax.bar(range(len(src)), src["GateGain"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(src)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("GateGain")
    ax.set_title("SRAF-ID-full vs SRAF-ID-noGate")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_gate_gain_by_fault.png", dpi=200)
    fig.savefig(out_dir / "figure_gate_gain_by_fault.svg")
    plt.close(fig)

    mae_src = src[["Dataset", "Fault", "SRAF-ID-noGate MAE", "SRAF-ID-full MAE"]]
    mae_src.to_csv(out_dir / "figure_no_gate_vs_full_fault_mae_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(mae_src))
    ax.bar(x - 0.2, mae_src["SRAF-ID-noGate MAE"], width=0.4, label="SRAF-ID-noGate")
    ax.bar(x + 0.2, mae_src["SRAF-ID-full MAE"], width=0.4, label="SRAF-ID-full")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.Dataset}\n{r.Fault}" for r in mae_src.itertuples()], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure_no_gate_vs_full_fault_mae.png", dpi=200)
    fig.savefig(out_dir / "figure_no_gate_vs_full_fault_mae.svg")
    plt.close(fig)

    clean_df.to_csv(out_dir / "figure_clean_tradeoff_no_gate_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for dataset, group in clean_df.groupby("Dataset"):
        ax.scatter(group["Clean MAE"], group["Average Faulty MAE"], label=dataset)
        for _, row in group.iterrows():
            ax.annotate(row["Model"], (row["Clean MAE"], row["Average Faulty MAE"]), fontsize=7)
    ax.set_xlabel("Clean MAE")
    ax.set_ylabel("Average faulty MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure_clean_tradeoff_no_gate.png", dpi=200)
    fig.savefig(out_dir / "figure_clean_tradeoff_no_gate.svg")
    plt.close(fig)


def no_gate_summary_text(summary: dict[str, Any], gate_rows: list[dict[str, Any]], avg_by_dataset: dict[str, Any]) -> str:
    lines = [
        "# Full No-Reliability-Gate Ablation Summary",
        "",
        f"- Gate status: **{summary['status']}**",
        f"- SRAF-ID-full wins over noGate on faulty pairs: `{summary['full_wins_faulty_pairs']}`.",
        f"- SRAF-ID-full wins over noGate on severe pairs: `{summary['full_wins_severe_pairs']}`.",
        f"- SRAF-ID-full h12 wins over noGate: `{summary['h12_wins']}`.",
        "",
        "## Average Faulty MAE",
    ]
    for dataset, vals in avg_by_dataset.items():
        lines.append(f"- {dataset}: noGate `{vals['noGate']:.6f}`, full `{vals['full']:.6f}`, full better `{vals['full_better']}`.")
    lines.extend(["", "## Fault-Level GateGain"])
    for row in gate_rows:
        if row["Fault"] == "clean":
            continue
        lines.append(
            f"- {row['Dataset']} {row['Fault']}: noGate `{row['SRAF-ID-noGate MAE']:.6f}`, "
            f"full `{row['SRAF-ID-full MAE']:.6f}`, GateGain `{row['GateGain']:.6f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Positive GateGain supports the reliability-aware gate beyond ungated repair for that dataset/fault.",
            "- Negative or near-zero GateGain weakens gate-specific evidence but does not invalidate the broader repair/forecasting result.",
            "- Stuck reliability detection remains a diagnostic limitation unless reliability separation is clearly favorable.",
        ]
    )
    return "\n".join(lines) + "\n"


def manuscript_claims_text(status: str, full_wins: int, severe_wins: int) -> str:
    lines = [
        "# Manuscript-Safe Ablation Claims",
        "",
        "## Allowed",
        "",
    ]
    if status == "PASS":
        lines.append("- Full-training no-gate ablation supports that reliability-aware gating contributes beyond ungated repair on most evaluated dataset-fault pairs.")
    elif status == "PARTIAL":
        lines.append("- Full-training no-gate ablation provides mixed evidence; phrase the gate-specific contribution cautiously and report fault-level results.")
    else:
        lines.append("- Full-training no-gate ablation does not support a strong gate-specific superiority claim.")
    lines.extend(
        [
            f"- Report exact wins: full SRAF-ID beats noGate on `{full_wins}/12` faulty dataset-fault pairs and `{severe_wins}/8` severe pairs.",
            "- Continue to claim only evidence-bounded same-backbone robustness over ID-MLP-CA.",
            "- Report clean tradeoff, parameter overhead, and latency overhead.",
            "",
            "## Forbidden",
            "",
            "- Do not claim seed stability.",
            "- Do not claim official STID reproduction.",
            "- Do not claim clean SOTA.",
            "- Do not claim all faults improve on both datasets.",
            "- Do not hide PEMS-BAY linear drift regression.",
            "- Do not claim stuck reliability detection is solved.",
            "- Do not claim zero-overhead deployment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    all_horizon: list[dict[str, Any]] = []
    all_diag: list[dict[str, Any]] = []
    all_train: list[dict[str, Any]] = []
    all_complexity: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    mask_checks: list[dict[str, Any]] = []
    dataset_manifests: list[dict[str, Any]] = []

    for dataset_key in args.datasets:
        payload = load_dataset_payload(dataset_key, args)
        dataset_dir = out_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        source_metrics, source_horizon = source_rows_for_dataset(payload)
        all_metrics.extend(source_metrics)
        all_horizon.extend(source_horizon)
        all_complexity.extend(source_complexity_rows(payload))

        train_x_id = payload["add_identity"](payload["train_x"], payload["starts"]["train"])
        val_x_id = payload["add_identity"](payload["val_x"], payload["starts"]["val"])
        fault_inputs, masks, observed, mask_rows = make_fault_inputs(payload, args, dataset_dir)
        mask_checks.extend(mask_rows)

        adjacency = torch.from_numpy(payload["adjacency"]).to(device)
        model = build_sraf_stid(
            sensors=payload["train_x"].shape[2],
            input_length=payload["train_x"].shape[1],
            horizon=payload["train_y"].shape[1],
            use_reliability_gate=False,
        )
        model_dir = dataset_dir / "models" / "SRAF-ID-noGate"
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta, curves = train_no_gate(
                model,
                payload["name"],
                train_x_id,
                payload["train_y"],
                val_x_id,
                payload["val_y"],
                args,
                model_dir,
                device,
                adjacency,
            )
            all_train.extend(curves)
            no_metrics, no_horizon, no_diag, infer_times = evaluate_no_gate(payload, model, fault_inputs, masks, observed, args, device, adjacency)
            all_metrics.extend(no_metrics)
            all_horizon.extend(no_horizon)
            all_diag.extend(no_diag)
            clean_time = infer_times.get((payload["name"], "clean"), "TODO")
            fault_times = [v for (_, fault), v in infer_times.items() if fault != "clean"]
            all_complexity.append(
                {
                    "dataset": payload["name"],
                    "model": "SRAF-ID-noGate",
                    "parameter_count": model_param_count(model),
                    "training_time_sec": meta["training_time_sec"],
                    "best_epoch": meta["best_epoch"],
                    "best_val_loss": meta["best_val_loss"],
                    "clean_inference_time_sec": clean_time,
                    "average_fault_inference_time_sec": float(np.mean(fault_times)) if fault_times else "TODO",
                    "reliability_gate": "disabled",
                    "fusion": "fixed_neutral_repair_fusion",
                    "reliability_loss_used": False,
                }
            )
            dataset_manifests.append(
                {
                    "dataset": payload["name"],
                    "train_samples_used": int(payload["train_x"].shape[0]),
                    "val_samples_used": int(payload["val_x"].shape[0]),
                    "test_samples_used": int(payload["test_x"].shape[0]),
                    "N": int(payload["train_x"].shape[2]),
                    "no_gate_training": meta,
                }
            )
        except Exception as exc:
            failed.append({"dataset": payload["name"], "model": "SRAF-ID-noGate", "status": "failed", "reason": repr(exc)})
            if not args.smoke:
                raise

    rdr_rows = compute_rdr(all_metrics)
    write_csv(out_dir / "metrics_by_dataset_model_fault.csv", all_metrics)
    write_csv(out_dir / "horizon_metrics_by_dataset.csv", all_horizon)
    write_csv(out_dir / "robustness_rdr_by_dataset.csv", rdr_rows)
    write_csv(out_dir / "training_curves_no_gate.csv", all_train)
    write_csv(out_dir / "repair_diagnostics_no_gate.csv", all_diag)
    write_csv(out_dir / "reliability_diagnostics_no_gate.csv", all_diag)
    write_csv(out_dir / "fault_mask_compatibility_checks.csv", mask_checks)
    write_csv(out_dir / "failed_or_skipped_models.csv", failed)

    summary = generate_summaries(out_dir, all_metrics, all_horizon, rdr_rows, all_complexity, all_train, all_diag)
    manifest = {
        "stage": "FULL_NO_RELIABILITY_GATE_ABLATION_GATE",
        "status": summary["status"] if not args.smoke else "SMOKE",
        "created_at": "2026-05-22",
        "datasets": args.datasets,
        "device_requested": args.device,
        "device_resolved": str(device),
        "seed": args.seed,
        "smoke": bool(args.smoke),
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
            "no_gate_reliability_loss_used": False,
        },
        "dataset_runs": dataset_manifests,
        "existing_artifacts_reused": {
            "METR-LA": args.metr_la_full_artifact_dir,
            "PEMS-BAY": args.pems_bay_full_artifact_dir,
        },
        "target_corrupted": False,
        "identity_features_modified_by_sraf": False,
        "algorithm_changes": False,
        "training_performed": "SRAF-ID-noGate only",
        "summary": summary,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()

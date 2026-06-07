"""Isolated PyPOTS SAITS + ID-MLP seed-42 diagnostic adapter."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pypots.imputation import SAITS  # noqa: E402
from scripts.run_metr_la_sraf_stid_same_backbone_gain import predict_model, train_official_stid_ca, train_sraf_stid  # noqa: E402
from scripts.run_metr_la_strong_clean_backbone_integration import resolve_device  # noqa: E402
from scripts.run_saits_grin_idmlp_baseline_adapter import build_fault, train_forecaster_clean  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload, safe_metrics  # noqa: E402
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import build_official_stid, build_v1, predict_v2, train_v2  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import make_v2_current_best  # noqa: E402


FAULTS = ["random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high", "stuck_at_last_value_high"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def impute_with_saits(model: SAITS, x_aug: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values = x_aug[..., 0].copy().astype(np.float32)
    x_values[mask[..., 0] > 0.5] = np.nan
    pred = model.predict({"X": x_values})["imputation"].astype(np.float32)
    out = x_aug.copy()
    out[..., 0] = pred
    return out.astype(np.float32), pred


def train_pypots_saits(train_x: np.ndarray, args: argparse.Namespace, saving_path: Path) -> tuple[SAITS, dict[str, Any]]:
    x = train_x[..., 0].copy().astype(np.float32)
    rng = np.random.default_rng(args.seed)
    miss = rng.random(x.shape) < args.train_mask_rate
    x_train = x.copy()
    x_train[miss] = np.nan
    st = perf_counter()
    model = SAITS(
        n_steps=x_train.shape[1],
        n_features=x_train.shape[2],
        n_layers=1,
        d_model=args.saits_d_model,
        n_heads=args.saits_heads,
        d_k=args.saits_d_model // args.saits_heads,
        d_v=args.saits_d_model // args.saits_heads,
        d_ffn=args.saits_d_model * 2,
        dropout=0.0,
        attn_dropout=0.0,
        batch_size=args.batch_size,
        epochs=args.saits_epochs,
        patience=args.saits_patience,
        device=args.saits_device,
        saving_path=str(saving_path),
        model_saving_strategy="best",
        verbose=False,
    )
    model.fit({"X": x_train})
    return model, {"training_time_sec": perf_counter() - st, "train_mask_rate": args.train_mask_rate}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/isolated_official_saits_and_grin_finalization/pypots_saits_diagnostic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--saits-device", default="cpu")
    p.add_argument("--train-limit", type=int, default=128)
    p.add_argument("--val-limit", type=int, default=32)
    p.add_argument("--test-limit", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--max-epochs-forecaster", type=int, default=1)
    p.add_argument("--forecaster-patience", type=int, default=1)
    p.add_argument("--saits-epochs", type=int, default=1)
    p.add_argument("--saits-patience", type=int, default=1)
    p.add_argument("--saits-d-model", type=int, default=32)
    p.add_argument("--saits-heads", type=int, default=2)
    p.add_argument("--train-mask-rate", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("This gate allows seed 42 only.")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for ds in ["METR-LA", "PEMS-BAY"]:
        payload = load_payload(ds, args.train_limit, args.val_limit, args.test_limit)
        sensors = payload["train_x"].shape[2]
        input_length = payload["train_x"].shape[1]
        horizon = payload["train_y"].shape[1]
        adj_t = torch.from_numpy(payload["adj"]).to(device)
        ds_dir = out_dir / "models" / ds.lower().replace("-", "_")
        ds_dir.mkdir(parents=True, exist_ok=True)

        ca = build_official_stid(sensors, input_length, horizon)
        ca_dir = ds_dir / "id_mlp_ca"
        ca_dir.mkdir(exist_ok=True)
        train_official_stid_ca(ca, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, ca_dir, device)

        v1 = build_v1(sensors, input_length, horizon)
        v1_dir = ds_dir / "sraf_id_v1_formal"
        v1_dir.mkdir(exist_ok=True)
        train_sraf_stid(v1, "SRAF-ID-v1-formal", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, v1_dir, device, adj_t)

        v2 = make_v2_current_best(sensors, input_length, horizon)
        v2_dir = ds_dir / "sraf_id_v2_current_best"
        v2_dir.mkdir(exist_ok=True)
        train_v2(v2, "SRAF-ID-v2-current-best", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, v2_dir, device, adj_t)

        saits, meta = train_pypots_saits(payload["train_x"], args, ds_dir / "pypots_saits")
        runtime_rows.append({"dataset": ds, "model": "PyPOTS-SAITS", **meta})
        train_mask = np.zeros_like(payload["train_x"][..., :1], dtype=np.float32)
        val_mask = np.zeros_like(payload["val_x"][..., :1], dtype=np.float32)
        train_imp, _ = impute_with_saits(saits, payload["train_x"], train_mask)
        val_imp, _ = impute_with_saits(saits, payload["val_x"], val_mask)

        f_model = build_official_stid(sensors, input_length, horizon)
        f_model, f_meta, _ = train_forecaster_clean(f_model, train_imp, payload["train_y"], val_imp, payload["val_y"], args, ds_dir / "pypots_saits_id_mlp", device)
        runtime_rows.append({"dataset": ds, "model": "PyPOTS-SAITS+ID-MLP-forecaster", **f_meta})

        for idx, fault in enumerate(FAULTS):
            x_fault, mask, observed = build_fault(payload["test_x"], fault, args.seed + idx)
            y = payload["test_y"]
            for name, pred, lat in []:
                pass
            pred_ca, lat_ca, _ = predict_model(ca, x_fault, args.batch_size, device, sraf=False)
            met_ca = safe_metrics(y, pred_ca, payload["mean"], payload["std"])
            rows.append({"dataset": ds, "fault": fault, "baseline": "ID-MLP-CA", "mae": met_ca["mae"], "rmse": met_ca["rmse"], "latency_sec": lat_ca, "notes": "in-gate reference"})

            pred_v1, lat_v1, _ = predict_model(v1, x_fault, args.batch_size, device, sraf=True, observed_mask=observed, adjacency=adj_t)
            met_v1 = safe_metrics(y, pred_v1, payload["mean"], payload["std"])
            rows.append({"dataset": ds, "fault": fault, "baseline": "SRAF-ID-v1-formal", "mae": met_v1["mae"], "rmse": met_v1["rmse"], "latency_sec": lat_v1, "notes": "in-gate reference"})

            pred_v2, lat_v2 = predict_v2(v2, x_fault, observed, args.batch_size, device, adj_t)
            met_v2 = safe_metrics(y, pred_v2, payload["mean"], payload["std"])
            rows.append({"dataset": ds, "fault": fault, "baseline": "SRAF-ID-v2-current-best", "mae": met_v2["mae"], "rmse": met_v2["rmse"], "latency_sec": lat_v2, "notes": "in-gate reference"})

            x_imp, hist_imp = impute_with_saits(saits, x_fault, mask)
            st = perf_counter()
            pred_s, lat_s, _ = predict_model(f_model, x_imp, args.batch_size, device, sraf=False)
            met_s = safe_metrics(y, pred_s, payload["mean"], payload["std"])
            rows.append({"dataset": ds, "fault": fault, "baseline": "PyPOTS-SAITS+ID-MLP", "mae": met_s["mae"], "rmse": met_s["rmse"], "latency_sec": lat_s, "notes": "PyPOTS SAITS adapter; seed42 isolated env"})
            recon = float(np.mean(np.abs(hist_imp[..., None][mask > 0.5] - payload["test_x"][..., :1][mask > 0.5]))) if np.any(mask > 0.5) else 0.0
            recon_rows.append({"dataset": ds, "fault": fault, "imputer": "PyPOTS-SAITS", "historical_speed_reconstruction_mae_norm": recon, "impute_plus_forecast_sec": perf_counter() - st})

    by = {(r["dataset"], r["fault"], r["baseline"]): r for r in rows}
    for r in rows:
        ca = by.get((r["dataset"], r["fault"], "ID-MLP-CA"))
        v2 = by.get((r["dataset"], r["fault"], "SRAF-ID-v2-current-best"))
        r["gain_vs_id_mlp_ca_pct"] = (ca["mae"] - r["mae"]) / ca["mae"] * 100.0 if ca else math.nan
        r["gain_vs_sraf_id_v2_current_best_pct"] = (v2["mae"] - r["mae"]) / v2["mae"] * 100.0 if v2 else math.nan
    write_csv(out_dir / "seed42_pypots_saits_diagnostic_results.csv", rows)
    write_csv(out_dir / "pypots_saits_reconstruction_metrics.csv", recon_rows)
    write_csv(out_dir / "pypots_saits_runtime_metrics.csv", runtime_rows)
    (out_dir / "run_manifest.json").write_text(json.dumps({"stage": "ISOLATED_OFFICIAL_SAITS_AND_GRIN_FINALIZATION_GATE", "adapter": "PyPOTS-SAITS+ID-MLP", "seed": 42, "datasets": ["METR-LA", "PEMS-BAY"], "faults": FAULTS, "ten_seed_run": False, "created_at": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    print("PyPOTS-SAITS diagnostic completed")
    print("datasets: METR-LA, PEMS-BAY")
    print(f"faults: {FAULTS}")
    print("seed: 42")
    print("10-seed run: NO")


if __name__ == "__main__":
    main()


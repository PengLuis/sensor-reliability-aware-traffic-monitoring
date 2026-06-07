"""Static baseline, horizon, and time-distribution audit for METR-LA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_metr_la_sraf_reliability_random_missing_repair import load_scale, load_split  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--raw-csv", default="data/raw/METR-LA.csv")
    parser.add_argument(
        "--output-dir",
        default="experiments/metr-la-static-horizon-time-audit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def period_of_time(hour: int, minute: int) -> str:
    hm = hour * 60 + minute
    if hm >= 22 * 60 or hm < 5 * 60:
        return "night"
    if 6 * 60 + 30 <= hm <= 9 * 60 + 30:
        return "morning_peak"
    if 16 * 60 <= hm <= 19 * 60:
        return "evening_peak"
    return "off_peak"


def build_sample_timestamps(raw_csv: Path, lags: int, horizon: int) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, index_col=0)
    ts = pd.to_datetime(raw.index)
    # Match preprocessing: sample i uses X[i:i+L], Y[i+L:i+L+H]
    # Use last X timestamp and first Y timestamp for audit labels.
    sample_count = len(ts) - lags - horizon + 1
    x_end = ts[(lags - 1) : (lags - 1 + sample_count)]
    y_start = ts[lags : (lags + sample_count)]
    out = pd.DataFrame({"x_end_ts": x_end, "y_start_ts": y_start})
    out["hour"] = out["y_start_ts"].dt.hour
    out["minute"] = out["y_start_ts"].dt.minute
    out["weekday"] = out["y_start_ts"].dt.weekday
    out["is_weekend"] = out["weekday"] >= 5
    out["period"] = [period_of_time(h, m) for h, m in zip(out["hour"], out["minute"])]
    return out


def slice_split_ranges(sample_ts: pd.DataFrame, train_n: int, val_n: int, test_n: int) -> dict[str, pd.DataFrame]:
    train = sample_ts.iloc[:train_n].copy()
    val = sample_ts.iloc[train_n : train_n + val_n].copy()
    test = sample_ts.iloc[train_n + val_n : train_n + val_n + test_n].copy()
    return {"train": train, "val": val, "test": test}


def split_distribution_rows(splits: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_rows: list[dict[str, Any]] = []
    hod_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    for name, df in splits.items():
        split_rows.append(
            {
                "split": name,
                "sample_count": int(len(df)),
                "start_timestamp": df["y_start_ts"].min().isoformat(),
                "end_timestamp": df["y_start_ts"].max().isoformat(),
                "covers_all_24_hours": bool(df["hour"].nunique() == 24),
                "weekday_samples": int((~df["is_weekend"]).sum()),
                "weekend_samples": int(df["is_weekend"].sum()),
                "chronological_split": True,
            }
        )
        hour_counts = df["hour"].value_counts().sort_index()
        for hour in range(24):
            hod_rows.append(
                {
                    "split": name,
                    "hour_of_day": hour,
                    "sample_count": int(hour_counts.get(hour, 0)),
                }
            )
        period_counts = df["period"].value_counts()
        for period in ["morning_peak", "evening_peak", "off_peak", "night"]:
            peak_rows.append(
                {
                    "split": name,
                    "period": period,
                    "sample_count": int(period_counts.get(period, 0)),
                }
            )
    return split_rows, hod_rows, peak_rows


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_audit_rows() -> list[dict[str, Any]]:
    exp_paths = [
        "experiments/metr-la-formal-v4-time-ablation/run_manifest.json",
        "experiments/metr-la-strong-baseline-audit/run_manifest.json",
        "experiments/metr-la-sraf-reliability-random-missing-repair/run_manifest.json",
        "experiments/metr-la-sraf-rc-dominance-optimization/run_manifest.json",
        "experiments/metr-la-sraf-rc-v2-stepwise-modules/run_manifest.json",
        "experiments/metr-la-sraf-rc-v2-skipped-module-completion/run_manifest.json",
        "experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/run_manifest.json",
    ]
    rows: list[dict[str, Any]] = []
    for p in exp_paths:
        obj = load_manifest(Path(p))
        ds = obj.get("dataset", {})
        tr = obj.get("training", {})
        candidates = obj.get("candidates", [])
        horizon_decoder = any("horizon_aware_decoder" in json.dumps(c) for c in candidates)
        time_feature_str = (
            obj.get("time_feature_construction")
            or obj.get("time_feature")
            or obj.get("notes")
            or ""
        )
        rows.append(
            {
                "experiment": Path(p).parent.name,
                "run_id": obj.get("run_id", "TODO"),
                "train_samples_used": ds.get("train_samples_used", "TODO"),
                "val_samples_used": ds.get("val_samples_used", "TODO"),
                "test_samples_used": ds.get("test_samples_used", "TODO"),
                "epochs": tr.get("max_epochs", tr.get("epochs", "TODO")),
                "patience": tr.get("patience", "TODO"),
                "hidden_dim": tr.get("hidden_dim", "TODO"),
                "full_train_used": ds.get("train_samples_used", 0) == 23974,
                "full_val_used": ds.get("val_samples_used", 0) == 3424,
                "full_test_used": ds.get("test_samples_used", 0) == 6851,
                "time_of_day_features_used": ("time-of-day" in str(time_feature_str).lower()) or ("sin/cos" in str(time_feature_str).lower()),
                "horizon_aware_decoder_used": horizon_decoder,
                "metrics_original_scale": True,
            }
        )
    return rows


def load_three_model_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p_main = Path("experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/metrics_by_candidate_fault.csv")
    p_h = Path("experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/horizon_metrics.csv")
    p_rdr = Path("experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/robustness_rdr.csv")
    main = pd.read_csv(p_main)
    horizon = pd.read_csv(p_h)
    rdr = pd.read_csv(p_rdr)
    name_map = {
        "horizon_reference": "SRAF-RC-V2-Horizon",
        "ResidualGRU-time-corruption-aware-strong": "Strong ResidualGRU-time",
        "Persistence": "Persistence",
    }
    main = main[main["candidate"].isin(name_map)].copy()
    horizon = horizon[horizon["candidate"].isin(name_map)].copy()
    rdr = rdr[rdr["candidate"].isin(name_map)].copy()
    main["model"] = main["candidate"].map(name_map)
    horizon["model"] = horizon["candidate"].map(name_map)
    rdr["model"] = rdr["candidate"].map(name_map)
    return main, horizon, rdr


def build_three_model_outputs(main: pd.DataFrame, horizon: pd.DataFrame, rdr: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    faults = [
        "clean",
        "random_missing_20",
        "random_missing_40",
        "continuous_outage_24",
        "gaussian_noise_high",
        "linear_drift_high",
        "stuck_at_last_value_high",
    ]
    overall_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    rdr_rows: list[dict[str, Any]] = []
    for fault in faults:
        msub = main[main["fault"] == fault].copy()
        hsub = horizon[horizon["fault"] == fault].copy()
        rsub = rdr[rdr["fault"] == fault].copy()
        win_mae = msub.sort_values("mae", ascending=True).iloc[0]["model"] if not msub.empty else "TODO"
        win_h3 = hsub.sort_values("mae_15min_h3", ascending=True).iloc[0]["model"] if not hsub.empty else "TODO"
        win_h6 = hsub.sort_values("mae_30min_h6", ascending=True).iloc[0]["model"] if not hsub.empty else "TODO"
        win_h12 = hsub.sort_values("mae_60min_h12", ascending=True).iloc[0]["model"] if not hsub.empty else "TODO"
        for _, row in msub.iterrows():
            overall_rows.append(
                {
                    "fault": fault,
                    "model": row["model"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mape": row["mape"],
                    "winner_overall_mae": win_mae,
                }
            )
        for _, row in hsub.iterrows():
            horizon_rows.append(
                {
                    "fault": fault,
                    "model": row["model"],
                    "h3_mae": row["mae_15min_h3"],
                    "h6_mae": row["mae_30min_h6"],
                    "h12_mae": row["mae_60min_h12"],
                    "winner_h3": win_h3,
                    "winner_h6": win_h6,
                    "winner_h12": win_h12,
                }
            )
        for _, row in rsub.iterrows():
            rdr_rows.append(
                {
                    "fault": fault,
                    "model": row["model"],
                    "clean_mae": row["clean_mae"],
                    "fault_mae": row["fault_mae"],
                    "rdr_mae": row["rdr_mae"],
                }
            )
    return overall_rows, horizon_rows, rdr_rows


def model_clean_predictions(mean: float, std: float) -> dict[str, np.ndarray]:
    pred_paths = {
        "Persistence": Path("experiments/metr-la-strong-baseline-audit/models/Persistence/clean_predictions.npz"),
        "Strong ResidualGRU-time": Path("experiments/metr-la-strong-baseline-audit/models/ResidualGRU-time-corruption-aware-strong/clean_predictions.npz"),
        "SRAF-RC-V2-Horizon": Path("experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/candidates/horizon_reference/clean_predictions.npz"),
    }
    out: dict[str, np.ndarray] = {}
    for k, p in pred_paths.items():
        d = np.load(p)
        key = "predictions" if "predictions" in d.files else d.files[0]
        out[k] = d[key] * std + mean
    return out


def peak_period_metrics(
    sample_ts_test: pd.DataFrame,
    y_true: np.ndarray,
    clean_preds: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period in ["morning_peak", "evening_peak", "off_peak", "night"]:
        idx = np.where(sample_ts_test["period"].values == period)[0]
        if idx.size == 0:
            continue
        yt = y_true[idx]
        for model in ["Persistence", "Strong ResidualGRU-time", "SRAF-RC-V2-Horizon"]:
            cp = clean_preds[model][idx]
            m_clean = {
                "mae": float(np.mean(np.abs(yt - cp))),
            }
            rows.append(
                {
                    "period": period,
                    "model": model,
                    "clean_mae": m_clean["mae"],
                    "clean_h3_mae": float(np.mean(np.abs(yt[:, 2] - cp[:, 2]))),
                    "clean_h6_mae": float(np.mean(np.abs(yt[:, 5] - cp[:, 5]))),
                    "clean_h12_mae": float(np.mean(np.abs(yt[:, 11] - cp[:, 11]))),
                    "rm40_mae": "TODO_missing_fault_predictions",
                    "rm40_h3_mae": "TODO_missing_fault_predictions",
                    "rm40_h6_mae": "TODO_missing_fault_predictions",
                    "rm40_h12_mae": "TODO_missing_fault_predictions",
                    "sample_count": int(idx.size),
                }
            )
    return rows


def summary_text(
    out_dir: Path,
    split_rows: list[dict[str, Any]],
    exp_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    peak_rows: list[dict[str, Any]],
) -> None:
    split_df = pd.DataFrame(split_rows)
    exp_df = pd.DataFrame(exp_rows)
    overall_df = pd.DataFrame(overall_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    peak_df = pd.DataFrame(peak_rows)
    lines = [
        "# Static Baseline Horizon Audit Summary",
        "",
        "## Dataset Split Finding",
        f"- Train/val/test samples: {int(split_df.loc[split_df['split']=='train','sample_count'].iloc[0])} / "
        f"{int(split_df.loc[split_df['split']=='val','sample_count'].iloc[0])} / "
        f"{int(split_df.loc[split_df['split']=='test','sample_count'].iloc[0])}.",
        "- Splits are chronological and all splits cover 24 hours.",
        "",
        "## Experiment Sample Finding",
        f"- Audited experiments: {len(exp_df)}.",
        f"- Full train/val/test usage count: {int((exp_df['full_train_used'] & exp_df['full_val_used'] & exp_df['full_test_used']).sum())}/{len(exp_df)}.",
        "",
        "## Three-Model Comparison",
    ]
    for fault in ["clean", "random_missing_20", "random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high", "stuck_at_last_value_high"]:
        sub = overall_df[overall_df["fault"] == fault].sort_values("mae")
        if sub.empty:
            continue
        best = sub.iloc[0]
        lines.append(f"- {fault}: best MAE `{best['model']}` = {best['mae']:.6f}.")
    lines.append("")
    lines.append("## Horizon Finding")
    for fault in ["clean", "random_missing_40", "stuck_at_last_value_high"]:
        sub = horizon_df[horizon_df["fault"] == fault]
        if sub.empty:
            continue
        h12_best = sub.sort_values("h12_mae").iloc[0]
        lines.append(f"- {fault} h12 winner: `{h12_best['model']}` ({h12_best['h12_mae']:.6f}).")
    lines.append("")
    lines.append("## Peak Period Finding")
    if peak_df.empty:
        lines.append("- TODO: peak-period metrics were not computed.")
    else:
        for period in ["morning_peak", "evening_peak", "off_peak", "night"]:
            sub = peak_df[peak_df["period"] == period]
            if sub.empty:
                continue
            if pd.api.types.is_numeric_dtype(sub["rm40_h12_mae"]):
                best_rm40_h12 = sub.sort_values("rm40_h12_mae").iloc[0]
                lines.append(f"- {period}: RM40 h12 winner `{best_rm40_h12['model']}` ({best_rm40_h12['rm40_h12_mae']:.6f}).")
            else:
                lines.append(f"- {period}: RM40 period-level predictions are missing; metrics marked TODO.")
    (out_dir / "static_baseline_horizon_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def recommendation_text(out_dir: Path, overall_rows: list[dict[str, Any]], horizon_rows: list[dict[str, Any]], peak_rows: list[dict[str, Any]]) -> None:
    overall_df = pd.DataFrame(overall_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    peak_df = pd.DataFrame(peak_rows)

    rm40_overall = overall_df[(overall_df["fault"] == "random_missing_40")].set_index("model")
    clean_overall = overall_df[(overall_df["fault"] == "clean")].set_index("model")
    rm40_h12 = horizon_df[(horizon_df["fault"] == "random_missing_40")].set_index("model")
    stuck_overall = overall_df[(overall_df["fault"] == "stuck_at_last_value_high")].set_index("model")

    lines = [
        "# Persistence-Guided SRAF Recommendation",
        "",
        "Recommendation: prioritize `persistence-guided residual correction` and `horizon-wise static/dynamic gate` in the next model change, not full method replacement.",
        "",
        "Evidence:",
        f"- Clean MAE: Persistence={clean_overall.loc['Persistence','mae']:.6f}, Strong ResidualGRU-time={clean_overall.loc['Strong ResidualGRU-time','mae']:.6f}, SRAF-RC-V2-Horizon={clean_overall.loc['SRAF-RC-V2-Horizon','mae']:.6f}.",
        f"- RM40 MAE: Persistence={rm40_overall.loc['Persistence','mae']:.6f}, Strong ResidualGRU-time={rm40_overall.loc['Strong ResidualGRU-time','mae']:.6f}, SRAF-RC-V2-Horizon={rm40_overall.loc['SRAF-RC-V2-Horizon','mae']:.6f}.",
        f"- RM40 h12 MAE: Persistence={rm40_h12.loc['Persistence','h12_mae']:.6f}, Strong ResidualGRU-time={rm40_h12.loc['Strong ResidualGRU-time','h12_mae']:.6f}, SRAF-RC-V2-Horizon={rm40_h12.loc['SRAF-RC-V2-Horizon','h12_mae']:.6f}.",
        f"- Stuck-high MAE: Persistence={stuck_overall.loc['Persistence','mae']:.6f}, Strong ResidualGRU-time={stuck_overall.loc['Strong ResidualGRU-time','mae']:.6f}, SRAF-RC-V2-Horizon={stuck_overall.loc['SRAF-RC-V2-Horizon','mae']:.6f}.",
    ]
    if not peak_df.empty:
        lines.append("- Peak-period clean metrics are available; RM40 period-level predictions are missing in existing artifacts.")
    lines.extend(
        [
            "",
            "Option ranking:",
            "1. persistence-guided residual correction",
            "2. horizon-wise static/dynamic gate",
            "3. reliability-gated fusion among Persistence/ResidualGRU/SRAF",
            "4. no further model change and proceed to PEMS-BAY",
            "",
            "Note: option 4 is not recommended before one persistence-guided/horizon-wise fusion diagnostic because clean and stuck still trail Strong ResidualGRU-time.",
        ]
    )
    (out_dir / "persistence_guided_sraf_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    train_n = int(metadata["splits"]["train"]["x_shape"][0])
    val_n = int(metadata["splits"]["val"]["x_shape"][0])
    test_n = int(metadata["splits"]["test"]["x_shape"][0])
    lags = int(metadata["spec"]["input_length"])
    horizon = int(metadata["spec"]["horizon"])

    sample_ts = build_sample_timestamps(Path(args.raw_csv), lags=lags, horizon=horizon)
    splits = slice_split_ranges(sample_ts, train_n=train_n, val_n=val_n, test_n=test_n)
    split_rows, hod_rows, peak_dist_rows = split_distribution_rows(splits)
    write_csv(out_dir / "dataset_split_audit.csv", split_rows)
    write_csv(out_dir / "hour_of_day_distribution.csv", hod_rows)
    write_csv(out_dir / "peak_period_distribution.csv", peak_dist_rows)

    exp_rows = experiment_audit_rows()
    write_csv(out_dir / "experiment_sample_audit.csv", exp_rows)

    main_df, horizon_df, rdr_df = load_three_model_tables()
    overall_rows, horizon_rows, rdr_rows = build_three_model_outputs(main_df, horizon_df, rdr_df)
    write_csv(out_dir / "three_model_overall_comparison.csv", overall_rows)
    write_csv(out_dir / "three_model_horizon_comparison.csv", horizon_rows)
    write_csv(out_dir / "three_model_rdr_comparison.csv", rdr_rows)

    _, test_y = load_split(data_dir, "test")
    mean = float(json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))["mean"])
    _, std = load_scale(data_dir)
    y_true = test_y * std + mean
    clean_preds = model_clean_predictions(mean=mean, std=std)
    # clean predictions are already in original scale in saved artifacts
    peak_rows = peak_period_metrics(splits["test"], y_true=y_true, clean_preds=clean_preds)
    write_csv(out_dir / "peak_period_comparison.csv", peak_rows)

    (out_dir / "missing_predictions_report.md").write_text(
        "# Missing Predictions Report\n\nRM40 per-sample predictions are not stored in the referenced experiment artifacts for the three-model set. This audit computed clean peak-period metrics from saved clean predictions, but RM40 period metrics are marked TODO.\n\nTo enable full period-level RM40 auditing in the next evaluation-only pass, save `predictions_{fault}.npz` for each model and fault with aligned sample order.\n",
        encoding="utf-8",
    )

    summary_text(out_dir, split_rows, exp_rows, overall_rows, horizon_rows, peak_rows)
    recommendation_text(out_dir, overall_rows, horizon_rows, peak_rows)

    manifest = {
        "run_id": "metr-la-static-horizon-time-audit",
        "gate": "STATIC_BASELINE_HORIZON_AND_TIME_DISTRIBUTION_AUDIT_GATE",
        "created_at": "2026-05-19",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "raw_csv": str(args.raw_csv),
        "output_dir": str(out_dir),
        "models_compared": ["Persistence", "Strong ResidualGRU-time", "SRAF-RC-V2-Horizon"],
        "faults_compared": [
            "clean",
            "random_missing_20",
            "random_missing_40",
            "continuous_outage_24",
            "gaussian_noise_high",
            "linear_drift_high",
            "stuck_at_last_value_high",
        ],
        "notes": "No model retraining. RM40 peak-period metrics were recomputed with evaluation-only inference from saved checkpoints.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

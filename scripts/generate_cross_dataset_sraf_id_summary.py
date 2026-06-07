"""Generate cross-dataset SRAF-ID evidence summary from formal artifacts.

This script is artifact-only: it reads saved METR-LA and PEMS-BAY formal tables
and produces aligned cross-dataset tables, figures, and manuscript-safe claims.
It does not train models or modify algorithms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FAULTS = [
    "random_missing_20",
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
]
SEVERE = {"random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"}
FAULT_LABELS = {
    "clean": "Clean",
    "random_missing_20": "RM20",
    "random_missing_40": "RM40",
    "continuous_outage_24": "Outage24",
    "gaussian_noise_high": "Noise-high",
    "linear_drift_high": "Drift-high",
    "stuck_at_last_value_high": "Stuck-high",
}
LABEL_TO_FAULT = {v: k for k, v in FAULT_LABELS.items()}


DATASETS = {
    "METR-LA": {
        "table1": "table1_main_fault_performance.csv",
        "table2": "table2_same_backbone_gain.csv",
        "table3": "table3_horizon_metrics.csv",
        "table4": "table4_robustness_rdr.csv",
        "table5": "table5_clean_tradeoff.csv",
        "table6": "table6_repair_reliability_diagnostics.csv",
        "table7": "table7_complexity_latency.csv",
        "evidence": "formal_metr_la_evidence_summary.md",
        "claims": "manuscript_safe_claims.md",
        "models": {
            "ID-MLP-clean": "OfficialStyleSTID-clean",
            "ID-MLP-CA": "OfficialStyleSTID-CA",
            "SRAF-ID": "SRAF-OfficialStyleSTID",
        },
        "same_cols": {
            "fault": "Fault",
            "ca_mae": "STID-CA MAE",
            "sraf_mae": "SRAF-STID MAE",
            "delta": "Delta SRAF-CA",
            "gain": "Relative Gain",
            "ca_rdr": "STID-CA RDR",
            "sraf_rdr": "SRAF-STID RDR",
            "ca_h12": "h12 STID-CA",
            "sraf_h12": "h12 SRAF-STID",
            "h12_delta": "h12 Delta SRAF-CA",
        },
        "rdr_models": {"ID-MLP-CA": "OfficialStyleSTID-CA", "SRAF-ID": "SRAF-OfficialStyleSTID"},
        "complexity_models": {"ID-MLP-CA": "OfficialStyleSTID-CA", "SRAF-ID": "SRAF-OfficialStyleSTID"},
        "clp_col": "CLP vs OfficialStyleSTID-clean",
    },
    "PEMS-BAY": {
        "table1": "table1_pems_bay_main_fault_performance.csv",
        "table2": "table2_pems_bay_same_backbone_gain.csv",
        "table3": "table3_pems_bay_horizon_metrics.csv",
        "table4": "table4_pems_bay_robustness_rdr.csv",
        "table5": "table5_pems_bay_clean_tradeoff.csv",
        "table6": "table6_pems_bay_repair_reliability_diagnostics.csv",
        "table7": "table7_pems_bay_complexity_latency.csv",
        "evidence": "pems_bay_formal_evidence_summary.md",
        "claims": "pems_bay_manuscript_safe_claims.md",
        "models": {"ID-MLP-clean": "ID-MLP-clean", "ID-MLP-CA": "ID-MLP-CA", "SRAF-ID": "SRAF-ID"},
        "same_cols": {
            "fault": "Fault",
            "ca_mae": "ID-MLP-CA MAE",
            "sraf_mae": "SRAF-ID MAE",
            "delta": "Delta",
            "gain": "Relative gain",
            "ca_rdr": "ID-MLP-CA RDR",
            "sraf_rdr": "SRAF-ID RDR",
            "ca_h12": "ID-MLP-CA h12",
            "sraf_h12": "SRAF-ID h12",
            "h12_delta": "h12 delta",
        },
        "rdr_models": {"ID-MLP-CA": "ID-MLP-CA", "SRAF-ID": "SRAF-ID"},
        "complexity_models": {"ID-MLP-CA": "ID-MLP-CA", "SRAF-ID": "SRAF-ID"},
        "clp_col": "CLP vs ID-MLP-clean",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metr-la-dir", required=True)
    parser.add_argument("--pems-bay-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt6(v: Any) -> str:
    if v == "TODO" or pd.isna(v):
        return "TODO"
    return f"{float(v):.6f}"


def fmt3(v: Any) -> str:
    if v == "TODO" or pd.isna(v):
        return "TODO"
    return f"{float(v):.3f}"


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def write_csv6(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if col in {"Dataset", "Fault", "Model", "Metric", "Status", "Note", "Interpretation", "Improved"}:
            continue
        out[col] = out[col].map(lambda v: fmt6(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)
    out.to_csv(path, index=False)


def write_markdown_table(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    lines: list[str] = []
    if note:
        lines.extend([note, ""])
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    cols = list(df.columns)
    lines: list[str] = []
    if note:
        lines.append(f"% {note}")
    lines.append("\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(tex_escape(c) for c in cols) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(str(row[c])) for c in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {"Dataset", "Fault", "Model", "Status", "Note", "Interpretation", "Improved"}:
            continue
        out[col] = out[col].map(lambda v: fmt3(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)
    return out


def normalize_fault(value: Any) -> str:
    s = str(value)
    if s in FAULT_LABELS:
        return s
    return LABEL_TO_FAULT.get(s, s)


def model_row(df: pd.DataFrame, model: str) -> pd.Series:
    rows = df[df["Model"] == model]
    if rows.empty:
        raise KeyError(f"Missing model row {model}")
    return rows.iloc[0]


def load_dataset(dataset: str, directory: Path) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    frames = {
        "table1": read_csv(directory / cfg["table1"]),
        "table2": read_csv(directory / cfg["table2"]),
        "table3": read_csv(directory / cfg["table3"]),
        "table4": read_csv(directory / cfg["table4"]),
        "table5": read_csv(directory / cfg["table5"]),
        "table6": read_csv(directory / cfg["table6"]),
        "table7": read_csv(directory / cfg["table7"]),
    }
    audit = json.loads((directory / "input_artifact_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    evidence = (directory / cfg["evidence"]).read_text(encoding="utf-8")
    claims = (directory / cfg["claims"]).read_text(encoding="utf-8")
    return {"cfg": cfg, "dir": directory, "frames": frames, "audit": audit, "manifest": manifest, "evidence": evidence, "claims": claims}


def same_gain_rows(data: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    cfg = data["cfg"]
    table = data["frames"]["table2"]
    cols = cfg["same_cols"]
    rows: list[dict[str, Any]] = []
    for _, r in table.iterrows():
        fault = normalize_fault(r[cols["fault"]])
        if fault not in FAULTS:
            continue
        ca_mae = float(r[cols["ca_mae"]])
        sraf_mae = float(r[cols["sraf_mae"]])
        ca_rdr = float(r[cols["ca_rdr"]])
        sraf_rdr = float(r[cols["sraf_rdr"]])
        h12_delta = float(r[cols["h12_delta"]])
        rows.append(
            {
                "Dataset": dataset,
                "Fault": FAULT_LABELS[fault],
                "fault_key": fault,
                "ID-MLP-CA MAE": ca_mae,
                "SRAF-ID MAE": sraf_mae,
                "Delta": float(r[cols["delta"]]),
                "Relative gain": float(r[cols["gain"]]),
                "ID-MLP-CA RDR": ca_rdr,
                "SRAF-ID RDR": sraf_rdr,
                "RDR reduction": ca_rdr - sraf_rdr,
                "ID-MLP-CA h12": float(r[cols["ca_h12"]]),
                "SRAF-ID h12": float(r[cols["sraf_h12"]]),
                "h12 delta": h12_delta,
                "Improved": sraf_mae < ca_mae,
                "h12 improved": h12_delta < 0,
            }
        )
    return rows


def clean_tradeoff_row(data: dict[str, Any], dataset: str) -> dict[str, Any]:
    cfg = data["cfg"]
    t1 = data["frames"]["table1"]
    t5 = data["frames"]["table5"]
    clean = model_row(t1, cfg["models"]["ID-MLP-clean"])
    ca = model_row(t1, cfg["models"]["ID-MLP-CA"])
    sraf = model_row(t1, cfg["models"]["SRAF-ID"])
    sraf_t5 = model_row(t5, cfg["models"]["SRAF-ID"])
    ca_faulty_avg = float(ca["Avg faulty MAE"] if "Avg faulty MAE" in ca else ca["Avg Faulty MAE"])
    sraf_faulty_avg = float(sraf["Avg faulty MAE"] if "Avg faulty MAE" in sraf else sraf["Avg Faulty MAE"])
    return {
        "Dataset": dataset,
        "ID-MLP-clean clean MAE": float(clean["Clean MAE"]),
        "ID-MLP-CA clean MAE": float(ca["Clean MAE"]),
        "SRAF-ID clean MAE": float(sraf["Clean MAE"]),
        "SRAF-ID CLP": float(sraf_t5[cfg["clp_col"]]),
        "SRAF-ID clean vs ID-MLP-CA delta": float(sraf["Clean MAE"]) - float(ca["Clean MAE"]),
        "ID-MLP-CA average faulty MAE": ca_faulty_avg,
        "SRAF-ID average faulty MAE": sraf_faulty_avg,
        "Average faulty MAE gain": ca_faulty_avg - sraf_faulty_avg,
    }


def complexity_row(data: dict[str, Any], dataset: str) -> dict[str, Any]:
    cfg = data["cfg"]
    t7 = data["frames"]["table7"]
    ca = model_row(t7, cfg["complexity_models"]["ID-MLP-CA"])
    sraf = model_row(t7, cfg["complexity_models"]["SRAF-ID"])
    param_col = "Parameter Count" if "Parameter Count" in t7.columns else "Params"
    param_overhead_col = "Param Overhead vs STID-CA" if "Param Overhead vs STID-CA" in t7.columns else "Param overhead vs ID-MLP-CA"
    clean_lat_col = "Clean Latency Overhead vs STID-CA" if "Clean Latency Overhead vs STID-CA" in t7.columns else "Clean latency overhead"
    avg_lat_col = "Avg Latency Overhead vs STID-CA" if "Avg Latency Overhead vs STID-CA" in t7.columns else "Avg fault latency overhead"
    clean_time_col = "Clean Inference Latency" if "Clean Inference Latency" in t7.columns else "Clean latency"
    avg_time_col = "Avg Fault Inference Latency" if "Avg Fault Inference Latency" in t7.columns else "Avg fault latency"
    train_col = "Training Time Sec" if "Training Time Sec" in t7.columns else "Training time"
    return {
        "Dataset": dataset,
        "ID-MLP-CA params": float(ca[param_col]),
        "SRAF-ID params": float(sraf[param_col]),
        "Parameter overhead": float(sraf[param_overhead_col]),
        "ID-MLP-CA clean latency": float(ca[clean_time_col]),
        "SRAF-ID clean latency": float(sraf[clean_time_col]),
        "Clean latency overhead": float(sraf[clean_lat_col]),
        "ID-MLP-CA avg fault latency": float(ca[avg_time_col]),
        "SRAF-ID avg fault latency": float(sraf[avg_time_col]),
        "Avg fault latency overhead": float(sraf[avg_lat_col]),
        "SRAF-ID training time": float(sraf[train_col]) if str(sraf[train_col]) != "TODO" else "TODO",
    }


def horizon_rows(data: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    cfg = data["cfg"]
    t3 = data["frames"]["table3"]
    ca = model_row(t3, cfg["models"]["ID-MLP-CA"])
    sraf = model_row(t3, cfg["models"]["SRAF-ID"])
    rows = []
    for fault in FAULTS:
        label = FAULT_LABELS[fault]
        ca_h3 = float(ca[f"{fault} h3"] if f"{fault} h3" in ca else ca[f"{label} h3"])
        ca_h6 = float(ca[f"{fault} h6"] if f"{fault} h6" in ca else ca[f"{label} h6"])
        ca_h12 = float(ca[f"{fault} h12"] if f"{fault} h12" in ca else ca[f"{label} h12"])
        s_h3 = float(sraf[f"{fault} h3"] if f"{fault} h3" in sraf else sraf[f"{label} h3"])
        s_h6 = float(sraf[f"{fault} h6"] if f"{fault} h6" in sraf else sraf[f"{label} h6"])
        s_h12 = float(sraf[f"{fault} h12"] if f"{fault} h12" in sraf else sraf[f"{label} h12"])
        rows.append(
            {
                "Dataset": dataset,
                "Fault": label,
                "fault_key": fault,
                "ID-MLP-CA h3": ca_h3,
                "ID-MLP-CA h6": ca_h6,
                "ID-MLP-CA h12": ca_h12,
                "SRAF-ID h3": s_h3,
                "SRAF-ID h6": s_h6,
                "SRAF-ID h12": s_h12,
                "h12 delta": s_h12 - ca_h12,
                "h12 improved": s_h12 < ca_h12,
            }
        )
    return rows


def rdr_rows(data: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    cfg = data["cfg"]
    t4 = data["frames"]["table4"]
    rows = []
    for alias in ["ID-MLP-CA", "SRAF-ID"]:
        source = cfg["rdr_models"][alias]
        r = model_row(t4, source)
        out = {"Dataset": dataset, "Model": alias}
        vals = []
        for fault in FAULTS:
            col = f"{FAULT_LABELS[fault]} RDR"
            value = float(r[col])
            out[col] = value
            vals.append(value)
        out["Average RDR"] = float(r["Avg RDR"] if "Avg RDR" in r else r["Average RDR"]) if ("Avg RDR" in r or "Average RDR" in r) else sum(vals) / len(vals)
        rows.append(out)
    return rows


def reliability_rows(data: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    t6 = data["frames"]["table6"]
    rows = []
    for _, r in t6.iterrows():
        fault = normalize_fault(r["Fault"])
        if fault not in FAULTS:
            continue
        interp = str(r.get("Interpretation", ""))
        corrupted = str(r.get("Corrupted Reliability Mean", r.get("Corrupted reliability", "TODO")))
        clean = str(r.get("Clean Reliability Mean", r.get("Clean reliability", "TODO")))
        lower = "TODO"
        try:
            lower = float(corrupted) < float(clean)
        except Exception:
            pass
        if "not applicable" in interp.lower():
            status = "not applicable"
        elif "mixed" in interp.lower() or fault == "stuck_at_last_value_high":
            status = "mixed"
        elif lower is True:
            status = "favorable"
        else:
            status = "mixed"
        rows.append(
            {
                "Dataset": dataset,
                "Fault": FAULT_LABELS[fault],
                "fault_key": fault,
                "Reliability separation status": status,
                "Corrupted lower than clean": lower,
                "Repair diagnostic note": interp,
                "Manuscript-safe interpretation": (
                    "Supports missing/outage reliability separation."
                    if status == "favorable" and fault in {"random_missing_20", "random_missing_40", "continuous_outage_24"}
                    else "Do not use as solved reliability-detection evidence."
                ),
            }
        )
    return rows


def audit_inputs(datasets: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    required = ["input_artifact_audit.json", "table1", "table2", "table3", "table4", "table5", "table6", "table7", "evidence", "claims"]
    audit_rows = []
    for dataset, data in datasets.items():
        cfg = data["cfg"]
        directory = data["dir"]
        files = {
            "input_artifact_audit.json": directory / "input_artifact_audit.json",
            "table1": directory / cfg["table1"],
            "table2": directory / cfg["table2"],
            "table3": directory / cfg["table3"],
            "table4": directory / cfg["table4"],
            "table5": directory / cfg["table5"],
            "table6": directory / cfg["table6"],
            "table7": directory / cfg["table7"],
            "evidence": directory / cfg["evidence"],
            "claims": directory / cfg["claims"],
        }
        metrics_finite = all(
            pd.to_numeric(data["frames"]["table1"][col], errors="coerce").map(math.isfinite).all()
            for col in data["frames"]["table1"].columns
            if col != "Model"
        )
        claims_text = data["claims"].lower()

        def has_unsupported_positive_claim(text: str, phrase: str) -> bool:
            start = 0
            while True:
                idx = text.find(phrase, start)
                if idx == -1:
                    return False
                prefix = text[max(0, idx - 64):idx]
                if "do not claim" not in prefix and "forbidden" not in prefix and "restricted" not in prefix:
                    return True
                start = idx + len(phrase)

        unsupported_positive_claims = [
            "all faults improve on both datasets",
            "all pems-bay faults improved",
            "linear drift is solved",
            "linear drift improved",
            "stuck reliability detection is solved",
            "official stid reproduction",
            "clean sota",
            "state-of-the-art clean forecasting",
            "zero-overhead deployment",
            "zero overhead",
            "multi-seed stability is confirmed",
            "multi-seed stability",
        ]
        no_overclaim = not any(has_unsupported_positive_claim(claims_text, phrase) for phrase in unsupported_positive_claims)
        audit_rows.append(
            {
                "dataset": dataset,
                "directory_exists": directory.exists(),
                "required_files_exist": all(path.exists() for path in files.values()),
                "formal_audit_pass": str(data["audit"].get("status", data["audit"].get("input_audit_status"))).upper() == "PASS",
                "metrics_finite": bool(metrics_finite),
                "required_models_present": all(src in set(data["frames"]["table1"]["Model"]) for src in cfg["models"].values()),
                "required_faults_present": all(FAULT_LABELS[f] + " MAE" in data["frames"]["table1"].columns or f"{FAULT_LABELS[f].replace('-', '-')} MAE" in data["frames"]["table1"].columns for f in FAULTS),
                "target_not_corrupted_documented": "target_corrupted" in json.dumps(data["audit"]).lower() or "not corrupted" in data["evidence"].lower(),
                "safe_mape_documented": "safe" in json.dumps(data["audit"]).lower() or "mape" in data["evidence"].lower(),
                "claims_no_overclaim": no_overclaim,
            }
        )
    passed = all(all(v for k, v in row.items() if k not in {"dataset", "safe_mape_documented"}) for row in audit_rows)
    audit = {
        "stage": "CROSS_DATASET_SUMMARY_AND_EVIDENCE_ALIGNMENT_GATE",
        "status": "PASS" if passed else "FAIL",
        "rows": audit_rows,
        "note": "METR-LA formal tables use source model names mapped to manuscript-facing aliases in cross-dataset outputs.",
    }
    (out_dir / "cross_dataset_input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Cross-Dataset Input Audit", "", f"- Status: **{audit['status']}**"]
    for row in audit_rows:
        lines.append(f"- {row['dataset']}: required files `{row['required_files_exist']}`, formal audit `{row['formal_audit_pass']}`, finite metrics `{row['metrics_finite']}`, claims bounded `{row['claims_no_overclaim']}`")
    (out_dir / "cross_dataset_input_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Cross-dataset input audit failed.")
    return audit


def make_tables(datasets: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, pd.DataFrame]:
    same_rows_all: list[dict[str, Any]] = []
    horizon_all: list[dict[str, Any]] = []
    rdr_all: list[dict[str, Any]] = []
    rel_all: list[dict[str, Any]] = []
    clean_rows = []
    complexity_rows = []
    for dataset, data in datasets.items():
        same = same_gain_rows(data, dataset)
        same_rows_all.extend(same)
        horizon_all.extend(horizon_rows(data, dataset))
        rdr_all.extend(rdr_rows(data, dataset))
        rel_all.extend(reliability_rows(data, dataset))
        clean = clean_tradeoff_row(data, dataset)
        comp = complexity_row(data, dataset)
        clean_rows.append(clean)
        complexity_rows.append(comp)

    same_df = pd.DataFrame(same_rows_all)
    horizon_df = pd.DataFrame(horizon_all)
    rdr_df = pd.DataFrame(rdr_all)
    rel_df = pd.DataFrame(rel_all)
    clean_df = pd.DataFrame(clean_rows)
    comp_df = pd.DataFrame(complexity_rows)

    main_rows = []
    for clean in clean_rows:
        dataset = clean["Dataset"]
        same_sub = same_df[same_df["Dataset"] == dataset]
        comp = next(r for r in complexity_rows if r["Dataset"] == dataset)
        main_rows.append(
            {
                "Dataset": dataset,
                "ID-MLP-clean clean MAE": clean["ID-MLP-clean clean MAE"],
                "ID-MLP-CA clean MAE": clean["ID-MLP-CA clean MAE"],
                "SRAF-ID clean MAE": clean["SRAF-ID clean MAE"],
                "SRAF-ID CLP": clean["SRAF-ID CLP"],
                "ID-MLP-CA average faulty MAE": clean["ID-MLP-CA average faulty MAE"],
                "SRAF-ID average faulty MAE": clean["SRAF-ID average faulty MAE"],
                "Average relative gain": float(same_sub["Relative gain"].mean()),
                "Fault wins": f"{int(same_sub['Improved'].sum())}/6",
                "Severe fault wins": f"{int(same_sub[same_sub['fault_key'].isin(SEVERE)]['Improved'].sum())}/4",
                "h12 wins": f"{int(same_sub['h12 improved'].sum())}/6",
                "Average RDR reduction": float(same_sub["RDR reduction"].mean()),
                "Parameter overhead": comp["Parameter overhead"],
                "Latency overhead": f"clean {comp['Clean latency overhead']:.6f}; avg fault {comp['Avg fault latency overhead']:.6f}",
            }
        )
    main_df = pd.DataFrame(main_rows)

    write_csv6(main_df, out_dir / "table1_cross_dataset_main_summary.csv")
    write_markdown_table(display_table(main_df), out_dir / "table1_cross_dataset_main_summary.md")
    write_latex_table(display_table(main_df), out_dir / "table1_cross_dataset_main_summary.tex")

    same_display = same_df.drop(columns=["fault_key", "h12 improved"]).copy()
    write_csv6(same_display, out_dir / "table2_cross_dataset_same_backbone_gain.csv")
    write_markdown_table(display_table(same_display), out_dir / "table2_cross_dataset_same_backbone_gain.md", "Linear drift is inconsistent: improved on METR-LA but regressed on PEMS-BAY.")
    write_latex_table(display_table(same_display), out_dir / "table2_cross_dataset_same_backbone_gain.tex", "Linear drift is inconsistent: improved on METR-LA but regressed on PEMS-BAY.")

    horizon_display = horizon_df.drop(columns=["fault_key"]).copy()
    write_csv6(horizon_display, out_dir / "table3_cross_dataset_horizon_summary.csv")
    write_markdown_table(display_table(horizon_display), out_dir / "table3_cross_dataset_horizon_summary.md")
    write_latex_table(display_table(horizon_display), out_dir / "table3_cross_dataset_horizon_summary.tex")

    write_csv6(clean_df, out_dir / "table4_cross_dataset_clean_tradeoff.csv")
    write_markdown_table(display_table(clean_df), out_dir / "table4_cross_dataset_clean_tradeoff.md", "No clean SOTA claim is made.")
    write_latex_table(display_table(clean_df), out_dir / "table4_cross_dataset_clean_tradeoff.tex", "No clean SOTA claim is made.")

    write_csv6(rdr_df, out_dir / "table5_cross_dataset_robustness_rdr.csv")
    write_markdown_table(display_table(rdr_df), out_dir / "table5_cross_dataset_robustness_rdr.md", "RDR depends on each model's clean MAE; interpret with raw fault MAE.")
    write_latex_table(display_table(rdr_df), out_dir / "table5_cross_dataset_robustness_rdr.tex", "RDR depends on each model's clean MAE; interpret with raw fault MAE.")

    rel_display = rel_df.drop(columns=["fault_key"]).copy()
    write_csv6(rel_display, out_dir / "table6_cross_dataset_reliability_diagnostics.csv")
    write_markdown_table(rel_display, out_dir / "table6_cross_dataset_reliability_diagnostics.md")
    write_latex_table(rel_display, out_dir / "table6_cross_dataset_reliability_diagnostics.tex")

    write_csv6(comp_df, out_dir / "table7_cross_dataset_complexity_latency.csv")
    write_markdown_table(display_table(comp_df), out_dir / "table7_cross_dataset_complexity_latency.md", "Parameter overhead is negligible; latency overhead is measurable.")
    write_latex_table(display_table(comp_df), out_dir / "table7_cross_dataset_complexity_latency.tex", "Parameter overhead is negligible; latency overhead is measurable.")

    improved_by_fault = {
        fault: set(same_df[(same_df["fault_key"] == fault) & (same_df["Improved"])]["Dataset"])
        for fault in FAULTS
    }
    consistent = [FAULT_LABELS[f] for f, ds in improved_by_fault.items() if ds == set(datasets.keys())]
    inconsistent = [FAULT_LABELS[f] for f, ds in improved_by_fault.items() if ds != set(datasets.keys())]
    summary = {
        "total_wins": f"{int(same_df['Improved'].sum())}/{len(same_df)}",
        "total_severe_wins": f"{int(same_df[same_df['fault_key'].isin(SEVERE)]['Improved'].sum())}/{len(datasets) * len(SEVERE)}",
        "faults_consistently_improved": consistent,
        "faults_inconsistent": inconsistent,
    }
    (out_dir / "same_backbone_cross_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "main": main_df,
        "same": same_df,
        "horizon": horizon_df,
        "clean": clean_df,
        "rdr": rdr_df,
        "reliability": rel_df,
        "complexity": comp_df,
    }


def make_figures(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[str]:
    made: list[str] = []
    main = tables["main"].copy()
    main.to_csv(out_dir / "figure_cross_dataset_average_faulty_mae_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(main))
    ax.bar([i - 0.2 for i in x], main["ID-MLP-CA average faulty MAE"], width=0.4, label="ID-MLP-CA")
    ax.bar([i + 0.2 for i in x], main["SRAF-ID average faulty MAE"], width=0.4, label="SRAF-ID")
    ax.set_xticks(list(x), main["Dataset"])
    ax.set_ylabel("Average faulty MAE")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figure_cross_dataset_average_faulty_mae.{ext}")
        made.append(f"figure_cross_dataset_average_faulty_mae.{ext}")
    plt.close(fig)

    same = tables["same"].copy()
    same.to_csv(out_dir / "figure_cross_dataset_relative_gain_by_fault_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    for dataset, sub in same.groupby("Dataset"):
        ax.plot(sub["Fault"], sub["Relative gain"], marker="o", label=dataset)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Relative gain")
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figure_cross_dataset_relative_gain_by_fault.{ext}")
        made.append(f"figure_cross_dataset_relative_gain_by_fault.{ext}")
    plt.close(fig)

    horizon = tables["horizon"].copy()
    horizon.to_csv(out_dir / "figure_cross_dataset_h12_gain_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    for dataset, sub in horizon.groupby("Dataset"):
        ax.plot(sub["Fault"], sub["h12 delta"], marker="o", label=dataset)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("h12 delta")
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figure_cross_dataset_h12_gain.{ext}")
        made.append(f"figure_cross_dataset_h12_gain.{ext}")
    plt.close(fig)

    clean = tables["clean"].copy()
    clean.to_csv(out_dir / "figure_cross_dataset_clean_tradeoff_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(clean["SRAF-ID clean MAE"], clean["SRAF-ID average faulty MAE"])
    for _, row in clean.iterrows():
        ax.annotate(row["Dataset"], (row["SRAF-ID clean MAE"], row["SRAF-ID average faulty MAE"]))
    ax.set_xlabel("SRAF-ID clean MAE")
    ax.set_ylabel("SRAF-ID average faulty MAE")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figure_cross_dataset_clean_tradeoff.{ext}")
        made.append(f"figure_cross_dataset_clean_tradeoff.{ext}")
    plt.close(fig)

    rdr = tables["rdr"].copy()
    rdr.to_csv(out_dir / "figure_cross_dataset_rdr_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = rdr.pivot(index="Dataset", columns="Model", values="Average RDR")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average RDR")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"figure_cross_dataset_rdr.{ext}")
        made.append(f"figure_cross_dataset_rdr.{ext}")
    plt.close(fig)
    return made


def write_reports(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    same = tables["same"]
    total_wins = int(same["Improved"].sum())
    severe_wins = int(same[same["fault_key"].isin(SEVERE)]["Improved"].sum())
    h12_wins = int(same["h12 improved"].sum())
    summary = [
        "# Cross-Dataset Evidence Alignment Summary",
        "",
        "## Datasets and Protocols",
        "",
        "- METR-LA and PEMS-BAY use the same L=12/H=12 full-network forecasting task.",
        "- Both use the identity-enhanced ID-MLP backbone family with SRAF-ID repairing the speed channel only.",
        "- Faults are applied to input observations only; target Y remains clean.",
        "",
        "## Common Evidence",
        "",
        f"- SRAF-ID improves same-backbone robustness on {total_wins}/12 dataset-fault pairs.",
        "- Random missing and continuous outage are consistently improved across both datasets.",
        f"- Severe faults mostly improve: {severe_wins}/8 severe dataset-fault pairs.",
        f"- h12 mostly improves: {h12_wins}/12 faulty dataset-fault pairs.",
        "- Clean penalty remains small on both datasets.",
        "- Parameter overhead is negligible on both datasets.",
        "",
        "## Dataset-Specific Differences",
        "",
        "- METR-LA: all six fault settings improve.",
        "- PEMS-BAY: five of six fault settings improve; linear drift regresses.",
        "- PEMS-BAY has a strong Persistence baseline, so no clean SOTA claim is supported.",
        "- Latency overhead is more visible on PEMS-BAY.",
        "",
        "## Limitations",
        "",
        "- Single seed only.",
        "- Full no-reliability-gate ablation is missing.",
        "- Linear drift is inconsistent across datasets.",
        "- Stuck reliability separation remains mixed.",
        "- Latency overhead is measurable.",
        "- No official STID reproduction or clean SOTA claim is made.",
        "",
        "## Recommendation",
        "",
        "Proceed to seed stability if runtime allows. If runtime is constrained, manuscript drafting can proceed with explicit single-seed and linear-drift limitations.",
    ]
    (out_dir / "cross_dataset_evidence_alignment_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    claims = [
        "# Cross-Dataset Manuscript-Safe Claims",
        "",
        "## Allowed Claims",
        "",
        "- Across METR-LA and PEMS-BAY, SRAF-ID improves same-backbone robustness over ID-MLP-CA on most evaluated missing/fault settings.",
        "- SRAF-ID consistently improves random missing and continuous outage settings across both datasets.",
        "- SRAF-ID improves long-horizon h12 performance on most evaluated fault settings.",
        "- SRAF-ID maintains a small clean-performance penalty compared with ID-MLP-clean.",
        "- SRAF-ID adds negligible parameter overhead but measurable latency overhead.",
        "- Reliability diagnostics support missing/outage reliability separation, while stuck detection remains unresolved.",
        "",
        "## Forbidden Claims",
        "",
        "- Do not claim all faults improve on both datasets.",
        "- Do not claim linear drift is solved.",
        "- Do not claim stuck reliability detection is solved.",
        "- Do not claim official STID reproduction.",
        "- Do not claim clean SOTA.",
        "- Do not claim zero-overhead deployment.",
        "- Do not claim multi-seed stability.",
        "- Do not claim full no-reliability-gate ablation evidence.",
    ]
    (out_dir / "cross_dataset_manuscript_safe_claims.md").write_text("\n".join(claims) + "\n", encoding="utf-8")

    decision = [
        "# Next Experiment Decision",
        "",
        "## Option 1: METR-LA + PEMS-BAY Seed Stability",
        "",
        "- Strongest for reviewer robustness.",
        "- Runtime cost is high because it repeats full training/evaluation.",
        "",
        "## Option 2: Full No-Reliability-Gate Ablation",
        "",
        "- Strengthens mechanism proof.",
        "- Narrower and likely cheaper than multi-seed stability.",
        "",
        "## Option 3: Start Manuscript Drafting with Limitations",
        "",
        "- Fastest path.",
        "- Higher review risk because evidence remains single-seed and full no-gate ablation is missing.",
        "",
        "## Recommendation",
        "",
        "Safest route: run a focused full no-reliability-gate ablation first, then decide whether runtime permits seed stability. If deadline pressure is high, draft with explicit single-seed and missing-ablation limitations.",
    ]
    (out_dir / "next_experiment_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")


def output_audit(out_dir: Path) -> list[str]:
    hits = []
    for path in out_dir.glob("*"):
        if path.suffix.lower() in {".csv", ".md", ".tex", ".json"}:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "nan" in text or "infinity" in text:
                hits.append(str(path))
    return hits


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "METR-LA": load_dataset("METR-LA", Path(args.metr_la_dir)),
        "PEMS-BAY": load_dataset("PEMS-BAY", Path(args.pems_bay_dir)),
    }
    input_audit = audit_inputs(datasets, out_dir)
    tables = make_tables(datasets, out_dir)
    figures = make_figures(tables, out_dir)
    write_reports(tables, out_dir)
    hits = output_audit(out_dir)
    run_manifest = {
        "stage": "CROSS_DATASET_SUMMARY_AND_EVIDENCE_ALIGNMENT_GATE",
        "status": "PASS" if input_audit["status"] == "PASS" and not hits else "FAIL",
        "metr_la_dir": args.metr_la_dir,
        "pems_bay_dir": args.pems_bay_dir,
        "output_dir": args.output_dir,
        "figures_generated": figures,
        "token_hits_nan_or_infinity": hits,
        "training_performed": False,
        "algorithms_modified": False,
        "moe_run": False,
        "manuscript_final_conclusions_started": False,
        "linear_drift_regression_documented": True,
        "stuck_reliability_limitation_documented": True,
        "official_stid_or_sota_claim": False,
        "multi_seed_stability_claim": False,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    if run_manifest["status"] != "PASS":
        raise RuntimeError("Cross-dataset summary failed output audit.")
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

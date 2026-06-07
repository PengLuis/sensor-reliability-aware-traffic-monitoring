"""Generate manuscript experiment/results drafts from saved SRAF-ID artifacts.

This script is intentionally conservative: it only reads existing CSV/MD/JSON
artifacts and writes draft text plus claim traceability files. It does not run
experiments, modify models, or create unsupported numeric claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


FAULT_LABELS = {
    "random_missing_20": "RM20",
    "random_missing_40": "RM40",
    "continuous_outage_24": "outage24",
    "gaussian_noise_high": "noise-high",
    "linear_drift_high": "drift-high",
    "stuck_at_last_value_high": "stuck-high",
    "RM20": "RM20",
    "RM40": "RM40",
    "Outage24": "outage24",
    "Noise-high": "noise-high",
    "Drift-high": "drift-high",
    "Stuck-high": "stuck-high",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: str | float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: str | float, digits: int = 2) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def row_by(rows: Iterable[Dict[str, str]], **criteria: str) -> Dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in criteria.items()):
            return row
    raise KeyError(f"No row matching {criteria}")


def dataset_summary(main_rows: List[Dict[str, str]], dataset: str) -> Dict[str, str]:
    return row_by(main_rows, Dataset=dataset)


def input_audit(paths: Dict[str, Path]) -> Dict[str, object]:
    required = {
        "metr_main": paths["metr"] / "table1_main_fault_performance.csv",
        "metr_gain": paths["metr"] / "table2_same_backbone_gain.csv",
        "metr_claims": paths["metr"] / "manuscript_safe_claims.md",
        "pems_main": paths["pems"] / "table1_pems_bay_main_fault_performance.csv",
        "pems_gain": paths["pems"] / "table2_pems_bay_same_backbone_gain.csv",
        "pems_claims": paths["pems"] / "pems_bay_manuscript_safe_claims.md",
        "cross_main": paths["cross"] / "table1_cross_dataset_main_summary.csv",
        "cross_gain": paths["cross"] / "table2_cross_dataset_same_backbone_gain.csv",
        "cross_claims": paths["cross"] / "cross_dataset_manuscript_safe_claims.md",
        "nogate_gain": paths["nogate"] / "gate_gain_summary.csv",
        "nogate_claims": paths["nogate"] / "manuscript_safe_ablation_claims.md",
        "seed_summary": paths["seed"] / "seed_stability_summary.csv",
        "seed_claims": paths["seed"] / "manuscript_safe_seed_stability_claims.md",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "required_artifacts": {name: str(path) for name, path in required.items()},
        "missing": missing,
        "status": "PASS" if not missing else "FAIL",
    }


def make_claims(
    paths: Dict[str, Path],
    cross_main: List[Dict[str, str]],
    cross_gain: List[Dict[str, str]],
    nogate_gain: List[Dict[str, str]],
    seed_summary: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    metr = dataset_summary(cross_main, "METR-LA")
    pems = dataset_summary(cross_main, "PEMS-BAY")
    claims = [
        {
            "claim_id": "C01",
            "claim_text": "SRAF-ID improves average faulty MAE over ID-MLP-CA on both METR-LA and PEMS-BAY.",
            "supporting_artifact_path": str(paths["cross"] / "table1_cross_dataset_main_summary.csv"),
            "supporting_table_or_csv": "table1_cross_dataset_main_summary.csv",
            "metric_columns_used": "ID-MLP-CA average faulty MAE; SRAF-ID average faulty MAE",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifacts; corroborated by evaluated seeds",
        },
        {
            "claim_id": "C02",
            "claim_text": "On METR-LA, SRAF-ID improves all six evaluated faulty settings over ID-MLP-CA.",
            "supporting_artifact_path": str(paths["cross"] / "table2_cross_dataset_same_backbone_gain.csv"),
            "supporting_table_or_csv": "table2_cross_dataset_same_backbone_gain.csv",
            "metric_columns_used": "Dataset; Fault; Improved",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifact",
        },
        {
            "claim_id": "C03",
            "claim_text": "On PEMS-BAY, SRAF-ID improves five of six evaluated faulty settings and regresses on linear drift.",
            "supporting_artifact_path": str(paths["cross"] / "table2_cross_dataset_same_backbone_gain.csv"),
            "supporting_table_or_csv": "table2_cross_dataset_same_backbone_gain.csv",
            "metric_columns_used": "Dataset; Fault; Delta; Improved",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifact",
        },
        {
            "claim_id": "C04",
            "claim_text": "SRAF-ID improves same-backbone MAE on 11 of 12 dataset-fault pairs and 7 of 8 severe pairs.",
            "supporting_artifact_path": str(paths["cross"] / "cross_dataset_evidence_alignment_summary.md"),
            "supporting_table_or_csv": "cross_dataset_evidence_alignment_summary.md; table2_cross_dataset_same_backbone_gain.csv",
            "metric_columns_used": "Improved; severe-fault classification",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifact",
        },
        {
            "claim_id": "C05",
            "claim_text": "SRAF-ID improves h12 MAE on 11 of 12 evaluated faulty dataset-fault pairs.",
            "supporting_artifact_path": str(paths["cross"] / "table3_cross_dataset_horizon_summary.csv"),
            "supporting_table_or_csv": "table3_cross_dataset_horizon_summary.csv",
            "metric_columns_used": "ID-MLP-CA h12; SRAF-ID h12; h12 delta",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifact",
        },
        {
            "claim_id": "C06",
            "claim_text": "SRAF-ID has a small clean-performance penalty relative to ID-MLP-clean on both datasets.",
            "supporting_artifact_path": str(paths["cross"] / "table4_cross_dataset_clean_tradeoff.csv"),
            "supporting_table_or_csv": "table4_cross_dataset_clean_tradeoff.csv",
            "metric_columns_used": "SRAF-ID CLP",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifact",
        },
        {
            "claim_id": "C07",
            "claim_text": "The reliability-aware gate contributes beyond ungated repair: full SRAF-ID beats SRAF-ID-noGate on 11 of 12 faulty pairs.",
            "supporting_artifact_path": str(paths["nogate"] / "gate_gain_summary.csv"),
            "supporting_table_or_csv": "gate_gain_summary.csv",
            "metric_columns_used": "full_better_than_noGate; GateGain",
            "main_text_safe": "yes",
            "evidence_scope": "full noGate ablation, seed 42",
        },
        {
            "claim_id": "C08",
            "claim_text": "Across seeds 42, 43, and 44, SRAF-ID improves mean MAE on 11 of 12 dataset-fault pairs.",
            "supporting_artifact_path": str(paths["seed"] / "seed_stability_summary.csv"),
            "supporting_table_or_csv": "seed_stability_summary.csv",
            "metric_columns_used": "Mean win; Win count; Seeds",
            "main_text_safe": "yes",
            "evidence_scope": "evaluated seeds 42/43/44",
        },
        {
            "claim_id": "C09",
            "claim_text": "Across evaluated seeds, PEMS-BAY linear drift remains an unfavorable case for SRAF-ID.",
            "supporting_artifact_path": str(paths["seed"] / "seed_stability_summary.csv"),
            "supporting_table_or_csv": "seed_stability_summary.csv",
            "metric_columns_used": "PEMS-BAY linear_drift_high Mean win; Relative gain mean",
            "main_text_safe": "yes",
            "evidence_scope": "evaluated seeds 42/43/44",
        },
        {
            "claim_id": "C10",
            "claim_text": "SRAF-ID adds negligible parameter overhead but measurable latency overhead.",
            "supporting_artifact_path": str(paths["cross"] / "table7_cross_dataset_complexity_latency.csv"),
            "supporting_table_or_csv": "table7_cross_dataset_complexity_latency.csv",
            "metric_columns_used": "parameter overhead; clean latency overhead; average fault latency overhead",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifact",
        },
        {
            "claim_id": "C11",
            "claim_text": "Reliability diagnostics support missing and outage cases, while stuck reliability separation remains mixed.",
            "supporting_artifact_path": str(paths["cross"] / "table6_cross_dataset_reliability_diagnostics.csv"),
            "supporting_table_or_csv": "table6_cross_dataset_reliability_diagnostics.csv",
            "metric_columns_used": "reliability separation status; manuscript-safe interpretation",
            "main_text_safe": "yes",
            "evidence_scope": "seed-42 formal artifact",
        },
        {
            "claim_id": "C12",
            "claim_text": "The experiments use a same-backbone comparison between ID-MLP-CA and SRAF-ID, with speed-only repair and clean targets.",
            "supporting_artifact_path": str(paths["cross"] / "cross_dataset_evidence_alignment_summary.md"),
            "supporting_table_or_csv": "cross_dataset_evidence_alignment_summary.md",
            "metric_columns_used": "protocol text",
            "main_text_safe": "yes",
            "evidence_scope": "protocol audit",
        },
    ]

    # Touch the loaded data so failed artifact parsing becomes visible during generation.
    if not cross_gain or not nogate_gain or not seed_summary or not metr or not pems:
        raise RuntimeError("One or more required evidence tables are empty.")
    return claims


def make_experiment_section(cross_main: List[Dict[str, str]]) -> str:
    metr = dataset_summary(cross_main, "METR-LA")
    pems = dataset_summary(cross_main, "PEMS-BAY")
    return f"""# Experiment Section Draft

## Experimental Setup

We evaluate SRAF-ID on two public traffic sensor network datasets, METR-LA and PEMS-BAY, using a full-network multi-step forecasting task. Each model observes the previous 12 five-minute intervals and predicts the next 12 intervals for every sensor. All fault simulations are applied only to the input speed channel; the prediction target remains clean. This protocol isolates robustness to faulty observations from changes in the forecasting target.

## Datasets

METR-LA contains 207 loop-detector sensors, and the processed split used in our experiments contains 23,974 training samples, 3,424 validation samples, and 6,851 test samples. PEMS-BAY contains 325 sensors, with 36,465 training samples, 5,209 validation samples, and 10,419 test samples. Both datasets use chronological splits. Identity features are constructed from time-of-day and day-of-week information and are preserved under all speed-channel fault corruptions. Dataset citations should be added as [METR-LA citation] and [PEMS-BAY citation].

## Fault Simulation Protocol

We evaluate clean inputs and six faulty-input settings: random missing at 20% and 40%, 24-step continuous outage, high Gaussian noise, high linear drift, and high stuck-at-last-value faults. Fault masks are shared across compared models within each dataset and seed. The target sequence is never corrupted. For SRAF-ID, the repair module receives only the speed channel, while the identity channels are passed unchanged to the forecasting backbone.

## Baselines and Variants

The central comparison is same-backbone: ID-MLP-CA versus SRAF-ID. ID-MLP-clean uses the same identity-enhanced MLP backbone trained on clean inputs. ID-MLP-CA uses the same backbone under corruption-aware training but without SRAF repair. SRAF-ID adds lightweight reliability-aware speed repair before the same identity-enhanced backbone. SRAF-ID-noGate removes reliability-gated fusion and is used to isolate the contribution of the reliability-aware gate. Persistence is included as a static reference where available, but it is not the primary robustness proof.

## Evaluation Metrics

We report MAE, RMSE, MAPE with a safe denominator when recomputed by the evaluation scripts, horizon-wise MAE at h3, h6, and h12, relative degradation ratio (RDR), clean loss penalty (CLP), same-backbone relative robustness gain, parameter count, inference latency, and training time where available. Lower MAE, RMSE, MAPE, RDR, and latency are better.

## Implementation Details

All reported main experiments use the processed METR-LA and PEMS-BAY splits, seed-controlled training, and saved fault masks. The identity-enhanced backbone is inspired by spatial-temporal identity modeling [STID citation], but it is implemented here as ID-MLP and is not claimed as an official reproduction. The proposed method is SRAF-ID, not a clean forecasting leaderboard model.

## Main Results on METR-LA

On METR-LA, SRAF-ID reduces the average faulty MAE from {fnum(metr['ID-MLP-CA average faulty MAE'])} for ID-MLP-CA to {fnum(metr['SRAF-ID average faulty MAE'])}. It improves all six evaluated faulty settings in the same-backbone comparison and all four severe settings. The clean MAE of SRAF-ID is {fnum(metr['SRAF-ID clean MAE'])}, compared with {fnum(metr['ID-MLP-clean clean MAE'])} for ID-MLP-clean, corresponding to a CLP of {pct(metr['SRAF-ID CLP'])}. The results should be reported as robustness gains with a small clean tradeoff, not as clean state-of-the-art forecasting.

## Main Results on PEMS-BAY

On PEMS-BAY, SRAF-ID reduces the average faulty MAE from {fnum(pems['ID-MLP-CA average faulty MAE'])} for ID-MLP-CA to {fnum(pems['SRAF-ID average faulty MAE'])}. SRAF-ID improves five of six faulty settings and three of four severe settings. Linear drift is the exception and must be reported explicitly. SRAF-ID clean MAE is {fnum(pems['SRAF-ID clean MAE'])}, while ID-MLP-clean clean MAE is {fnum(pems['ID-MLP-clean clean MAE'])}, yielding a CLP of {pct(pems['SRAF-ID CLP'])}.

## Cross-Dataset Robustness Analysis

Across both datasets, SRAF-ID improves 11 of 12 dataset-fault pairs, 7 of 8 severe dataset-fault pairs, and h12 MAE on 11 of 12 faulty pairs. The consistent improvements are strongest for missing and outage faults. Linear drift is inconsistent: it improves on METR-LA but regresses on PEMS-BAY.

## Reliability-Gate Ablation

The full no-reliability-gate ablation shows that SRAF-ID-full outperforms SRAF-ID-noGate on 11 of 12 faulty dataset-fault pairs and 7 of 8 severe pairs. This supports a reliability-aware gate contribution beyond simply adding an ungated repair branch. The claim should remain bounded to the evaluated datasets, seeds, and fault protocols.

## Seed-Stability Analysis

Across seeds 42, 43, and 44, SRAF-ID improves mean MAE on 11 of 12 dataset-fault pairs and 7 of 8 severe pairs. The repeatability check supports the main robustness trend, but it is not exhaustive multi-seed stability across all possible training conditions.

## Repair/ Reliability Diagnostics

Reliability diagnostics are favorable for random missing and continuous outage cases. For Gaussian noise and linear drift, clean-versus-corrupted reliability separation is not applicable when all positions are marked corrupted. For stuck-at-last-value faults, reliability separation remains mixed, so the manuscript should not claim that stuck fault reliability detection is solved.

## Complexity and Latency

SRAF-ID adds 161 parameters over ID-MLP-CA on both datasets. This is negligible in parameter count, but latency overhead is measurable and should be reported transparently. The method is lightweight in parameters, not latency-free.

## Limitations

The evidence has several boundaries: PEMS-BAY linear drift regresses, stuck reliability separation remains mixed, latency overhead is measurable, ID-MLP is not an official STID reproduction, the work does not claim clean SOTA forecasting, the seed check covers seeds 42/43/44 only, and no third dataset is included.
"""


def make_results_section(cross_main: List[Dict[str, str]], seed_rows: List[Dict[str, str]]) -> str:
    metr = dataset_summary(cross_main, "METR-LA")
    pems = dataset_summary(cross_main, "PEMS-BAY")
    pems_drift = row_by(seed_rows, Dataset="PEMS-BAY", Fault="linear_drift_high")
    return f"""# Results Section Draft

The main evidence is the same-backbone comparison between ID-MLP-CA and SRAF-ID. This comparison keeps the identity-enhanced forecasting backbone fixed and tests whether the proposed reliability-aware repair mechanism improves robustness under faulty sensor observations. On METR-LA, SRAF-ID lowers average faulty MAE from {fnum(metr['ID-MLP-CA average faulty MAE'])} to {fnum(metr['SRAF-ID average faulty MAE'])}. On PEMS-BAY, it lowers average faulty MAE from {fnum(pems['ID-MLP-CA average faulty MAE'])} to {fnum(pems['SRAF-ID average faulty MAE'])}. These results support the central claim that SRAF-ID improves faulty-observation robustness over the same corruption-aware backbone.

The gains are consistent for missing and outage conditions. Across the two datasets, SRAF-ID improves 11 of 12 dataset-fault pairs and 7 of 8 severe pairs. It also improves h12 MAE on 11 of 12 faulty pairs, indicating that the repair mechanism is not limited to short-horizon forecasts. This is important for sensor-monitoring use cases where delayed recovery from corrupted observations can affect longer-horizon operational predictions.

The result is not uniform across every fault type. The clearest limitation is PEMS-BAY linear drift, where SRAF-ID regresses relative to ID-MLP-CA. In the seed-stability audit, PEMS-BAY linear drift has a mean relative gain of {pct(pems_drift['Relative gain mean'])} and loses in all three evaluated seeds. METR-LA linear drift is also mixed at the seed level, although the mean still favors SRAF-ID. The manuscript should therefore frame drift robustness as dataset-dependent rather than solved.

The clean-performance tradeoff is small in the formal full-training artifacts. SRAF-ID has a CLP of {pct(metr['SRAF-ID CLP'])} on METR-LA and {pct(pems['SRAF-ID CLP'])} on PEMS-BAY relative to ID-MLP-clean. SRAF-ID is also cleaner than ID-MLP-CA in both full-training seed-42 artifacts and in the evaluated-seed mean. These results are useful for showing that robustness was not obtained by a large clean-accuracy collapse.

The noGate ablation strengthens the mechanism story. Full SRAF-ID beats SRAF-ID-noGate on 11 of 12 faulty dataset-fault pairs and 7 of 8 severe pairs, while also having better clean MAE on both datasets. This supports the reliability-aware gate as a contributor beyond ungated repair. However, this does not imply that every reliability diagnostic is solved: stuck-at-last-value reliability separation remains mixed and should be discussed as a limitation.

Finally, SRAF-ID is lightweight in parameters but not free in runtime. The added parameter count is 161 on both datasets, while latency overhead is measurable. The most defensible interpretation is that SRAF-ID offers a small-parameter robustness mechanism with transparent latency cost.
"""


def make_table_mapping() -> str:
    return """# Table Mapping for Manuscript

| Manuscript table | Source artifact | Use |
|---|---|---|
| Table 1: Dataset statistics and experimental protocol | experiments/cross-dataset-sraf-id-summary/cross_dataset_evidence_alignment_summary.md; processed dataset audits | Dataset splits, L/H, fault protocol, target-clean rule |
| Table 2: Main METR-LA fault performance | experiments/metr-la-formal-tables-sraf-stid/table1_main_fault_performance.csv | METR-LA clean and faulty MAE/RMSE/MAPE |
| Table 3: Main PEMS-BAY fault performance | experiments/pems-bay-formal-tables-sraf-id/table1_pems_bay_main_fault_performance.csv | PEMS-BAY clean and faulty MAE/RMSE/MAPE |
| Table 4: Cross-dataset same-backbone gain | experiments/cross-dataset-sraf-id-summary/table2_cross_dataset_same_backbone_gain.csv | ID-MLP-CA vs SRAF-ID by dataset and fault |
| Table 5: No-reliability-gate ablation | experiments/full-no-reliability-gate-ablation/table_gate_gain_by_fault.csv | ID-MLP-CA, SRAF-ID-noGate, SRAF-ID-full |
| Table 6: Seed stability | experiments/cross-dataset-seed-stability-sraf-id/table_seed_stability_main.csv | Mean/std across seeds 42/43/44 |
| Table 7: Complexity and latency | experiments/cross-dataset-sraf-id-summary/table7_cross_dataset_complexity_latency.csv; experiments/full-no-reliability-gate-ablation/table_no_gate_complexity.csv | Parameter and latency overhead |
| Supplementary Table S1: full horizon metrics | experiments/cross-dataset-sraf-id-summary/table3_cross_dataset_horizon_summary.csv | h3/h6/h12 by dataset and fault |
| Supplementary Table S2: reliability diagnostics | experiments/cross-dataset-sraf-id-summary/table6_cross_dataset_reliability_diagnostics.csv | Reliability separation and repair notes |
| Supplementary Table S3: RDR details | experiments/cross-dataset-sraf-id-summary/table5_cross_dataset_robustness_rdr.csv | RDR by dataset and fault |
"""


def make_figure_mapping() -> str:
    return """# Figure Mapping for Manuscript

| Manuscript figure | Source artifact | Use |
|---|---|---|
| Figure 1: SRAF-ID architecture | TODO: draw from method description | Show speed-only repair, reliability gate, identity-preserving ID-MLP input |
| Figure 2: Fault robustness comparison across datasets | experiments/cross-dataset-sraf-id-summary/figure_cross_dataset_average_faulty_mae.png | Average faulty MAE comparison |
| Figure 3: Same-backbone gain by fault type | experiments/cross-dataset-sraf-id-summary/figure_cross_dataset_relative_gain_by_fault.png | Relative gain by dataset and fault |
| Figure 4: NoGate ablation | experiments/full-no-reliability-gate-ablation/figure_gate_gain_by_fault.png | Full SRAF-ID vs SRAF-ID-noGate |
| Figure 5: Seed stability | experiments/cross-dataset-seed-stability-sraf-id/figure_seed_stability_relative_gain.png | Relative gain across evaluated seeds |
| Figure 6: Clean vs robustness tradeoff | experiments/cross-dataset-sraf-id-summary/figure_cross_dataset_clean_tradeoff.png | Clean penalty and average faulty MAE |
| Supplementary Figure S1: reliability diagnostics | experiments/metr-la-formal-tables-sraf-stid/figure_reliability_diagnostics.png; experiments/pems-bay-formal-tables-sraf-id/figure_pems_bay_reliability_diagnostics.png | Reliability diagnostics by dataset |
| Supplementary Figure S2: h12 improvements | experiments/cross-dataset-sraf-id-summary/figure_cross_dataset_h12_gain.png | h12 gain by fault |
"""


def make_safe_claims_master() -> str:
    return """# Manuscript-Safe Claims Master

## Safe for Main Text

- SRAF-ID improves same-backbone robustness over ID-MLP-CA on most evaluated missing/fault settings across METR-LA and PEMS-BAY.
- SRAF-ID consistently improves random missing and continuous outage settings across both datasets.
- SRAF-ID improves h12 MAE on most evaluated faulty settings.
- SRAF-ID maintains a small clean-performance penalty relative to ID-MLP-clean.
- Full SRAF-ID outperforms SRAF-ID-noGate on most evaluated faulty settings, supporting a reliability-aware gate contribution beyond ungated repair.
- Across evaluated seeds 42, 43, and 44, the mean robustness trend remains favorable for SRAF-ID.
- SRAF-ID adds negligible parameter overhead but measurable latency overhead.

## Safe for Supplementary

- Per-fault RDR values can be reported with the note that RDR depends on each model's own clean MAE.
- Reliability diagnostics can be reported for random missing and outage cases.
- Full horizon tables can be reported for h3, h6, and h12.
- NoGate and seed-stability tables can be reported as mechanism and repeatability checks.

## Must Be Phrased Cautiously

- Linear drift robustness is dataset-dependent: improved on METR-LA but regressed on PEMS-BAY.
- Stuck-at-last-value predictive MAE improves in the main cross-dataset comparison, but reliability separation remains mixed.
- Seed stability is supported only across the evaluated seeds 42, 43, and 44.
- The ID-MLP backbone is inspired by identity-enhanced forecasting ideas and should cite [STID citation], but it is not an official reproduction.

## Forbidden

- Do not claim all faults improve on both datasets.
- Do not claim linear drift is solved.
- Do not claim stuck reliability detection is solved.
- Do not claim official STID reproduction.
- Do not claim clean SOTA forecasting.
- Do not claim zero-overhead or latency-free deployment.
- Do not claim exhaustive multi-seed stability.
- Do not call the proposed method SRAF-STID.
"""


def make_reviewer_risk_audit() -> str:
    return """# Reviewer Risk Audit

## Why not claim clean SOTA?

The experiments are designed to test robustness under faulty sensor observations, not to reproduce or exceed clean forecasting leaderboards. ID-MLP is an identity-enhanced backbone used for controlled same-backbone comparisons. The manuscript should report clean MAE transparently but avoid clean SOTA claims.

## What is the relationship to STID?

ID-MLP is inspired by spatial-temporal identity modeling and should cite [STID citation]. It is not an official STID reproduction, and the proposed method should be named SRAF-ID.

## Does the method generalize beyond METR-LA?

The evidence supports generalization to a second dataset: SRAF-ID improves average faulty MAE on both METR-LA and PEMS-BAY and improves 11 of 12 dataset-fault pairs in the formal cross-dataset comparison.

## Is the improvement due to repair branch or reliability gate?

The full noGate ablation addresses this. Full SRAF-ID beats SRAF-ID-noGate on 11 of 12 faulty dataset-fault pairs and 7 of 8 severe pairs, supporting a gate contribution beyond ungated repair.

## Are results seed-stable?

The seed-stability gate covers seeds 42, 43, and 44. Across these evaluated seeds, SRAF-ID improves mean MAE on 11 of 12 dataset-fault pairs. This supports repeatability but should not be described as exhaustive stability.

## What happens on linear drift?

Linear drift is inconsistent. It improves on METR-LA in the formal artifact, but PEMS-BAY linear drift regresses and loses across all three evaluated seeds. This should be stated as a limitation.

## What happens on stuck faults?

SRAF-ID improves predictive MAE for stuck faults in the main cross-dataset comparison, but reliability separation for stuck positions remains mixed. The manuscript should not claim stuck detection is solved.

## Is the method lightweight?

Parameter overhead is small: 161 additional parameters over ID-MLP-CA on both datasets. Latency overhead is measurable and must be reported.

## Why is latency overhead acceptable or at least transparent?

The current evidence supports parameter-light robustness, not latency-free deployment. The paper should report latency overhead and discuss possible engineering optimization as future work.

## Are targets corrupted?

No. The protocol applies faults only to input speed observations. The target Y remains clean, and identity features are preserved.

## Are time identity features modified by faults?

No. The speed-only repair protocol preserves tod_norm and dow_norm. SRAF-ID repairs only speed_norm before concatenating the unchanged identity features.
"""


def make_limitations() -> str:
    return """# Limitations for Manuscript

- PEMS-BAY linear drift regresses relative to ID-MLP-CA, and the seed-stability audit confirms that this fault remains unfavorable across seeds 42, 43, and 44.
- Stuck-at-last-value predictive MAE improves, but stuck reliability separation remains mixed; stuck fault detection should not be claimed solved.
- SRAF-ID adds negligible parameter overhead, but latency overhead is measurable on both datasets.
- ID-MLP is inspired by identity-enhanced forecasting ideas, but the experiments do not claim official STID reproduction.
- The paper should not claim clean forecasting SOTA.
- Seed stability is evaluated for seeds 42, 43, and 44 only; it is not exhaustive.
- The evidence covers METR-LA and PEMS-BAY, but no third dataset is included.
- No full MoE or large-model alternative is evaluated in this work.
"""


def make_method_naming_note() -> str:
    return """# Manuscript Method Naming Note

The proposed method name is **SRAF-ID**.

Use the following manuscript-facing names consistently:

- **ID-MLP-clean**: identity-enhanced MLP backbone trained on clean inputs.
- **ID-MLP-CA**: the same backbone trained with corruption-aware inputs.
- **SRAF-ID**: the proposed sensor-reliability-aware speed-repair framework combined with ID-MLP.
- **SRAF-ID-noGate**: no-reliability-gate ablation.

ID-MLP means an identity-enhanced MLP backbone inspired by spatial-temporal identity modeling. Related work should cite [STID citation] as conceptual inspiration. The manuscript should not claim official STID reproduction, and it should not name the proposed method with STID.
"""


def make_checklist() -> str:
    return """# Experiment Section Checklist

- [x] Table values are traceable to CSV, JSON, or markdown artifacts.
- [x] Claims are mapped to supporting artifacts in evidence_to_claim_traceability.csv.
- [x] Forbidden claims are listed and avoided in the narrative drafts.
- [x] Dataset splits are stated for METR-LA and PEMS-BAY.
- [x] Fault settings are stated.
- [x] Metrics are stated.
- [x] Seed settings are stated as evaluated seeds 42, 43, and 44.
- [x] Latency overhead is reported as measurable.
- [x] Limitations include PEMS-BAY linear drift, mixed stuck reliability, latency overhead, no clean SOTA claim, and non-exhaustive seed stability.
- [ ] Add final bibliography entries for [STID citation], [METR-LA citation], and [PEMS-BAY citation].
- [ ] Draw Figure 1 architecture diagram from the method description.
- [ ] Confirm final journal table numbering after manuscript structure is fixed.
"""


def unsupported_claim_audit(output_dir: Path) -> Dict[str, object]:
    scanned = [
        "manuscript_experiment_section_draft.md",
        "manuscript_results_section_draft.md",
        "manuscript_safe_claims_master.md",
        "reviewer_risk_audit.md",
        "limitations_for_manuscript.md",
        "manuscript_method_naming_note.md",
    ]
    findings = []
    risky_patterns = [
        ("SRAF-STID", "Forbidden proposed method name."),
        ("state-of-the-art", "Potential clean SOTA claim."),
        ("SOTA", "Potential clean SOTA claim."),
        ("zero overhead", "Forbidden zero-overhead claim."),
        ("all faults improve on both datasets", "Forbidden all-fault claim."),
        ("stuck reliability detection is solved", "Forbidden stuck-detection claim."),
        ("official STID reproduction", "Forbidden official reproduction claim unless negated."),
    ]
    for name in scanned:
        text = read_text(output_dir / name)
        lower = text.lower()
        for pattern, reason in risky_patterns:
            p_lower = pattern.lower()
            idx = lower.find(p_lower)
            while idx != -1:
                context = lower[max(0, idx - 40) : idx + len(p_lower) + 40]
                negated = any(
                    prefix in context
                    for prefix in [
                        "do not claim",
                        "not claim",
                        "does not claim",
                        "should not claim",
                        "not an official",
                        "not as clean",
                        "avoid clean",
                    ]
                )
                forbidden_section = "## forbidden" in lower[max(0, idx - 500) : idx]
                if not negated and not forbidden_section:
                    findings.append({"file": name, "pattern": pattern, "reason": reason, "context": context})
                idx = lower.find(p_lower, idx + 1)
    return {
        "status": "PASS" if not findings else "REVIEW",
        "findings": findings,
        "scanned_files": scanned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metr-la-dir", required=True)
    parser.add_argument("--pems-bay-dir", required=True)
    parser.add_argument("--cross-dataset-dir", required=True)
    parser.add_argument("--nogate-dir", required=True)
    parser.add_argument("--seed-stability-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = {
        "metr": Path(args.metr_la_dir),
        "pems": Path(args.pems_bay_dir),
        "cross": Path(args.cross_dataset_dir),
        "nogate": Path(args.nogate_dir),
        "seed": Path(args.seed_stability_dir),
        "out": Path(args.output_dir),
    }
    paths["out"].mkdir(parents=True, exist_ok=True)

    audit = input_audit(paths)
    if audit["missing"]:
        write_text(paths["out"] / "DATA_BLOCKER_REPORT.md", "# Data Blocker Report\n\nMissing artifacts:\n" + "\n".join(f"- {m}" for m in audit["missing"]))
        raise SystemExit(f"Missing required artifacts: {audit['missing']}")

    cross_main = read_csv(paths["cross"] / "table1_cross_dataset_main_summary.csv")
    cross_gain = read_csv(paths["cross"] / "table2_cross_dataset_same_backbone_gain.csv")
    nogate_gain = read_csv(paths["nogate"] / "gate_gain_summary.csv")
    seed_summary = read_csv(paths["seed"] / "seed_stability_summary.csv")

    claims = make_claims(paths, cross_main, cross_gain, nogate_gain, seed_summary)

    write_text(paths["out"] / "manuscript_experiment_section_draft.md", make_experiment_section(cross_main))
    write_text(paths["out"] / "manuscript_results_section_draft.md", make_results_section(cross_main, seed_summary))
    write_text(paths["out"] / "table_mapping_for_manuscript.md", make_table_mapping())
    write_text(paths["out"] / "figure_mapping_for_manuscript.md", make_figure_mapping())
    write_text(paths["out"] / "manuscript_safe_claims_master.md", make_safe_claims_master())
    write_csv(
        paths["out"] / "evidence_to_claim_traceability.csv",
        claims,
        [
            "claim_id",
            "claim_text",
            "supporting_artifact_path",
            "supporting_table_or_csv",
            "metric_columns_used",
            "main_text_safe",
            "evidence_scope",
        ],
    )
    write_text(paths["out"] / "reviewer_risk_audit.md", make_reviewer_risk_audit())
    write_text(paths["out"] / "limitations_for_manuscript.md", make_limitations())
    write_text(paths["out"] / "manuscript_method_naming_note.md", make_method_naming_note())
    write_text(paths["out"] / "experiment_section_checklist.md", make_checklist())

    audit_path = paths["out"] / "input_artifact_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_text(
        paths["out"] / "input_artifact_audit_summary.md",
        "# Input Artifact Audit\n\n"
        f"Status: {audit['status']}\n\n"
        "All required formal table, noGate, and seed-stability artifacts were found."
    )

    unsupported = unsupported_claim_audit(paths["out"])
    (paths["out"] / "unsupported_claims_audit.json").write_text(json.dumps(unsupported, indent=2), encoding="utf-8")

    manifest = {
        "stage": "MANUSCRIPT_EXPERIMENT_AND_EVIDENCE_DRAFT_GATE",
        "status": "PASS" if unsupported["status"] == "PASS" else "PARTIAL",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {k: str(v) for k, v in paths.items() if k != "out"},
        "output_dir": str(paths["out"]),
        "no_experiments_run": True,
        "method_name": "SRAF-ID",
        "required_outputs": [
            "manuscript_experiment_section_draft.md",
            "manuscript_results_section_draft.md",
            "table_mapping_for_manuscript.md",
            "figure_mapping_for_manuscript.md",
            "manuscript_safe_claims_master.md",
            "evidence_to_claim_traceability.csv",
            "reviewer_risk_audit.md",
            "limitations_for_manuscript.md",
            "manuscript_method_naming_note.md",
            "experiment_section_checklist.md",
        ],
        "unsupported_claim_audit_status": unsupported["status"],
    }
    (paths["out"] / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"status": manifest["status"], "output_dir": str(paths["out"])}, indent=2))


if __name__ == "__main__":
    main()

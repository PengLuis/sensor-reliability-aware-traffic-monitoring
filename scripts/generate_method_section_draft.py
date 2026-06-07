"""Generate SRAF-ID method-section draft artifacts from code/evidence context.

The script writes manuscript-facing method text, formulas, pseudocode, figure
planning notes, and conservative audits. It does not run experiments or modify
models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


REQUIRED_CODE_ARTIFACTS = [
    Path("src/models/strong_backbones.py"),
    Path("src/models/residual_models.py"),
    Path("scripts/run_metr_la_sraf_stid_same_backbone_gain.py"),
    Path("scripts/run_pems_bay_sraf_id_transfer.py"),
    Path("scripts/run_full_no_reliability_gate_ablation.py"),
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def artifact_audit(paths: Dict[str, Path]) -> Dict[str, object]:
    required = {
        "evidence_run_manifest": paths["evidence"] / "run_manifest.json",
        "evidence_claim_traceability": paths["evidence"] / "evidence_to_claim_traceability.csv",
        "cross_dataset_summary": paths["cross"] / "cross_dataset_evidence_alignment_summary.md",
        "nogate_manifest": paths["nogate"] / "run_manifest.json",
        "seed_manifest": paths["seed"] / "run_manifest.json",
    }
    for code_path in REQUIRED_CODE_ARTIFACTS:
        required[f"code::{code_path.as_posix()}"] = code_path
    missing = [name for name, path in required.items() if not path.exists()]
    return {
        "stage": "METHOD_SECTION_AND_ARCHITECTURE_FIGURE_GATE",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "required_artifacts": {name: str(path) for name, path in required.items()},
        "missing": missing,
        "status": "PASS" if not missing else "FAIL",
    }


def method_section_text() -> str:
    return r"""# Method Section Draft

## Problem Formulation

We study robust full-network traffic sensor forecasting under missing and faulty observations. For each forecasting window, the model receives a historical speed sequence \(X \in R^{L \times N \times 1}\), where \(L\) is the input length and \(N\) is the number of sensors. The target is a clean multi-horizon speed sequence \(Y \in R^{H \times N}\), where \(H\) is the prediction horizon. In our experiments, \(L=12\) and \(H=12\), corresponding to one hour of history and one hour of future predictions for five-minute traffic data.

The input used by the forecasting backbone is identity-augmented. For each time step and sensor, the normalized speed channel is concatenated with time-of-day and day-of-week identity channels. We denote this augmented input as \([X^{speed}, E^{tod}, E^{dow}]\). Faults are applied only to the speed channel, while the temporal identity channels are preserved. The target \(Y\) is never corrupted.

## Overview of SRAF-ID

SRAF-ID is a lightweight sensor-reliability-aware forecasting framework. It combines three components: a speed-channel reliability estimator, a temporal/spatial speed repair module, and an identity-enhanced forecasting backbone named ID-MLP. The central design is to repair potentially unreliable speed observations before forecasting while leaving the temporal identity features unchanged.

Given a possibly corrupted speed history \(X^c\), SRAF-ID estimates a reliability score \(R_{t,i}\in[0,1]\) for each time step \(t\) and sensor \(i\). A high reliability score means the observed speed should be trusted; a low score means the repair estimate should receive more weight. The repaired speed is concatenated with the original time-of-day and day-of-week identity features, and the resulting sequence is passed to the ID-MLP backbone to generate all future horizons.

## Identity-Enhanced Forecasting Backbone

ID-MLP is the identity-enhanced forecasting backbone used for clean, corruption-aware, and SRAF-ID models. It is inspired by spatial-temporal identity modeling, and related work should cite [STID citation]. It should not be described as an official STID reproduction.

The implemented backbone uses a Conv2d time-series embedding over the input window, node identity embeddings, time-in-day embeddings, day-in-week embeddings, residual 1x1 Conv2d MLP blocks, and a Conv2d regression head that outputs all horizons. The backbone ignores the traffic adjacency matrix; adjacency is used only inside the SRAF repair module. In the manuscript, the backbone should be named ID-MLP rather than STID.

## Sensor Reliability Estimation

The reliability estimator operates only on the speed channel. In the implemented SRAF-ID wrapper, the repairer receives \(X^{speed}\) with shape \([B,L,N,1]\), not the time identity channels. Reliability features include the filled speed signal, temporal change information, and the finite observed mask. The reliability head is a small MLP with a sigmoid output. In full SRAF-ID, missing observations are also handled by a hard missing gate so that missing speed positions receive near-zero reliability.

During corruption-aware training, the simulated corruption mask \(M\) is defined as \(M_{t,i}=1\) for corrupted positions and \(M_{t,i}=0\) for clean positions. Reliability supervision uses target \(1-M\), so corrupted positions are encouraged to have low reliability and clean positions high reliability.

## Speed-Only Temporal/Spatial Repair

SRAF-ID repairs only normalized speed observations. The temporal repair candidate carries forward the most recent observed speed within the input window. The spatial repair candidate uses the physical sensor adjacency matrix to compute a weighted neighbor average. These candidates are blended into a repair estimate. In the current selected SRAF-ID configuration, the temporal/spatial blend uses the fixed lightweight repair path implemented in the SRAF repair module rather than a graph-attention or Transformer mechanism.

The time-of-day and day-of-week identity channels bypass the repair module. The wrapper concatenates the repaired speed with the unchanged identity channels before invoking ID-MLP. Runtime checks in the wrapper raise an error if the identity features are modified.

## Reliability-Aware Fusion

The final repaired speed is computed by reliability-aware fusion between the filled corrupted speed and the repair estimate. When reliability is high, the model keeps more of the observed speed. When reliability is low, it relies more on the repair estimate. The no-gate ablation, SRAF-ID-noGate, disables reliability-gated fusion and uses fixed neutral repair fusion. The full noGate ablation shows that the reliability-aware gate contributes beyond simply adding a repair branch.

## Multi-Horizon Forecasting

After speed repair, SRAF-ID passes \([X^r, E^{tod}, E^{dow}]\) to ID-MLP. The backbone embeds the full input window, concatenates node and temporal identity embeddings, applies residual lightweight Conv2d MLP blocks, and uses a regression head to produce \(\hat{Y}_{1:H}\) for all horizons simultaneously. Horizon-wise metrics are reported at h3, h6, and h12.

## Training Objectives

SRAF-ID is trained with corruption-aware input batches. For each batch, a fault sampler corrupts only the speed channel, preserving time identity features and leaving the target clean. The training objective combines forecast loss, repair consistency loss, and reliability supervision:

\[
L = L_{forecast} + \lambda_{repair} L_{repair} + \lambda_{rel} L_{rel}.
\]

The forecast loss is MAE between \(\hat{Y}\) and clean \(Y\). The repair consistency loss measures speed repair error on corrupted input positions where the clean input speed is known. The reliability loss is MSE between predicted reliability and \(1-M\). In the full-confirmation experiments, \(\lambda_{repair}=0.05\) and \(\lambda_{rel}=0.01\).

## Inference Procedure

At inference time, SRAF-ID receives a speed history that may contain missing or faulty observations. The model constructs or preserves the identity channels, estimates speed reliability, computes temporal and spatial repair candidates, fuses observed and repaired speed, concatenates the repaired speed with the unchanged identity features, and forecasts all horizons. No target information is used during inference.

## Complexity Discussion

SRAF-ID is designed to be lightweight in parameters. The cross-dataset formal artifacts report 161 additional parameters over ID-MLP-CA on both METR-LA and PEMS-BAY. Latency overhead is measurable, however, and should be reported as a deployment cost rather than hidden. The defensible claim is small parameter overhead with transparent latency overhead, not zero-overhead inference.

## Relationship to STID-Inspired Identity Modeling

The ID-MLP backbone is inspired by identity-enhanced spatial-temporal forecasting ideas and should cite [STID citation]. The manuscript should not claim parity with an official STID implementation, should not use STID in the proposed method name, and should not claim clean forecasting SOTA. The contribution is the SRAF-ID reliability-aware speed repair framework evaluated through same-backbone comparisons with ID-MLP-CA.
"""


def formulas_text() -> str:
    return r"""# Method Formulas

## A. Input and Prediction Task

For a batch item, let:

\[
X \in R^{L \times N \times 1}, \quad Y \in R^{H \times N \times 1}.
\]

The identity-augmented input is:

\[
X^{id} = [X^{speed}, E^{tod}, E^{dow}] \in R^{L \times N \times 3},
\]

where \(E^{tod}\) and \(E^{dow}\) are normalized time-of-day and day-of-week identity channels. The ID-MLP forecasting backbone predicts:

\[
\hat{Y}_{1:H}=f_{\theta}(X^{id}).
\]

## B. Corrupted Input

During corruption-aware training and faulty evaluation, a fault operator \(C\) corrupts only the speed channel:

\[
X^{c,speed}=C(X^{speed};\omega),
\]

while:

\[
E^{tod,c}=E^{tod}, \quad E^{dow,c}=E^{dow}.
\]

The target remains clean:

\[
Y^c = Y.
\]

Define the simulated fault mask:

\[
M_{t,i}=1 \text{ if position } (t,i) \text{ is corrupted}, \quad M_{t,i}=0 \text{ otherwise}.
\]

The finite observed mask used by the repairer is separate:

\[
O_{t,i}=1 \text{ if } X^{c,speed}_{t,i} \text{ is finite/observed}, \quad O_{t,i}=0 \text{ otherwise}.
\]

## C. Reliability Estimation

SRAF-ID estimates reliability from speed-only features:

\[
R = g_{\phi}(X^{c,speed}_{filled}, \Delta X^{c,speed}, O),
\]

where \(R_{t,i}\in[0,1]\). Lower \(R_{t,i}\) means the observation is less reliable. The selected wrapper uses a hard missing gate:

\[
R_{t,i} \leftarrow R_{t,i} O_{t,i},
\]

so missing speed observations receive low reliability.

## D. Repair Candidates

The temporal repair candidate carries forward the most recent observed value:

\[
X^{temp}_{t,i} =
\begin{cases}
X^{c,speed}_{t,i}, & O_{t,i}=1,\\
X^{temp}_{t-1,i}, & O_{t,i}=0.
\end{cases}
\]

For the first step, the implementation initializes from the filled first input value. The spatial repair candidate uses the physical adjacency matrix \(A\):

\[
X^{sp}_{t,i} = \frac{\sum_j A_{i,j} X^{c,speed}_{filled,t,j}}{\max(\sum_j A_{i,j}, \epsilon)}.
\]

The repair blend is:

\[
\bar{X}^{speed}_{t,i} = \alpha X^{temp}_{t,i} + (1-\alpha)X^{sp}_{t,i}.
\]

For the selected SRAF-ID configuration, this is the lightweight temporal/spatial repair path in the implemented SRAF repairer.

## E. Reliability-Aware Fusion

The final repaired speed is:

\[
X^{r}_{t,i} = R_{t,i} X^{c,speed}_{filled,t,i} + (1-R_{t,i})\bar{X}^{speed}_{t,i}.
\]

SRAF-ID-noGate replaces \(R_{t,i}\) with a fixed neutral value and does not optimize reliability loss:

\[
X^{r,noGate}_{t,i} = 0.5 X^{c,speed}_{filled,t,i} + 0.5\bar{X}^{speed}_{t,i}.
\]

## F. Backbone Prediction

The repaired speed is concatenated with unchanged identity channels:

\[
X^{SRAF-ID} = [X^r, E^{tod}, E^{dow}].
\]

The final forecast is:

\[
\hat{Y}_{1:H}=f_{\theta}(X^{SRAF-ID}).
\]

## G. Training Loss

Forecast loss:

\[
L_{forecast} = \frac{1}{H N}\sum_{h=1}^{H}\sum_{i=1}^{N} |\hat{Y}_{h,i}-Y_{h,i}|.
\]

Repair consistency loss on corrupted input positions:

\[
L_{repair} =
\frac{\sum_{t,i} M_{t,i}|X^r_{t,i}-X^{speed}_{t,i}|}
{\max(\sum_{t,i}M_{t,i},1)}.
\]

Reliability supervision:

\[
L_{rel} = \frac{1}{LN}\sum_{t,i}(R_{t,i}-(1-M_{t,i}))^2.
\]

Total objective:

\[
L = L_{forecast} + \lambda_{repair}L_{repair} + \lambda_{rel}L_{rel}.
\]

In the validated full-confirmation runs, \(\lambda_{repair}=0.05\) and \(\lambda_{rel}=0.01\). These formulas match the implemented SRAF-ID training objective in the full-confirmation scripts.
"""


def pseudocode_text() -> str:
    return """# Algorithm Pseudocode

## Algorithm 1: Training SRAF-ID

**Inputs:** clean training windows `(X, Y)`, adjacency matrix `A`, fault sampler `C`, identity features `(tod, dow)`, loss weights `lambda_repair` and `lambda_rel`.

1. Initialize ID-MLP backbone parameters and SRAF speed-repair parameters.
2. For each training epoch:
3. Sample a mini-batch of clean windows `(X, Y)`.
4. Construct identity-augmented input `[X_speed, E_tod, E_dow]`.
5. Sample a fault setting from the corruption-aware training schedule.
6. Corrupt only the speed channel: `X_speed_c = C(X_speed)`.
7. Preserve identity channels: `E_tod_c = E_tod`, `E_dow_c = E_dow`.
8. Build the simulated corruption mask `M`, where `M=1` means corrupted.
9. Build the finite observed mask `O` from the corrupted speed channel.
10. Estimate reliability `R = g_phi(X_speed_c, O)`.
11. Compute temporal speed repair candidate.
12. Compute spatial speed repair candidate using adjacency `A`.
13. Fuse observed and repaired speed using reliability-aware fusion.
14. Concatenate repaired speed with unchanged identity channels.
15. Forecast all horizons with ID-MLP.
16. Compute `L_forecast = MAE(Y_hat, Y)`.
17. Compute `L_repair` on corrupted speed positions using clean input speed.
18. Compute `L_rel = MSE(R, 1 - M)`.
19. Optimize `L = L_forecast + lambda_repair L_repair + lambda_rel L_rel`.
20. Select the best checkpoint using the clean/fault validation criterion used in the scripts.

## Algorithm 2: Inference with SRAF-ID

**Inputs:** possibly faulty speed history, adjacency matrix `A`, time identity features.

1. Receive the latest speed history.
2. Construct or preserve time-of-day and day-of-week identity features.
3. Replace non-finite speed values with filled values for neural computation.
4. Estimate speed reliability.
5. Compute temporal and spatial repair candidates.
6. Fuse observed speed and repair estimate using reliability.
7. Concatenate repaired speed with unchanged identity channels.
8. Run ID-MLP to forecast horizons `1...H`.
9. Return multi-horizon predictions, including h3, h6, and h12 evaluation outputs.
"""


def figure_plan_text() -> str:
    return """# Architecture Figure Plan

Figure title: **SRAF-ID architecture for reliability-aware traffic sensor forecasting**

## Panel A: Input Construction

- Visual elements: a tensor block labeled `Speed history X_speed [L x N]`, two thin aligned strips labeled `time-of-day identity` and `day-of-week identity`, and a sensor identity lookup block.
- Arrows: speed goes toward fault/repair path; tod/dow bypass the repair path toward the backbone.
- Labels: `input window L=12`, `N sensors`, `identity channels preserved`.
- Formula to display: `X_id = [X_speed, E_tod, E_dow]`.
- Color suggestions: speed in blue, temporal identities in green, sensor identity in gray.
- Do not include: any label naming the method as STID.

## Panel B: Fault/Corruption Scenario

- Visual elements: small icons/labels for missing, outage, noise, drift, and stuck faults attached only to the speed stream.
- Arrows: corrupted speed stream flows into SRAF repair.
- Labels: `faults apply to speed only`, `target Y remains clean`.
- Formula to display: `X_c = C(X_speed; omega)`.
- Color suggestions: faulty speed overlay in orange/red.
- Do not include: corruption arrows pointing to tod/dow or target Y.

## Panel C: Reliability Estimation and Repair

- Visual elements: reliability estimator box, temporal repair box, spatial repair box with adjacency matrix `A`, and a reliability gate.
- Arrows: corrupted speed and mask feed reliability estimator; speed feeds temporal repair; speed plus adjacency feeds spatial repair; repair candidates feed fusion gate.
- Labels: `R in [0,1]`, `lower R = less reliable`, `temporal carry-forward`, `adjacency neighbor repair`.
- Formulas to display: `R = g_phi(X_c, O)`, `X_sp = sum_j A_ij X_j / sum_j A_ij`, `X_r = R X_c + (1-R) X_bar`.
- Color suggestions: reliability in purple, temporal repair in teal, spatial repair in amber, gate in dark gray.
- Do not include: graph attention, Transformer, or MoE blocks.

## Panel D: Identity-Enhanced Forecasting Backbone

- Visual elements: concatenation block `[X_r, E_tod, E_dow]`, sequence embedding, sensor identity embedding, temporal identity embedding, residual lightweight blocks, regression head.
- Arrows: repaired speed plus identities flow left-to-right through ID-MLP.
- Labels: `ID-MLP backbone`, `identity-enhanced forecasting backbone`, `multi-horizon regression`.
- Formula to display: `Y_hat = f_theta([X_r, E_tod, E_dow])`.
- Color suggestions: backbone modules in neutral blue/gray.
- Do not include: the term `STID` as a module label.

## Panel E: Training Losses

- Visual elements: three small loss boxes connected to a summed objective.
- Arrows: predictions to forecast loss; repaired speed to repair consistency loss; reliability scores to reliability supervision.
- Labels: `forecast MAE`, `repair consistency`, `reliability supervision`.
- Formula to display: `L = L_forecast + lambda_repair L_repair + lambda_rel L_rel`.
- Color suggestions: loss boxes in pale yellow.
- Do not include: losses not implemented in the full-confirmation SRAF-ID runs.

## Panel F: Output

- Visual elements: prediction tensor or three horizon markers labeled `h3`, `h6`, `h12`.
- Arrows: regression head to robust multi-horizon forecast.
- Labels: `robust prediction under faulty observations`.
- Formula to display: `Y_hat_{1:H}`.
- Color suggestions: output in clean blue/green.
- Do not include: claims of all-fault improvement or clean SOTA.
"""


def svg_spec_text() -> str:
    return """# Figure 1 SVG Prompt or Specification

Create a clean academic SVG figure in a horizontal left-to-right layout titled:

**SRAF-ID architecture for reliability-aware traffic sensor forecasting**

Canvas:
- Size: 1800 x 850 px.
- Background: white.
- Font: clean sans-serif such as Arial or Helvetica.
- Use consistent rounded rectangles with modest radius, thin strokes, and high contrast.

Layout:
1. Left column: `Input history`
   - Blue block: `Speed history X_speed`
   - Green strips: `time-of-day identity` and `day-of-week identity`
   - Gray chip: `sensor identity`
   - Add note: `tod/dow preserved`

2. Upper middle: `Faulty speed observations`
   - Orange overlay on the speed stream only.
   - Small tags: `missing`, `outage`, `noise`, `drift`, `stuck`.
   - Add note: `target Y is clean`.

3. Middle: `SRAF speed repair`
   - Box 1: `Reliability estimator g_phi`
   - Box 2: `Temporal repair`
   - Box 3: `Spatial repair with adjacency A`
   - Diamond/gate: `Reliability-aware fusion`
   - Display equation near gate: `X_r = R X_c + (1-R) X_bar`.
   - Show tod/dow bypass line around this module.

4. Right middle: `ID-MLP backbone`
   - Concatenation node: `[X_r, E_tod, E_dow]`
   - Boxes: `sequence embedding`, `identity embeddings`, `residual 1x1 MLP blocks`, `multi-horizon regression`.
   - Label as `Identity-enhanced forecasting backbone`.

5. Right column: `Forecast output`
   - Output block: `Y_hat_{1:H}`
   - Three markers: `h3`, `h6`, `h12`.

6. Bottom strip: `Training losses`
   - Three small boxes: `forecast MAE`, `repair consistency`, `reliability supervision`.
   - Summation equation: `L = L_forecast + lambda_repair L_repair + lambda_rel L_rel`.

Legend:
- Blue: speed signal.
- Green: temporal identity features.
- Purple: reliability estimation.
- Amber: repair candidates.
- Gray: forecasting backbone.
- Red/orange: simulated input faults.

What to avoid:
- Do not call the model SRAF-STID.
- Do not label the backbone as STID.
- Do not show tod/dow flowing through the repair module.
- Do not show target Y being corrupted.
- Do not include Transformer, graph attention, MoE, or any large-model block.
- Do not include claims such as clean SOTA, zero overhead, or all faults improved.
"""


def claim_traceability_text() -> str:
    rows = [
        ("SRAF-ID repairs speed only.", "src/models/strong_backbones.py::SRAFOfficialStyleSTIDWrapper; scripts/run_pems_bay_sraf_id_transfer.py", "directly implemented", "low"),
        ("tod/dow identity features bypass SRAF repair and are preserved.", "src/models/strong_backbones.py::SRAFOfficialStyleSTIDWrapper; PEMS/METR run manifests", "directly implemented and experiment-audited", "low"),
        ("ID-MLP uses sequence embedding, node identity, time-in-day, day-in-week embeddings, residual lightweight blocks, and regression head.", "src/models/strong_backbones.py::OfficialStyleSTID", "directly implemented", "low"),
        ("ID-MLP is inspired by identity modeling but not official STID reproduction.", "experiments/manuscript-evidence-draft/manuscript_method_naming_note.md", "manuscript positioning", "medium"),
        ("Reliability scores are in [0,1] and lower values indicate less reliable speed observations.", "src/models/residual_models.py::SRAFResidualGRU.repair_components", "directly implemented", "low"),
        ("Hard missing gate reduces reliability for missing speed observations.", "src/models/strong_backbones.py::SRAFOfficialStyleSTIDWrapper passes hard_missing_gate=True", "directly implemented", "low"),
        ("Temporal repair carries forward the most recent observed value.", "src/models/residual_models.py::basic_temporal_repair", "directly implemented", "low"),
        ("Spatial repair uses adjacency-weighted neighbor averaging.", "src/models/residual_models.py::spatial_repair", "directly implemented", "low"),
        ("Reliability-aware fusion combines filled corrupted speed and repair estimate.", "src/models/residual_models.py::SRAFResidualGRU.repair_components", "directly implemented", "low"),
        ("Training uses forecast, repair consistency, and reliability supervision losses.", "scripts/run_metr_la_sraf_stid_same_backbone_gain.py::train_sraf_stid; scripts/run_pems_bay_sraf_id_transfer.py", "directly implemented", "low"),
        ("Mask M is 1 for corrupted positions and reliability target is 1-M.", "scripts/run_metr_la_sraf_stid_same_backbone_gain.py::train_sraf_stid", "directly implemented", "low"),
        ("SRAF-ID-noGate disables reliability-gated fusion and uses fixed neutral repair fusion.", "scripts/run_full_no_reliability_gate_ablation.py; src/models/residual_models.py", "directly implemented", "low"),
        ("NoGate ablation supports reliability gate contribution.", "experiments/full-no-reliability-gate-ablation/gate_gain_summary.csv", "experiment-backed", "low"),
        ("Latency overhead is measurable.", "experiments/cross-dataset-sraf-id-summary/table7_cross_dataset_complexity_latency.csv", "experiment-backed", "low"),
    ]
    out = ["# Method Claim Traceability", "", "| Statement | Supporting artifact | Status | Risk |", "|---|---|---|---|"]
    out.extend(f"| {s} | `{a}` | {status} | {risk} |" for s, a, status, risk in rows)
    return "\n".join(out)


def reviewer_risk_text() -> str:
    return """# Method Reviewer Risk Audit

## Is this just STID with a new name?

No. The manuscript-facing backbone is ID-MLP, an identity-enhanced MLP-style forecasting backbone inspired by spatial-temporal identity modeling. The proposed method is SRAF-ID, which adds speed-only reliability estimation and repair before the same backbone. The paper should cite [STID citation] as inspiration but should not claim official reproduction.

## What exactly is new compared with ID-MLP-CA?

ID-MLP-CA uses corruption-aware training but directly forecasts from corrupted speed plus identity features. SRAF-ID adds reliability estimation, temporal/spatial speed repair, and reliability-aware fusion before the same ID-MLP backbone.

## Why repair speed only?

The faults represent sensor observation failures in the measured speed channel. Time-of-day and day-of-week identities are metadata features, not sensor measurements, so corrupting or repairing them would create an unfair and unrealistic protocol.

## How do you avoid target leakage?

Faults are applied only to the input speed history. The target sequence remains clean and is used only for supervised forecast loss. Repair consistency uses clean input speed at corrupted input positions during training, not future target values.

## Are time identities corrupted?

No. The scripts audit that speed-only corruption preserves tod/dow features, and the SRAF-ID wrapper concatenates repaired speed with unchanged identity features.

## Does reliability estimation actually help?

The full noGate ablation supports this: full SRAF-ID outperforms SRAF-ID-noGate on most faulty dataset-fault pairs and severe pairs. The claim should be bounded to the evaluated protocol.

## What does noGate prove?

It tests whether an ungated repair branch is enough. Since full SRAF-ID usually outperforms noGate, the evidence supports the reliability-aware gate as a useful component beyond fixed repair fusion.

## Why is stuck reliability mixed?

The reliability signal separates random missing and outage more clearly than stuck faults. Stuck observations can remain finite and locally plausible, making them harder to distinguish using the current lightweight features. The paper should report this as an unresolved limitation.

## Why does PEMS-BAY linear drift regress?

The current evidence shows dataset-dependent drift behavior: METR-LA mean favors SRAF-ID, but PEMS-BAY linear drift regresses across evaluated seeds. The method section should not explain this as solved; the results section should present it as a limitation and possible target for future drift-specific reliability features.

## Is parameter overhead small but latency overhead large?

Parameter overhead is small, but latency overhead is measurable. This is consistent with a lightweight parameter design that still performs extra repair computation at inference time. The paper should report both.

## Why no MoE?

The paper targets a lightweight, explainable sensor-reliability framework. MoE would introduce a different research direction and additional complexity beyond the validated SRAF-ID evidence.

## Why no third dataset?

The current scope validates METR-LA and PEMS-BAY with full formal tables, noGate ablation, and seed stability. A third dataset would strengthen external validity but is outside the current completed evidence.
"""


def checklist_text() -> str:
    return """# Method Section Checklist

- [x] Method name is SRAF-ID only.
- [x] No official STID reproduction claim is made.
- [x] Formulas are consistent with inspected code.
- [x] Mask definition is explicit: `M=1` means corrupted.
- [x] Target Y remains clean.
- [x] Speed-only repair is stated.
- [x] tod/dow preservation is stated.
- [x] Identity backbone wording is cautious and uses ID-MLP.
- [x] Latency overhead is acknowledged.
- [x] Limitations align with the experiment section.
- [ ] Add final citation entry for [STID citation].
- [ ] Convert Figure 1 plan into final SVG or publication figure.
"""


def unsupported_method_audit(out_dir: Path) -> Dict[str, object]:
    files = [
        "method_section_draft.md",
        "method_formulas.md",
        "algorithm_pseudocode.md",
        "architecture_figure_plan.md",
        "figure1_svg_prompt_or_spec.md",
        "method_reviewer_risk_audit.md",
        "method_section_checklist.md",
    ]
    patterns = [
        ("SRAF-STID", "Forbidden method name"),
        ("official STID reproduction", "Forbidden official reproduction claim"),
        ("clean SOTA", "Forbidden clean SOTA claim"),
        ("all-fault improvement", "Forbidden all-fault improvement claim"),
        ("all faults improve", "Forbidden all-fault improvement claim"),
        ("stuck reliability detection is solved", "Forbidden solved stuck claim"),
        ("zero overhead", "Forbidden zero-overhead claim"),
        ("target Y is corrupted", "Forbidden target-corruption implication"),
    ]
    findings = []
    negations = [
        "do not",
        "not claim",
        "should not",
        "not be",
        "no ",
        "avoid",
        "never",
        "without",
    ]
    for file_name in files:
        text = read_text(out_dir / file_name)
        lower = text.lower()
        for pattern, reason in patterns:
            p = pattern.lower()
            idx = lower.find(p)
            while idx != -1:
                context = lower[max(0, idx - 80) : idx + len(p) + 80]
                negated = any(n in context for n in negations)
                avoid_list = "what to avoid" in lower[max(0, idx - 500) : idx] or "forbidden" in lower[max(0, idx - 500) : idx]
                if not negated and not avoid_list:
                    findings.append({"file": file_name, "pattern": pattern, "reason": reason, "context": context})
                idx = lower.find(p, idx + 1)
    return {"status": "PASS" if not findings else "FAIL", "findings": findings, "scanned_files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--cross-dataset-dir", required=True)
    parser.add_argument("--nogate-dir", required=True)
    parser.add_argument("--seed-stability-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = {
        "evidence": Path(args.evidence_dir),
        "cross": Path(args.cross_dataset_dir),
        "nogate": Path(args.nogate_dir),
        "seed": Path(args.seed_stability_dir),
        "out": Path(args.output_dir),
    }
    paths["out"].mkdir(parents=True, exist_ok=True)

    audit = artifact_audit(paths)
    (paths["out"] / "input_artifact_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if audit["missing"]:
        write_text(paths["out"] / "BLOCKER_REPORT.md", "# Blocker Report\n\nMissing required artifacts:\n" + "\n".join(f"- {m}" for m in audit["missing"]))
        raise SystemExit(f"Missing required artifacts: {audit['missing']}")

    write_text(paths["out"] / "method_section_draft.md", method_section_text())
    write_text(paths["out"] / "method_formulas.md", formulas_text())
    write_text(paths["out"] / "algorithm_pseudocode.md", pseudocode_text())
    write_text(paths["out"] / "architecture_figure_plan.md", figure_plan_text())
    write_text(paths["out"] / "figure1_svg_prompt_or_spec.md", svg_spec_text())
    write_text(paths["out"] / "method_claim_traceability.md", claim_traceability_text())
    write_text(paths["out"] / "method_reviewer_risk_audit.md", reviewer_risk_text())
    write_text(paths["out"] / "method_section_checklist.md", checklist_text())

    unsupported = unsupported_method_audit(paths["out"])
    (paths["out"] / "unsupported_method_claims_audit.json").write_text(json.dumps(unsupported, indent=2), encoding="utf-8")

    manifest = {
        "stage": "METHOD_SECTION_AND_ARCHITECTURE_FIGURE_GATE",
        "status": "PASS" if unsupported["status"] == "PASS" else "FAIL",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {k: str(v) for k, v in paths.items() if k != "out"},
        "output_dir": str(paths["out"]),
        "no_experiments_run": True,
        "algorithm_changes": False,
        "method_name": "SRAF-ID",
        "unsupported_method_claims_audit_status": unsupported["status"],
        "required_outputs": [
            "method_section_draft.md",
            "method_formulas.md",
            "algorithm_pseudocode.md",
            "architecture_figure_plan.md",
            "figure1_svg_prompt_or_spec.md",
            "method_claim_traceability.md",
            "method_reviewer_risk_audit.md",
            "method_section_checklist.md",
            "unsupported_method_claims_audit.json",
        ],
    }
    (paths["out"] / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_dir": str(paths["out"])}, indent=2))


if __name__ == "__main__":
    main()

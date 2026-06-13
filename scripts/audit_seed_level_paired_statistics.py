"""Recompute seed-level paired statistics for SRAF-ID vs ID-MLP-CA.

Outputs are built only from saved per-run metric CSV files. The script avoids
SciPy dependency so it can run in the bundled Codex Python runtime.
"""

from __future__ import annotations

import csv
import math
from itertools import product
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_PACKAGE = ROOT / "experiments" / "sraf_id_final_figure_table_package"
FORMAL_BASELINE = ROOT / "experiments" / "id_mlp_ca_matched_fault_distribution_10seed"
OUT_CSV = ROOT / "seed_level_paired_statistics.csv"
OUT_MD = ROOT / "seed_level_paired_statistics_summary.md"

DATASETS = ["METR-LA", "PEMS-BAY"]
SEEDS = list(range(42, 52))
FAULTS = [
    "random_missing_20",
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
]


def dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def read_one_metric_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def t_pdf(x: float, df: int) -> float:
    return (
        math.gamma((df + 1.0) / 2.0)
        / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))
        * (1.0 + x * x / df) ** (-(df + 1.0) / 2.0)
    )


def simpson_integral(fn, a: float, b: float, n: int = 20000) -> float:
    if b <= a:
        return 0.0
    if n % 2:
        n += 1
    h = (b - a) / n
    s = fn(a) + fn(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * fn(a + i * h)
    return s * h / 3.0


def paired_t_pvalue(t_stat: float, df: int) -> float:
    if not math.isfinite(t_stat):
        return 0.0
    area = simpson_integral(lambda x: t_pdf(x, df), 0.0, abs(t_stat))
    return max(0.0, min(1.0, 1.0 - 2.0 * area))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and abs(values[order[j]] - values[order[i]]) < 1.0e-12:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def wilcoxon_exact_pvalue(diffs: list[float]) -> tuple[float, float]:
    nonzero = [d for d in diffs if abs(d) > 1.0e-12]
    if not nonzero:
        return 0.0, 1.0
    ranks = average_ranks([abs(d) for d in nonzero])
    total = sum(ranks)
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    observed = min(w_plus, total - w_plus)
    extreme = 0
    for signs in product([0, 1], repeat=len(ranks)):
        wp = sum(r for r, sign in zip(ranks, signs) if sign)
        if min(wp, total - wp) <= observed + 1.0e-12:
            extreme += 1
    return observed, extreme / (2 ** len(ranks))


def main() -> None:
    sraf = pd.read_csv(FINAL_PACKAGE / "sraf_id_softmax_formal_per_seed_metrics.csv")
    baseline = pd.read_csv(FORMAL_BASELINE / "aggregate" / "formal_10seed_per_seed_metrics.csv")
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for fault in FAULTS:
            pairs: list[tuple[int, float, float, float]] = []
            for seed in SEEDS:
                baseline_row = baseline[
                    (baseline["dataset"] == dataset)
                    & (baseline["seed"] == seed)
                    & (baseline["fault"] == fault)
                ]
                if len(baseline_row) != 1:
                    raise ValueError(f"Missing matched ID-MLP-CA row: {dataset} {fault} seed={seed}")
                b = float(baseline_row.iloc[0]["mae"])
                subset = sraf[(sraf["dataset"] == dataset) & (sraf["seed"] == seed) & (sraf["fault"] == fault)]
                if len(subset) != 1:
                    raise ValueError(f"Missing SRAF-ID row: {dataset} {fault} seed={seed}")
                sv = float(subset.iloc[0]["mae"])
                pairs.append((seed, b, sv, b - sv))
            deltas = [p[3] for p in pairs]
            n = len(deltas)
            mean_delta = sum(deltas) / n
            sd_delta = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / (n - 1))
            se = sd_delta / math.sqrt(n)
            t_stat = mean_delta / se if se > 0 else float("inf")
            p_t = paired_t_pvalue(t_stat, n - 1)
            w_stat, p_w = wilcoxon_exact_pvalue(deltas)
            tcrit_95_df9 = 2.2621571627409915
            rows.append(
                {
                    "dataset": dataset,
                    "fault": fault,
                    "n": n,
                    "id_mlp_ca_mae_mean": sum(p[1] for p in pairs) / n,
                    "sraf_id_mae_mean": sum(p[2] for p in pairs) / n,
                    "mean_delta_id_minus_sraf": mean_delta,
                    "sd_delta": sd_delta,
                    "ci95_low": mean_delta - tcrit_95_df9 * se,
                    "ci95_high": mean_delta + tcrit_95_df9 * se,
                    "positive_seed_count": sum(1 for d in deltas if d > 0),
                    "paired_t_statistic": t_stat,
                    "paired_t_p_value": p_t,
                    "wilcoxon_signed_rank_statistic": w_stat,
                    "wilcoxon_exact_two_sided_p_value": p_w,
                    "raw_seed_deltas": ";".join(f"{seed}:{delta:.9f}" for seed, _, _, delta in pairs),
                    "id_mlp_ca_source": str(FORMAL_BASELINE / "aggregate" / "formal_10seed_per_seed_metrics.csv"),
                    "sraf_id_source": str(FINAL_PACKAGE / "sraf_id_softmax_formal_per_seed_metrics.csv"),
                    "interpretation": (
                        "10/10 seed wins"
                        if all(d > 0 for d in deltas)
                        else "0/10 seed wins; SRAF-ID worse in all seeds"
                        if all(d < 0 for d in deltas)
                        else "positive aggregate delta with mixed seed-level behavior"
                        if mean_delta > 0
                        else "negative aggregate delta with mixed seed-level behavior"
                    ),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    lines = [
        "# Seed-Level Paired Statistics Summary",
        "",
        "Delta is ID-MLP-CA MAE minus SRAF-ID MAE; positive values favor SRAF-ID.",
        "All values are recomputed from saved per-run metrics by `scripts/audit_seed_level_paired_statistics.py`.",
        "",
        "| Dataset | Fault | n | Mean delta | 95% CI | Positive seeds | paired t p | Wilcoxon p | Interpretation |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in df.itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {r.fault} | {r.n} | {r.mean_delta_id_minus_sraf:.6f} | "
            f"[{r.ci95_low:.6f}, {r.ci95_high:.6f}] | {int(r.positive_seed_count)}/10 | "
            f"{r.paired_t_p_value:.6g} | {r.wilcoxon_exact_two_sided_p_value:.6g} | {r.interpretation} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Paired t-test p-values are computed by numerical integration of the Student t density with df=9.",
            "- Wilcoxon p-values are exact two-sided signed-rank p-values from enumerating all sign assignments.",
            "- Interpretations are generated directly from the paired deltas and are not copied from earlier baseline runs.",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

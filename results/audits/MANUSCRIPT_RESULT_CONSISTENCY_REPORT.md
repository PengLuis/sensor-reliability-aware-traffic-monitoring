# Manuscript Result Consistency Report

Status: **PASS**

| Check | Expected | Actual | Pass | Source |
|---|---:|---:|:---:|---|
| METR-LA ID-MLP-CA faulty average | 5.141500 | 5.141455 | PASS | `public paper-ready tables bundled in results/paper_ready_tables` |
| METR-LA SRAF-ID faulty average | 4.857600 | 4.857578 | PASS | `public paper-ready tables bundled in results/paper_ready_tables` |
| METR-LA relative reduction | 5.521000 | 5.521329 | PASS | `public paper-ready tables bundled in results/paper_ready_tables` |
| PEMS-BAY ID-MLP-CA faulty average | 1.991000 | 1.990985 | PASS | `public paper-ready tables bundled in results/paper_ready_tables` |
| PEMS-BAY SRAF-ID faulty average | 1.954200 | 1.954166 | PASS | `public paper-ready tables bundled in results/paper_ready_tables` |
| PEMS-BAY relative reduction | 1.849000 | 1.849298 | PASS | `public paper-ready tables bundled in results/paper_ready_tables` |
| positive fault-wise MAE comparisons | 11 | 11.000000 | PASS | `results/paper_ready_tables/supplementary_table_s3_faultwise_mae.csv` |
| positive horizon-12 comparisons | 11 | 11.000000 | PASS | `results/paper_ready_tables/supplementary_table_s6_horizon12.csv` |
| METR-LA parameter increase | 0.178000 | 0.178165 | PASS | `results/paper_ready_tables/supplementary_table_s4_complexity_latency.csv` |
| METR-LA latency ratio | 1.906000 | 1.906238 | PASS | `results/paper_ready_tables/supplementary_table_s4_complexity_latency.csv` |
| PEMS-BAY parameter increase | 0.173000 | 0.172635 | PASS | `results/paper_ready_tables/supplementary_table_s4_complexity_latency.csv` |
| PEMS-BAY latency ratio | 2.184000 | 2.184387 | PASS | `results/paper_ready_tables/supplementary_table_s4_complexity_latency.csv` |
| manuscript figure source data files | 8 | 8 | PASS | `results/figure_source_data/figure1_controlled_fault_protocol.csv through figure8_ablation_comparison.csv` |

All gains use `(baseline - SRAF-ID) / baseline * 100`. Faulty-input standard deviations in Table 3 are computed over ten seed-level six-fault averages (population SD, ddof=0), not over 60 pooled fault-seed observations.

The public figure-source directory contains manuscript-aligned source CSVs for Figure 1 through Figure 8. Schematic figures use component/source descriptions; numerical figures use the corresponding Table 3, Table 4, and Supplementary Tables S3, S5, and S6 evidence chain.

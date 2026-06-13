# Manuscript Result Consistency Report

Status: **PASS**

| Check | Expected | Actual | Pass | Source |
|---|---:|---:|:---:|---|
| METR-LA ID-MLP-CA faulty average | 5.1415 | 5.141455 | PASS | `formal per-seed metrics for matched ID-MLP-CA and SRAF-ID` |
| METR-LA SRAF-ID faulty average | 4.8576 | 4.857578 | PASS | `formal per-seed metrics for matched ID-MLP-CA and SRAF-ID` |
| METR-LA relative reduction | 5.521 | 5.521329 | PASS | `formal per-seed metrics for matched ID-MLP-CA and SRAF-ID` |
| PEMS-BAY ID-MLP-CA faulty average | 1.991 | 1.990985 | PASS | `formal per-seed metrics for matched ID-MLP-CA and SRAF-ID` |
| PEMS-BAY SRAF-ID faulty average | 1.9542 | 1.954166 | PASS | `formal per-seed metrics for matched ID-MLP-CA and SRAF-ID` |
| PEMS-BAY relative reduction | 1.849 | 1.849298 | PASS | `formal per-seed metrics for matched ID-MLP-CA and SRAF-ID` |
| positive fault-wise MAE comparisons | 11 | 11.000000 | PASS | `E:\paper_projects\sensors_faulty_sensor_paper\github_release\sraf-id-public-code\results\paper_ready_tables\supplementary_table_s3_faultwise_mae.csv` |
| positive horizon-12 comparisons | 11 | 11.000000 | PASS | `E:\paper_projects\sensors_faulty_sensor_paper\github_release\sraf-id-public-code\results\paper_ready_tables\supplementary_table_s6_horizon12.csv` |
| METR-LA parameter increase | 0.178 | 0.178165 | PASS | `E:\paper_projects\sensors_faulty_sensor_paper\github_release\sraf-id-public-code\results\paper_ready_tables\supplementary_table_s4_complexity_latency.csv` |
| METR-LA latency ratio | 1.906 | 1.906238 | PASS | `E:\paper_projects\sensors_faulty_sensor_paper\github_release\sraf-id-public-code\results\paper_ready_tables\supplementary_table_s4_complexity_latency.csv` |
| PEMS-BAY parameter increase | 0.173 | 0.172635 | PASS | `E:\paper_projects\sensors_faulty_sensor_paper\github_release\sraf-id-public-code\results\paper_ready_tables\supplementary_table_s4_complexity_latency.csv` |
| PEMS-BAY latency ratio | 2.184 | 2.184387 | PASS | `E:\paper_projects\sensors_faulty_sensor_paper\github_release\sraf-id-public-code\results\paper_ready_tables\supplementary_table_s4_complexity_latency.csv` |

All gains use `(baseline - SRAF-ID) / baseline * 100`. Faulty-input standard deviations are computed over ten seed-level six-fault averages (population SD, ddof=0), not over 60 pooled fault-seed observations.
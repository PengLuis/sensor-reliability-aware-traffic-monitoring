# final_summary_metrics

## SRAF-ID vs ID-MLP-CA
- METR-LA avg faulty gain: `3.953%`
- PEMS-BAY avg faulty gain: `1.579%`
- overall faulty pair win count: `11/12`
- h12 win count: `11/12`

## SRAF-ID vs KNN+ID-MLP
- METR-LA gain: `24.545%`
- PEMS-BAY gain: `24.420%`

## SRAF-ID vs PPCA-lite+ID-MLP
- METR-LA gain: `35.316%`
- PEMS-BAY gain: `29.399%`

## SRAF-ID vs PyPOTS-SAITS+ID-MLP
- METR-LA gain: `37.285%`
- PEMS-BAY gain: `36.123%`

## Negative cases vs ID-MLP-CA
- PEMS-BAY / continuous_outage_24: SRAF-ID `2.082120` vs ID-MLP-CA `1.997822`

## Ablation
- SRAF-ID is the no-gate repair-only version.
- SRAF-ID-gated variant is retained as an ablation/supplementary variant and is not adopted because it did not improve aggregate MAE.
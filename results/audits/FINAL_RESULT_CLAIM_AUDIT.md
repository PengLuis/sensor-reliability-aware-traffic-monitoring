# FINAL_RESULT_CLAIM_AUDIT

## SUPPORTED
- SRAF-ID improves over ID-MLP-CA on 11/12 faulty dataset-fault pairs.
- SRAF-ID reduces average faulty MAE vs ID-MLP-CA on METR-LA and PEMS-BAY.
- SRAF-ID outperforms KNN+ID-MLP, PPCA-lite+ID-MLP, and PyPOTS-SAITS+ID-MLP in average faulty MAE.
- SRAF-ID maintains comparable clean-input performance.

## CAUTIOUS
- Robustness improvement is modest relative to ID-MLP-CA.
- PEMS-BAY continuous_outage_24 remains a negative case.
- Gated variant was tested but not adopted.

## FORBIDDEN
- SRAF-ID is clean SOTA.
- SRAF-ID solves all faults.
- SRAF-ID beats every baseline on every fault.
- Reliability gate is the main contributor.
- Official STID or official GRIN reproduction.
- G6 adopted as final model.
- v1/v2 historical narrative in manuscript.
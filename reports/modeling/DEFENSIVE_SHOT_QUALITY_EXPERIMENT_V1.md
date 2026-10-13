# Defensive Shot Quality Experiment v1

Experimental only. Production model, Decision Policy v1, production inference, and frontend were not changed.

## Chronological Split
- test: 17,928 rows, 136 games, 2015-02-09 to 2015-03-04
- train: 90,922 rows, 632 games, 2014-10-28 to 2015-01-21
- validation: 19,219 rows, 136 games, 2015-01-21 to 2015-02-08

## Model Results
| model_key | label | estimator | calibration | validation_log_loss | test_log_loss | validation_brier | test_brier | test_roc_auc | test_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F_hgb | PT Defend as-of profile | hgb | isotonic | 0.643111 | 0.646378 | 0.227089 | 0.226586 | 0.647873 | 0.010330 |
| G_hgb | Full generalizable | hgb | isotonic | 0.643561 | 0.645725 | 0.227213 | 0.226277 | 0.649436 | 0.011333 |
| RAW_ID_hgb | Raw defender ID ablation | hgb | isotonic | 0.643684 | 0.647559 | 0.227389 | 0.226872 | 0.647941 | 0.011686 |
| E_hgb | Defender shot-log history | hgb | isotonic | 0.643933 | 0.646029 | 0.227441 | 0.226419 | 0.648659 | 0.011724 |
| D_hgb | Pressure + creation | hgb | isotonic | 0.644004 | 0.646254 | 0.227465 | 0.226705 | 0.647997 | 0.012514 |
| B_hgb | Defensive pressure | hgb | isotonic | 0.648942 | 0.652122 | 0.229397 | 0.229269 | 0.638932 | 0.010051 |
| RAW_ID_logistic | Raw defender ID ablation | logistic | isotonic | 0.652237 | 0.654507 | 0.230597 | 0.230678 | 0.635427 | 0.016386 |
| G_logistic | Full generalizable | logistic | isotonic | 0.652768 | 0.652897 | 0.230861 | 0.229962 | 0.639499 | 0.011689 |
| E_logistic | Defender shot-log history | logistic | isotonic | 0.653016 | 0.653348 | 0.230982 | 0.230178 | 0.638899 | 0.012334 |
| F_logistic | PT Defend as-of profile | logistic | isotonic | 0.653017 | 0.653390 | 0.230965 | 0.230251 | 0.638769 | 0.008454 |
| C_hgb | Offensive creation context | hgb | isotonic | 0.653161 | 0.655261 | 0.230967 | 0.230605 | 0.634728 | 0.009680 |
| D_logistic | Pressure + creation | logistic | isotonic | 0.653646 | 0.652816 | 0.231282 | 0.230275 | 0.638635 | 0.010500 |
| B_logistic | Defensive pressure | logistic | isotonic | 0.653927 | 0.655468 | 0.231447 | 0.230923 | 0.635918 | 0.009706 |
| C_logistic | Offensive creation context | logistic | isotonic | 0.660313 | 0.661518 | 0.234010 | 0.233866 | 0.624336 | 0.010433 |
| A_logistic | Base shot quality | logistic | isotonic | 0.661813 | 0.666154 | 0.234779 | 0.235197 | 0.618697 | 0.009900 |
| A_hgb | Base shot quality | hgb | isotonic | 0.661977 | 0.664257 | 0.234925 | 0.234781 | 0.620142 | 0.009244 |

## Recommendation
A. DEFENSIVE CONTEXT CLEARLY IMPROVES SHOT QUALITY

Closest-defender distance and creation context should be considered experimental inputs only until modern shot-level tracking validation is available.
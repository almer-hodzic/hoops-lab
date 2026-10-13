# Defensive Shot Quality V1.1

Experimental robustness/deployment experiment. Production model, Decision Policy v1, production API, and frontend were not modified.

## Split
| split | rows | games |
| --- | --- | --- |
| train | 90922 | 632 |
| validation | 19219 | 136 |
| test | 17928 | 136 |

## Variants Tested
| variant | label | uses_attempt_counts | uses_defender_profile | deployable |
| --- | --- | --- | --- | --- |
| V11_A | tracking + shooter rates only | False | False | True |
| V11_B | tracking + shooter rates + PT defender profile | False | True | True |
| V11_C | tracking + shooter rates + incompatible attempt counts | True | False | False |

## Candidate Ranking
| candidate_key | variant | model | calibration | validation_log_loss | validation_brier | validation_ece | test_log_loss | test_brier | test_ece | validation_pct_gt_95 | validation_near_rim_2pt_pct_gt_95 | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V11_A_hgb_regularized_sigmoid | V11_A | hgb_regularized | sigmoid | 0.647321 | 0.228564 | 0.010304 | 0.644358 | 0.226957 | 0.008954 | 0.000000 | 0.000000 | 0.647321 |
| V11_A_hgb_regularized_uncalibrated | V11_A | hgb_regularized | uncalibrated | 0.647683 | 0.228653 | 0.011600 | 0.645032 | 0.227226 | 0.016159 | 0.000000 | 0.000000 | 0.647683 |
| V11_A_hgb_v1_like_sigmoid | V11_A | hgb_v1_like | sigmoid | 0.646357 | 0.228239 | 0.009874 | 0.642422 | 0.226176 | 0.008633 | 0.002758 | 0.012740 | 0.648051 |
| V11_A_hgb_v1_like_uncalibrated | V11_A | hgb_v1_like | uncalibrated | 0.646470 | 0.228302 | 0.010016 | 0.642790 | 0.226346 | 0.015703 | 0.002810 | 0.012990 | 0.648198 |
| V11_A_hgb_conservative_sigmoid | V11_A | hgb_conservative | sigmoid | 0.649438 | 0.229408 | 0.011684 | 0.646479 | 0.227843 | 0.007680 | 0.000000 | 0.000000 | 0.649438 |
| V11_B_hgb_v1_like_uncalibrated | V11_B | hgb_v1_like | uncalibrated | 0.645839 | 0.227963 | 0.006686 | 0.642835 | 0.226327 | 0.009660 | 0.001561 | 0.006995 | 0.649772 |
| V11_B_hgb_v1_like_sigmoid | V11_B | hgb_v1_like | sigmoid | 0.645818 | 0.227955 | 0.008123 | 0.642734 | 0.226282 | 0.007247 | 0.001613 | 0.007245 | 0.649784 |
| V11_B_hgb_regularized_sigmoid | V11_B | hgb_regularized | sigmoid | 0.647010 | 0.228398 | 0.011607 | 0.643963 | 0.226751 | 0.009950 | 0.000000 | 0.000000 | 0.650010 |
| V11_B_hgb_regularized_uncalibrated | V11_B | hgb_regularized | uncalibrated | 0.647399 | 0.228487 | 0.013471 | 0.644624 | 0.227009 | 0.017142 | 0.000000 | 0.000000 | 0.650399 |
| V11_A_hgb_conservative_uncalibrated | V11_A | hgb_conservative | uncalibrated | 0.650468 | 0.229705 | 0.013198 | 0.647931 | 0.228390 | 0.020799 | 0.000000 | 0.000000 | 0.650468 |
| V11_B_hgb_conservative_sigmoid | V11_B | hgb_conservative | sigmoid | 0.649273 | 0.229318 | 0.011915 | 0.646058 | 0.227646 | 0.008056 | 0.000000 | 0.000000 | 0.652273 |
| V11_B_hgb_conservative_uncalibrated | V11_B | hgb_conservative | uncalibrated | 0.650381 | 0.229646 | 0.014260 | 0.647621 | 0.228237 | 0.021928 | 0.000000 | 0.000000 | 0.653381 |
| V11_A_logistic_sigmoid | V11_A | logistic | sigmoid | 0.657546 | 0.232761 | 0.017220 | 0.653652 | 0.230928 | 0.014802 | 0.000312 | 0.001499 | 0.657805 |
| V11_A_logistic_uncalibrated | V11_A | logistic | uncalibrated | 0.657697 | 0.232866 | 0.023234 | 0.653859 | 0.231066 | 0.021231 | 0.000416 | 0.001749 | 0.657992 |
| V11_A_hgb_conservative_isotonic | V11_A | hgb_conservative | isotonic | 0.646234 | 0.228352 | 0.000000 | 0.650135 | 0.228226 | 0.008898 | 0.007284 | 0.034474 | 0.658046 |
| V11_C_hgb_regularized_sigmoid | V11_C | hgb_regularized | sigmoid | 0.647224 | 0.228522 | 0.010171 | 0.644155 | 0.226863 | 0.008753 | 0.000000 | 0.000000 | 0.659224 |
| V11_A_hgb_v1_like_isotonic | V11_A | hgb_v1_like | isotonic | 0.643779 | 0.227428 | 0.000000 | 0.645923 | 0.226369 | 0.010027 | 0.008273 | 0.034724 | 0.659277 |
| V11_C_hgb_regularized_uncalibrated | V11_C | hgb_regularized | uncalibrated | 0.647671 | 0.228651 | 0.013251 | 0.644993 | 0.227210 | 0.018669 | 0.000000 | 0.000000 | 0.659671 |
| V11_C_hgb_v1_like_sigmoid | V11_C | hgb_v1_like | sigmoid | 0.646655 | 0.228303 | 0.006802 | 0.642936 | 0.226392 | 0.009839 | 0.001717 | 0.007994 | 0.659717 |
| V11_B_hgb_v1_like_isotonic | V11_B | hgb_v1_like | isotonic | 0.642945 | 0.227007 | 0.000000 | 0.646036 | 0.226707 | 0.008445 | 0.006244 | 0.026480 | 0.659853 |

## Selected Model
```json
{
  "candidate_key": "V11_A_hgb_regularized_sigmoid",
  "variant": "V11_A",
  "variant_label": "tracking + shooter rates only",
  "model": "hgb_regularized",
  "calibration": "sigmoid",
  "deployable": true,
  "uses_attempt_counts": false,
  "uses_defender_profile": false,
  "validation_rows": 19219,
  "validation_actual_fg_pct": 0.44913887299027005,
  "validation_predicted_fg_pct": 0.4491351733859341,
  "validation_log_loss": 0.6473207409488423,
  "validation_brier": 0.22856438669124912,
  "validation_roc_auc": 0.6402472916712078,
  "validation_ece": 0.010303549653273732,
  "validation_pct_lt_01": 0.0,
  "validation_pct_gt_90": 0.008741349706020085,
  "validation_pct_gt_95": 0.0,
  "validation_pct_gt_99": 0.0,
  "validation_min_probability": 0.0652316276424828,
  "validation_max_probability": 0.9453867318513238,
  "validation_near_rim_2pt_pct_gt_90": 0.039470397202098426,
  "validation_near_rim_2pt_pct_gt_95": 0.0,
  "validation_near_rim_2pt_pct_gt_99": 0.0,
  "validation_near_rim_2pt_max_probability": 0.9453867318513238,
  "validation_midrange_2pt_pct_gt_90": 0.0,
  "validation_midrange_2pt_pct_gt_95": 0.0,
  "validation_midrange_2pt_pct_gt_99": 0.0,
  "validation_midrange_2pt_max_probability": 0.6969741632488243,
  "validation_three_pt_pct_gt_90": 0.0,
  "validation_three_pt_pct_gt_95": 0.0,
  "validation_three_pt_pct_gt_99": 0.0,
  "validation_three_pt_max_probability": 0.7240952515324561,
  "test_rows": 17928,
  "test_actual_fg_pct": 0.4428826416778224,
  "test_predicted_fg_pct": 0.4503380223098585,
  "test_log_loss": 0.6443583638428471,
  "test_brier": 0.22695692293624956,
  "test_roc_auc": 0.6469555532129739,
  "test_ece": 0.0089536507912231,
  "test_pct_lt_01": 0.0,
  "test_pct_gt_90": 0.010095939312806782,
  "test_pct_gt_95": 0.0,
  "test_pct_gt_99": 0.0,
  "test_min_probability": 0.06493419626537214,
  "test_max_probability": 0.9484809718569011,
  "test_near_rim_2pt_pct_gt_90": 0.045321247667288726,
  "test_near_rim_2pt_pct_gt_95": 0.0,
  "test_near_rim_2pt_pct_gt_99": 0.0,
  "test_near_rim_2pt_max_probability": 0.9484809718569011,
  "test_midrange_2pt_pct_gt_90": 0.0,
  "test_midrange_2pt_pct_gt_95": 0.0,
  "test_midrange_2pt_pct_gt_99": 0.0,
  "test_midrange_2pt_max_probability": 0.7039605353543935,
  "test_three_pt_pct_gt_90": 0.0,
  "test_three_pt_pct_gt_95": 0.0,
  "test_three_pt_pct_gt_99": 0.0,
  "test_three_pt_max_probability": 0.5757009203970617,
  "selection_penalty": 0.0,
  "selection_score": 0.6473207409488423
}
```

## Experimental V1 Comparison
```json
{
  "selected_model_key": "F_hgb",
  "validation_rows": 19219,
  "validation_actual_fg_pct": 0.44913887299027005,
  "validation_predicted_fg_pct": 0.44913886913991363,
  "validation_log_loss": 0.6431108650034079,
  "validation_brier": 0.2270888960892776,
  "validation_roc_auc": 0.6441974931572312,
  "validation_ece": 7.804776542447e-09,
  "validation_pct_lt_01": 0.001977210052552162,
  "validation_pct_gt_90": 0.02695249492689526,
  "validation_pct_gt_95": 0.00946979551485509,
  "validation_pct_gt_99": 0.005827566470680056,
  "validation_min_probability": 1e-06,
  "validation_max_probability": 0.999999,
  "validation_near_rim_2pt_pct_gt_90": 0.11416437671746191,
  "validation_near_rim_2pt_pct_gt_95": 0.04071946040469648,
  "validation_near_rim_2pt_pct_gt_99": 0.02498126405196103,
  "validation_near_rim_2pt_max_probability": 0.999999,
  "validation_midrange_2pt_pct_gt_90": 0.0,
  "validation_midrange_2pt_pct_gt_95": 0.0,
  "validation_midrange_2pt_pct_gt_99": 0.0,
  "validation_midrange_2pt_max_probability": 0.8347826086956521,
  "validation_three_pt_pct_gt_90": 0.0,
  "validation_three_pt_pct_gt_95": 0.0,
  "validation_three_pt_pct_gt_99": 0.0,
  "validation_three_pt_max_probability": 0.6363636363636366,
  "test_rows": 17928,
  "test_actual_fg_pct": 0.4428826416778224,
  "test_predicted_fg_pct": 0.4514105372005718,
  "test_log_loss": 0.646378401318959,
  "test_brier": 0.22658637994001327,
  "test_roc_auc": 0.6478732098165154,
  "test_ece": 0.010329556074622815,
  "test_pct_lt_01": 0.002175368139223561,
  "test_pct_gt_90": 0.027666220437304774,
  "test_pct_gt_95": 0.01109995537706381,
  "test_pct_gt_99": 0.007697456492637216,
  "test_min_probability": 1e-06,
  "test_max_probability": 0.999999,
  "test_near_rim_2pt_pct_gt_90": 0.11890162623300453,
  "test_near_rim_2pt_pct_gt_95": 0.04825379898693682,
  "test_near_rim_2pt_pct_gt_99": 0.03385763796320981,
  "test_near_rim_2pt_max_probability": 0.999999,
  "test_midrange_2pt_pct_gt_90": 0.0,
  "test_midrange_2pt_pct_gt_95": 0.0,
  "test_midrange_2pt_pct_gt_99": 0.0,
  "test_midrange_2pt_max_probability": 0.86,
  "test_three_pt_pct_gt_90": 0.0,
  "test_three_pt_pct_gt_95": 0.0,
  "test_three_pt_pct_gt_99": 0.0,
  "test_three_pt_max_probability": 0.501945525291829
}
```

## Representative Support-Aware Curves
| scenario | closest_defender_distance | raw_probability | calibrated_probability | supportStatus | sparseContext |
| --- | --- | --- | --- | --- | --- |
| median_near_rim_2pt | 0.500000 | 0.498269 | 0.495091 | IN_DISTRIBUTION | False |
| median_near_rim_2pt | 1 | 0.498269 | 0.495091 | IN_DISTRIBUTION | False |
| median_near_rim_2pt | 2 | 0.507908 | 0.505640 | IN_DISTRIBUTION | False |
| median_near_rim_2pt | 3 | 0.631087 | 0.640030 | IN_DISTRIBUTION | False |
| median_near_rim_2pt | 4 | 0.722754 | 0.738180 | IN_DISTRIBUTION | False |
| median_near_rim_2pt | 5 | 0.759178 | 0.776342 | LIMITED_SUPPORT | True |
| median_near_rim_2pt | 6 | 0.821353 | 0.839895 | LIMITED_SUPPORT | True |
| median_near_rim_2pt | 7 | 0.866662 | 0.884560 | LIMITED_SUPPORT | True |
| median_near_rim_2pt | 8 | 0.891055 | 0.907864 | LIMITED_SUPPORT | True |
| median_near_rim_2pt | 10 | 0.915986 | 0.931016 | LIMITED_SUPPORT | True |
| strong_2pt_near_rim_2pt | 0.500000 | 0.615605 | 0.623239 | IN_DISTRIBUTION | False |
| strong_2pt_near_rim_2pt | 1 | 0.615605 | 0.623239 | IN_DISTRIBUTION | False |
| strong_2pt_near_rim_2pt | 2 | 0.632173 | 0.641205 | IN_DISTRIBUTION | False |
| strong_2pt_near_rim_2pt | 3 | 0.779377 | 0.797231 | IN_DISTRIBUTION | False |
| strong_2pt_near_rim_2pt | 4 | 0.844119 | 0.862541 | IN_DISTRIBUTION | False |
| strong_2pt_near_rim_2pt | 5 | 0.858780 | 0.876910 | LIMITED_SUPPORT | True |
| strong_2pt_near_rim_2pt | 6 | 0.880941 | 0.898273 | LIMITED_SUPPORT | True |
| strong_2pt_near_rim_2pt | 7 | 0.903721 | 0.919718 | LIMITED_SUPPORT | True |
| strong_2pt_near_rim_2pt | 8 | 0.918320 | 0.933145 | LIMITED_SUPPORT | True |
| strong_2pt_near_rim_2pt | 10 | 0.925909 | 0.940012 | LIMITED_SUPPORT | True |
| weak_2pt_near_rim_2pt | 0.500000 | 0.597593 | 0.603652 | LIMITED_SUPPORT | False |
| weak_2pt_near_rim_2pt | 1 | 0.597593 | 0.603652 | LIMITED_SUPPORT | False |
| weak_2pt_near_rim_2pt | 2 | 0.614452 | 0.621986 | LIMITED_SUPPORT | False |
| weak_2pt_near_rim_2pt | 3 | 0.779246 | 0.797096 | LIMITED_SUPPORT | False |
| weak_2pt_near_rim_2pt | 4 | 0.847777 | 0.866143 | LIMITED_SUPPORT | False |
| weak_2pt_near_rim_2pt | 5 | 0.863635 | 0.881629 | LIMITED_SUPPORT | True |
| weak_2pt_near_rim_2pt | 6 | 0.883356 | 0.900572 | LIMITED_SUPPORT | True |
| weak_2pt_near_rim_2pt | 7 | 0.903984 | 0.919963 | LIMITED_SUPPORT | True |
| weak_2pt_near_rim_2pt | 8 | 0.916145 | 0.931162 | LIMITED_SUPPORT | True |
| weak_2pt_near_rim_2pt | 10 | 0.928442 | 0.942286 | LIMITED_SUPPORT | True |
| median_midrange_2pt | 0.500000 | 0.404230 | 0.392567 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 1 | 0.404230 | 0.392567 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 2 | 0.404230 | 0.392567 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 3 | 0.414417 | 0.403616 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 4 | 0.443010 | 0.434720 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 5 | 0.450152 | 0.442507 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 6 | 0.461057 | 0.454408 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 7 | 0.481412 | 0.476651 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 8 | 0.482707 | 0.478068 | IN_DISTRIBUTION | False |
| median_midrange_2pt | 10 | 0.509038 | 0.506877 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 0.500000 | 0.370298 | 0.355923 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 1 | 0.370298 | 0.355923 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 2 | 0.370298 | 0.355923 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 3 | 0.388490 | 0.375537 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 4 | 0.427248 | 0.417558 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 5 | 0.436346 | 0.427459 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 6 | 0.442314 | 0.433961 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 7 | 0.457681 | 0.450722 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 8 | 0.458969 | 0.452129 | IN_DISTRIBUTION | False |
| strong_2pt_midrange_2pt | 10 | 0.463854 | 0.457463 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 0.500000 | 0.303333 | 0.284551 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 1 | 0.303333 | 0.284551 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 2 | 0.303333 | 0.284551 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 3 | 0.324935 | 0.307413 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 4 | 0.373550 | 0.359423 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 5 | 0.395269 | 0.382866 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 6 | 0.410390 | 0.399246 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 7 | 0.444756 | 0.436623 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 8 | 0.444756 | 0.436623 | IN_DISTRIBUTION | False |
| weak_2pt_midrange_2pt | 10 | 0.464399 | 0.458057 | IN_DISTRIBUTION | False |
| median_three_pt | 0.500000 | 0.324067 | 0.306490 | IN_DISTRIBUTION | False |
| median_three_pt | 1 | 0.324067 | 0.306490 | IN_DISTRIBUTION | False |
| median_three_pt | 2 | 0.324067 | 0.306490 | IN_DISTRIBUTION | False |
| median_three_pt | 3 | 0.327145 | 0.309761 | IN_DISTRIBUTION | False |
| median_three_pt | 4 | 0.341684 | 0.325251 | IN_DISTRIBUTION | False |
| median_three_pt | 5 | 0.364179 | 0.349345 | IN_DISTRIBUTION | False |
| median_three_pt | 6 | 0.366014 | 0.351317 | IN_DISTRIBUTION | False |
| median_three_pt | 7 | 0.371284 | 0.356984 | IN_DISTRIBUTION | False |
| median_three_pt | 8 | 0.372496 | 0.358289 | IN_DISTRIBUTION | False |
| median_three_pt | 10 | 0.377102 | 0.363250 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 0.500000 | 0.327648 | 0.310296 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 1 | 0.327648 | 0.310296 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 2 | 0.327648 | 0.310296 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 3 | 0.340105 | 0.323565 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 4 | 0.350416 | 0.334586 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 5 | 0.373744 | 0.359633 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 6 | 0.375600 | 0.361631 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 7 | 0.382354 | 0.368912 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 8 | 0.383580 | 0.370236 | IN_DISTRIBUTION | False |
| strong_2pt_three_pt | 10 | 0.388239 | 0.375266 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 0.500000 | 0.331665 | 0.314569 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 1 | 0.331665 | 0.314569 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 2 | 0.331665 | 0.314569 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 3 | 0.337455 | 0.320739 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 4 | 0.351078 | 0.335294 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 5 | 0.374020 | 0.359930 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 6 | 0.375875 | 0.361928 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 7 | 0.383347 | 0.369984 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 8 | 0.384574 | 0.371309 | IN_DISTRIBUTION | False |
| weak_2pt_three_pt | 10 | 0.401030 | 0.389101 | IN_DISTRIBUTION | False |

## Modern Transfer Scenarios
Jokic:
| distance | probability | supportStatus | sparseContext | supportWarnings |
| --- | --- | --- | --- | --- |
| 1 | 0.638808 | OUT_OF_DISTRIBUTION | False | One or more numeric features are outside training support thresholds. |
| 3 | 0.753166 | OUT_OF_DISTRIBUTION | False | One or more numeric features are outside training support thresholds. |
| 6 | 0.874115 | OUT_OF_DISTRIBUTION | True | Nearest 2014/15 support for this scenario is sparse.; One or more numeric features are outside training support thresholds. |
| 8 | 0.924836 | OUT_OF_DISTRIBUTION | True | Nearest 2014/15 support for this scenario is sparse.; One or more numeric features are outside training support thresholds. |

Curry:
| distance | probability | supportStatus | sparseContext | supportWarnings |
| --- | --- | --- | --- | --- |
| 1 | 0.302396 | IN_DISTRIBUTION | False |  |
| 3 | 0.312214 | IN_DISTRIBUTION | False |  |
| 6 | 0.355513 | IN_DISTRIBUTION | False |  |
| 8 | 0.369712 | IN_DISTRIBUTION | False |  |

## Recommendation
Use display policy A: show experimental P(make) normally only when IN_DISTRIBUTION, show a warning for LIMITED_SUPPORT, and hide absolute probability for OUT_OF_DISTRIBUTION while still allowing context/sensitivity exploration.
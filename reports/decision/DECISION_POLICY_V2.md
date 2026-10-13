# Decision Policy v2

## Method
Decision Policy v2 keeps the completed Shooter Skill v1.1 historical shooter-profile model frozen and replaces the v1 generic replacement value with a learned possession-continuation value from pre-shot context.

`decision_margin_v2 = shot_xpps - continuation_xpps`; TAKE and DEFER are assigned outside the validation-selected REVIEW band.

## Continuation Models
| Model         | Validation MAE | Validation RMSE | Validation Bias |
| ------------- | -------------- | --------------- | --------------- |
| global        | 1.1608         | 1.2575          | -0.0068         |
| context_table | 1.1597         | 1.2568          | -0.0072         |
| ridge         | 1.1520         | 1.2537          | 0.0167          |
| hgb           | 1.1521         | 1.2531          | -0.0119         |

Selected continuation model: `hgb`
Selected v2 threshold: `0.2`

## Frozen Evaluation
| Dataset                | Decision | Rows   | Share | Shot xPPS | Cont xPPS | Shot PPS | Poss PPP |
| ---------------------- | -------- | ------ | ----- | --------- | --------- | -------- | -------- |
| 2025_26_regular        | TAKE     | 21116  | 0.096 | 1.675     | 1.245     | 1.684    | 1.694    |
| 2025_26_regular        | REVIEW   | 77666  | 0.354 | 1.180     | 1.248     | 1.189    | 1.375    |
| 2025_26_regular        | DEFER    | 120376 | 0.549 | 0.937     | 1.327     | 0.924    | 1.240    |
| 2025_26_regular_clutch | TAKE     | 519    | 0.073 | 1.659     | 1.234     | 1.672    | 1.690    |
| 2025_26_regular_clutch | REVIEW   | 1740   | 0.245 | 1.152     | 1.212     | 1.186    | 1.399    |
| 2025_26_regular_clutch | DEFER    | 4840   | 0.682 | 0.917     | 1.347     | 0.866    | 1.249    |
| 2025_26_playoffs       | TAKE     | 1578   | 0.109 | 1.663     | 1.237     | 1.628    | 1.648    |
| 2025_26_playoffs       | REVIEW   | 5740   | 0.397 | 1.178     | 1.236     | 1.115    | 1.344    |
| 2025_26_playoffs       | DEFER    | 7155   | 0.494 | 0.951     | 1.332     | 0.858    | 1.201    |
| 2025_26_playoff_clutch | TAKE     | 61     | 0.128 | 1.659     | 1.224     | 1.508    | 1.574    |
| 2025_26_playoff_clutch | REVIEW   | 128    | 0.269 | 1.167     | 1.212     | 0.938    | 1.219    |
| 2025_26_playoff_clutch | DEFER    | 287    | 0.603 | 0.921     | 1.352     | 0.763    | 1.084    |

## V1 vs V2
| Dataset                | Rows   | Agreement |
| ---------------------- | ------ | --------- |
| 2025_26_playoff_clutch | 476    | 0.546     |
| 2025_26_playoffs       | 14473  | 0.590     |
| 2025_26_regular        | 219158 | 0.579     |
| 2025_26_regular_clutch | 7099   | 0.532     |

## Validation Notes
The possession builder linked every canonical shot to a possession. Remaining possession-quality flags are carried forward in `reports/possession/POSSESSION_VALIDATION.md` and should be treated as known noise for continuation modeling.
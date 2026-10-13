# Experimental Inference Support Audit

Forensic audit of the experimental Defensive Shot Quality V1 integration. Production inference and Decision Policy v1 were not modified.

## Exact Jokic Probability Curve
| closest_defender_distance | raw_probability | calibrated_probability |
| --- | --- | --- |
| 0.500000 | 0.620702 | 0.621118 |
| 1 | 0.620702 | 0.621118 |
| 2 | 0.638699 | 0.636364 |
| 3 | 0.712895 | 0.697509 |
| 4 | 0.791565 | 0.762712 |
| 5 | 0.856982 | 0.929461 |
| 6 | 0.892296 | 0.929461 |
| 7 | 0.907798 | 0.971429 |
| 8 | 0.918160 | 0.999999 |
| 10 | 0.950070 | 0.999999 |

## Raw Model Input Rows
| closest_defender_distance | shot_distance | shot_value | period | seconds_remaining_period | is_home | shooter_prior_fga | shooter_prior_fg_pct_shrunk | shooter_prior_2pa | shooter_prior_2p_pct_shrunk | shooter_prior_3pa | shooter_prior_3p_pct_shrunk | shooter_distance_prior_attempts | shooter_distance_prior_fg_pct_shrunk | distance_bucket | shooter_cold_start | closest_defender_distance | shot_clock | dribbles | touch_time | shot_clock_missing | touch_time_missing | touch_time_invalid | defender_pt_overall_d_fg_pct | defender_pt_overall_d_fga | defender_pt_overall_normal_fg_pct | defender_pt_overall_pct_plusminus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.500000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 0.500000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 1.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 1.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 2.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 2.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 3.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 3.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 4.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 4.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 5.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 5.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 6.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 6.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 7.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 7.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 8.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 8.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |
| 10.000000 | 3 | 2 | 4 | 240.000000 | True | 8914 | 0.572198 | 7076 | 0.624712 | 1838 | 0.373141 | 2892 | 0.719732 | 0-4 | False | 10.000000 | 12.000000 | 0 | 1.200000 | False | False | False | 0.453000 | 808.000000 | 0.491000 | -0.038000 |

## Raw vs Calibrated Trace
| closest_defender_distance | raw_probability | calibrated_probability |
| --- | --- | --- |
| 1 | 0.620702 | 0.621118 |
| 3 | 0.712895 | 0.697509 |
| 6 | 0.892296 | 0.929461 |

## Modern Shooter Adapter Semantics
| feature | 2014/15 training semantics | training median | training p95 | training max | Jokic modern value | support |
| --- | --- | --- | --- | --- | --- | --- |
| shooter_prior_fga | 2014/15 season-to-date FGA by shooter from completed previous games only; current game excluded. | 168.000000 | 495.000000 | 751.000000 | 8914 | SEMANTIC_MISMATCH |
| shooter_prior_fg_pct_shrunk | 2014/15 season-to-date overall FG% by shooter from completed previous games only, Bayesian-shrunk toward train population overall FG%. | 0.454598 | 0.510548 | 0.640179 | 0.572198 | OUTSIDE_P1_P99 |
| shooter_prior_2pa | 2014/15 season-to-date 2PA by shooter from completed previous games only; current game excluded. | 117.000000 | 390.000000 | 712.000000 | 7076 | SEMANTIC_MISMATCH |
| shooter_prior_2p_pct_shrunk | 2014/15 season-to-date 2P% by shooter from completed previous games only, Bayesian-shrunk toward train population 2P%. | 0.489416 | 0.544500 | 0.668201 | 0.624712 | OUTSIDE_P1_P99 |
| shooter_prior_3pa | 2014/15 season-to-date 3PA by shooter from completed previous games only; current game excluded. | 32.000000 | 163.000000 | 323.000000 | 1838 | SEMANTIC_MISMATCH |
| shooter_prior_3p_pct_shrunk | 2014/15 season-to-date 3P% by shooter from completed previous games only, Bayesian-shrunk toward train population 3P%. | 0.353926 | 0.394198 | 0.492180 | 0.373141 | IN_SUPPORT |
| shooter_distance_prior_attempts | 2014/15 season-to-date attempts by shooter and experimental distance bucket from completed previous games only; current game excluded. | 41.000000 | 146.000000 | 307.000000 | 2892 | SEMANTIC_MISMATCH |
| shooter_distance_prior_fg_pct_shrunk | 2014/15 season-to-date FG% by shooter and experimental distance bucket from completed previous games only, Bayesian-shrunk toward train population. | 0.449862 | 0.585871 | 0.713203 | 0.719732 | OUTSIDE_MIN_MAX |

## Numeric Distribution Support
| feature | min | p1 | p5 | median | p95 | p99 | max | Jokic | Jokic flag | Curry | Curry flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shot_distance | 0.000000 | 0.700000 | 1.700000 | 13.700000 | 25.500000 | 27.700000 | 47.200000 | 3 | IN_SUPPORT | 26 | IN_SUPPORT |
| shot_value | 2.000000 | 2.000000 | 2.000000 | 2.000000 | 3.000000 | 3.000000 | 3.000000 | 2 | IN_SUPPORT | 3 | IN_SUPPORT |
| period | 1.000000 | 1.000000 | 1.000000 | 2.000000 | 4.000000 | 4.000000 | 7.000000 | 4 | IN_SUPPORT | 4 | IN_SUPPORT |
| seconds_remaining_period | 0.000000 | 1.000000 | 28.000000 | 352.000000 | 675.000000 | 705.000000 | 720.000000 | 240.000000 | IN_SUPPORT | 240.000000 | IN_SUPPORT |
| is_home | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True | IN_SUPPORT | True | IN_SUPPORT |
| shooter_prior_fga | 0.000000 | 0.000000 | 11.000000 | 168.000000 | 495.000000 | 611.000000 | 751.000000 | 8914 | SEMANTIC_MISMATCH | 8105 | SEMANTIC_MISMATCH |
| shooter_prior_fg_pct_shrunk | 0.351076 | 0.384469 | 0.403642 | 0.454598 | 0.510548 | 0.546538 | 0.640179 | 0.572198 | OUTSIDE_P1_P99 | 0.462677 | IN_SUPPORT |
| shooter_prior_2pa | 0.000000 | 0.000000 | 7.000000 | 117.000000 | 390.000000 | 511.000000 | 712.000000 | 7076 | SEMANTIC_MISMATCH | 3305 | SEMANTIC_MISMATCH |
| shooter_prior_2p_pct_shrunk | 0.381717 | 0.419432 | 0.435654 | 0.489416 | 0.544500 | 0.575497 | 0.668201 | 0.624712 | OUTSIDE_P1_P99 | 0.549172 | IN_SUPPORT |
| shooter_prior_3pa | 0.000000 | 0.000000 | 0.000000 | 32.000000 | 163.000000 | 230.000000 | 323.000000 | 1838 | SEMANTIC_MISMATCH | 4800 | SEMANTIC_MISMATCH |
| shooter_prior_3p_pct_shrunk | 0.270104 | 0.296221 | 0.315544 | 0.353926 | 0.394198 | 0.423487 | 0.492180 | 0.373141 | IN_SUPPORT | 0.401867 | IN_SUPPORT |
| shooter_distance_prior_attempts | 0.000000 | 0.000000 | 2.000000 | 41.000000 | 146.000000 | 206.000000 | 307.000000 | 2892 | SEMANTIC_MISMATCH | 4077 | SEMANTIC_MISMATCH |
| shooter_distance_prior_fg_pct_shrunk | 0.283847 | 0.327720 | 0.362017 | 0.449862 | 0.585871 | 0.645792 | 0.713203 | 0.719732 | OUTSIDE_MIN_MAX | 0.396409 | IN_SUPPORT |
| closest_defender_distance | 0.000000 | 0.100000 | 0.800000 | 3.700000 | 8.800000 | 14.100000 | 53.200000 | 3.000000 | IN_SUPPORT | 3.000000 | IN_SUPPORT |
| shot_clock | 0.000000 | 1.000000 | 3.000000 | 12.300000 | 22.500000 | 24.000000 | 24.000000 | 12.000000 | IN_SUPPORT | 12.000000 | IN_SUPPORT |
| dribbles | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 10.000000 | 17.000000 | 32.000000 | 0 | IN_SUPPORT | 0 | IN_SUPPORT |
| touch_time | 0.000000 | 0.000000 | 0.400000 | 1.600000 | 8.800000 | 14.800000 | 24.500000 | 1.200000 | IN_SUPPORT | 1.200000 | IN_SUPPORT |
| shot_clock_missing | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | False | IN_SUPPORT | False | IN_SUPPORT |
| touch_time_missing | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | IN_SUPPORT | False | IN_SUPPORT |
| touch_time_invalid | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | IN_SUPPORT | False | IN_SUPPORT |
| defender_pt_overall_d_fg_pct | 0.000000 | 0.100000 | 0.276000 | 0.452000 | 0.588000 | 0.750000 | 1.000000 | 0.453000 | IN_SUPPORT | 0.453000 | IN_SUPPORT |
| defender_pt_overall_d_fga | 1.000000 | 2.000000 | 6.000000 | 107.000000 | 347.000000 | 440.000000 | 513.000000 | 808.000000 | OUTSIDE_MIN_MAX | 808.000000 | OUTSIDE_MIN_MAX |
| defender_pt_overall_normal_fg_pct | 0.227000 | 0.396000 | 0.413000 | 0.445000 | 0.481000 | 0.502000 | 0.604000 | 0.491000 | IN_SUPPORT | 0.491000 | IN_SUPPORT |
| defender_pt_overall_pct_plusminus | -0.479000 | -0.312000 | -0.157000 | 0.005000 | 0.141000 | 0.295000 | 0.645000 | -0.038000 | IN_SUPPORT | -0.038000 | IN_SUPPORT |

## Transformed Feature Trace
Key selected Jokic row at 6 ft:

```json
{
  "selectedColumns": {
    "shot_distance": 3,
    "shot_value": 2,
    "period": 4,
    "seconds_remaining_period": 240.0,
    "is_home": true,
    "shooter_prior_fga": 8914,
    "shooter_prior_fg_pct_shrunk": 0.572197883221409,
    "shooter_prior_2pa": 7076,
    "shooter_prior_2p_pct_shrunk": 0.6247115761001854,
    "shooter_prior_3pa": 1838,
    "shooter_prior_3p_pct_shrunk": 0.373140833022283,
    "shooter_distance_prior_attempts": 2892,
    "shooter_distance_prior_fg_pct_shrunk": 0.7197319947772476,
    "distance_bucket": "0-4",
    "shooter_cold_start": false,
    "closest_defender_distance": 6.0,
    "shot_clock": 12.0,
    "dribbles": 0,
    "touch_time": 1.2,
    "shot_clock_missing": false,
    "touch_time_missing": false,
    "touch_time_invalid": false,
    "defender_pt_overall_d_fg_pct": 0.453,
    "defender_pt_overall_d_fga": 808.0,
    "defender_pt_overall_normal_fg_pct": 0.491,
    "defender_pt_overall_pct_plusminus": -0.038
  },
  "hgbInputLength": 32,
  "hgbInputHead": [
    3.0,
    2.0,
    4.0,
    240.0,
    1.0,
    8914.0,
    0.572197883221409,
    7076.0,
    0.6247115761001854,
    1838.0,
    0.373140833022283,
    2892.0,
    0.7197319947772476,
    6.0,
    12.0,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
    0.453,
    808.0,
    0.491,
    -0.038,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ]
}
```

## Nearest Training Neighborhood
Summary:

```json
{
  "candidate_rows": 3291,
  "nearest_rows": 500,
  "actual_fg_pct": 0.664,
  "mean_shooter_prior_2p_pct_shrunk": 0.5118256497348175,
  "mean_shooter_distance_prior_fg_pct_shrunk": 0.6066501979969922,
  "defender_distance_min": 0.0,
  "defender_distance_median": 2.2,
  "defender_distance_p95": 5.104999999999995,
  "rows_5_to_7_ft": 28,
  "fg_pct_5_to_7_ft": 0.8571428571428571
}
```

| defender_distance_bucket | rows | actual_fg_pct |
| --- | --- | --- |
| 0-2 | 223 | 0.573991 |
| 2-4 | 214 | 0.686916 |
| 4-6 | 55 | 0.909091 |
| 6-8 | 7 | 0.857143 |
| 8+ | 1 | 1.000000 |

## Representative Training-Supported Sensitivity Curves
| scenario | closest_defender_distance | calibrated_probability |
| --- | --- | --- |
| median_2014_15_shooter_near_rim_2pt | 0.500000 | 0.511976 |
| median_2014_15_shooter_near_rim_2pt | 1 | 0.511976 |
| median_2014_15_shooter_near_rim_2pt | 2 | 0.533195 |
| median_2014_15_shooter_near_rim_2pt | 3 | 0.655260 |
| median_2014_15_shooter_near_rim_2pt | 4 | 0.762712 |
| median_2014_15_shooter_near_rim_2pt | 5 | 0.762712 |
| median_2014_15_shooter_near_rim_2pt | 6 | 0.762712 |
| median_2014_15_shooter_near_rim_2pt | 7 | 0.916667 |
| median_2014_15_shooter_near_rim_2pt | 8 | 0.929461 |
| median_2014_15_shooter_near_rim_2pt | 10 | 0.999999 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 0.500000 | 0.655260 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 1 | 0.655260 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 2 | 0.655260 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 3 | 0.834783 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 4 | 0.916667 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 5 | 0.929461 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 6 | 0.971429 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 7 | 0.999999 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 8 | 0.999999 |
| strong_2014_15_2pt_shooter_near_rim_2pt | 10 | 0.999999 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 0.500000 | 0.533195 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 1 | 0.533195 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 2 | 0.653846 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 3 | 0.784314 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 4 | 0.929461 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 5 | 0.929461 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 6 | 0.929461 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 7 | 0.999999 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 8 | 0.999999 |
| weak_2014_15_2pt_shooter_near_rim_2pt | 10 | 0.999999 |
| median_2014_15_shooter_midrange_2pt | 0.500000 | 0.400744 |
| median_2014_15_shooter_midrange_2pt | 1 | 0.400744 |
| median_2014_15_shooter_midrange_2pt | 2 | 0.414128 |
| median_2014_15_shooter_midrange_2pt | 3 | 0.414128 |
| median_2014_15_shooter_midrange_2pt | 4 | 0.446182 |
| median_2014_15_shooter_midrange_2pt | 5 | 0.446182 |
| median_2014_15_shooter_midrange_2pt | 6 | 0.446182 |
| median_2014_15_shooter_midrange_2pt | 7 | 0.463248 |
| median_2014_15_shooter_midrange_2pt | 8 | 0.463248 |
| median_2014_15_shooter_midrange_2pt | 10 | 0.501946 |
| strong_2014_15_2pt_shooter_midrange_2pt | 0.500000 | 0.367423 |
| strong_2014_15_2pt_shooter_midrange_2pt | 1 | 0.367423 |
| strong_2014_15_2pt_shooter_midrange_2pt | 2 | 0.367423 |
| strong_2014_15_2pt_shooter_midrange_2pt | 3 | 0.367423 |
| strong_2014_15_2pt_shooter_midrange_2pt | 4 | 0.414128 |
| strong_2014_15_2pt_shooter_midrange_2pt | 5 | 0.414128 |
| strong_2014_15_2pt_shooter_midrange_2pt | 6 | 0.414128 |
| strong_2014_15_2pt_shooter_midrange_2pt | 7 | 0.414128 |
| strong_2014_15_2pt_shooter_midrange_2pt | 8 | 0.414128 |
| strong_2014_15_2pt_shooter_midrange_2pt | 10 | 0.446182 |
| weak_2014_15_2pt_shooter_midrange_2pt | 0.500000 | 0.257384 |
| weak_2014_15_2pt_shooter_midrange_2pt | 1 | 0.257384 |
| weak_2014_15_2pt_shooter_midrange_2pt | 2 | 0.296552 |
| weak_2014_15_2pt_shooter_midrange_2pt | 3 | 0.336406 |
| weak_2014_15_2pt_shooter_midrange_2pt | 4 | 0.367423 |
| weak_2014_15_2pt_shooter_midrange_2pt | 5 | 0.414128 |
| weak_2014_15_2pt_shooter_midrange_2pt | 6 | 0.414128 |
| weak_2014_15_2pt_shooter_midrange_2pt | 7 | 0.446182 |
| weak_2014_15_2pt_shooter_midrange_2pt | 8 | 0.446182 |
| weak_2014_15_2pt_shooter_midrange_2pt | 10 | 0.446182 |
| median_2014_15_shooter_three_pt | 0.500000 | 0.296552 |
| median_2014_15_shooter_three_pt | 1 | 0.296552 |
| median_2014_15_shooter_three_pt | 2 | 0.324759 |
| median_2014_15_shooter_three_pt | 3 | 0.324759 |
| median_2014_15_shooter_three_pt | 4 | 0.324759 |
| median_2014_15_shooter_three_pt | 5 | 0.324759 |
| median_2014_15_shooter_three_pt | 6 | 0.332952 |
| median_2014_15_shooter_three_pt | 7 | 0.332952 |
| median_2014_15_shooter_three_pt | 8 | 0.332952 |
| median_2014_15_shooter_three_pt | 10 | 0.332952 |
| strong_2014_15_2pt_shooter_three_pt | 0.500000 | 0.296552 |
| strong_2014_15_2pt_shooter_three_pt | 1 | 0.296552 |
| strong_2014_15_2pt_shooter_three_pt | 2 | 0.324759 |
| strong_2014_15_2pt_shooter_three_pt | 3 | 0.332952 |
| strong_2014_15_2pt_shooter_three_pt | 4 | 0.336406 |
| strong_2014_15_2pt_shooter_three_pt | 5 | 0.367423 |
| strong_2014_15_2pt_shooter_three_pt | 6 | 0.367423 |
| strong_2014_15_2pt_shooter_three_pt | 7 | 0.367423 |
| strong_2014_15_2pt_shooter_three_pt | 8 | 0.367423 |
| strong_2014_15_2pt_shooter_three_pt | 10 | 0.367423 |
| weak_2014_15_2pt_shooter_three_pt | 0.500000 | 0.296296 |
| weak_2014_15_2pt_shooter_three_pt | 1 | 0.296296 |
| weak_2014_15_2pt_shooter_three_pt | 2 | 0.324759 |
| weak_2014_15_2pt_shooter_three_pt | 3 | 0.332952 |
| weak_2014_15_2pt_shooter_three_pt | 4 | 0.336406 |
| weak_2014_15_2pt_shooter_three_pt | 5 | 0.354369 |
| weak_2014_15_2pt_shooter_three_pt | 6 | 0.367423 |
| weak_2014_15_2pt_shooter_three_pt | 7 | 0.367423 |
| weak_2014_15_2pt_shooter_three_pt | 8 | 0.367423 |
| weak_2014_15_2pt_shooter_three_pt | 10 | 0.400744 |

## Adapter Alternatives
| adapter_strategy | closest_defender_distance | calibrated_probability |
| --- | --- | --- |
| A_current | 0.500000 | 0.621118 |
| A_current | 1 | 0.621118 |
| A_current | 2 | 0.636364 |
| A_current | 3 | 0.697509 |
| A_current | 4 | 0.762712 |
| A_current | 5 | 0.929461 |
| A_current | 6 | 0.929461 |
| A_current | 7 | 0.971429 |
| A_current | 8 | 0.999999 |
| A_current | 10 | 0.999999 |
| B_clip_numeric_p1_p99 | 0.500000 | 0.533195 |
| B_clip_numeric_p1_p99 | 1 | 0.533195 |
| B_clip_numeric_p1_p99 | 2 | 0.533195 |
| B_clip_numeric_p1_p99 | 3 | 0.655260 |
| B_clip_numeric_p1_p99 | 4 | 0.762712 |
| B_clip_numeric_p1_p99 | 5 | 0.924933 |
| B_clip_numeric_p1_p99 | 6 | 0.929461 |
| B_clip_numeric_p1_p99 | 7 | 0.971429 |
| B_clip_numeric_p1_p99 | 8 | 0.999999 |
| B_clip_numeric_p1_p99 | 10 | 0.999999 |
| C_rates_only_counts_training_median | 0.500000 | 0.621118 |
| C_rates_only_counts_training_median | 1 | 0.621118 |
| C_rates_only_counts_training_median | 2 | 0.636364 |
| C_rates_only_counts_training_median | 3 | 0.697509 |
| C_rates_only_counts_training_median | 4 | 0.787234 |
| C_rates_only_counts_training_median | 5 | 0.929461 |
| C_rates_only_counts_training_median | 6 | 0.971429 |
| C_rates_only_counts_training_median | 7 | 0.999999 |
| C_rates_only_counts_training_median | 8 | 0.999999 |
| C_rates_only_counts_training_median | 10 | 0.999999 |
| D_counts_season_to_date_median_scale | 0.500000 | 0.621118 |
| D_counts_season_to_date_median_scale | 1 | 0.621118 |
| D_counts_season_to_date_median_scale | 2 | 0.636364 |
| D_counts_season_to_date_median_scale | 3 | 0.697509 |
| D_counts_season_to_date_median_scale | 4 | 0.787234 |
| D_counts_season_to_date_median_scale | 5 | 0.929461 |
| D_counts_season_to_date_median_scale | 6 | 0.971429 |
| D_counts_season_to_date_median_scale | 7 | 0.999999 |
| D_counts_season_to_date_median_scale | 8 | 0.999999 |
| D_counts_season_to_date_median_scale | 10 | 0.999999 |
| E_training_population_shooter_profile | 0.500000 | 0.501946 |
| E_training_population_shooter_profile | 1 | 0.501946 |
| E_training_population_shooter_profile | 2 | 0.501946 |
| E_training_population_shooter_profile | 3 | 0.655260 |
| E_training_population_shooter_profile | 4 | 0.762712 |
| E_training_population_shooter_profile | 5 | 0.860000 |
| E_training_population_shooter_profile | 6 | 0.929461 |
| E_training_population_shooter_profile | 7 | 0.971429 |
| E_training_population_shooter_profile | 8 | 0.999999 |
| E_training_population_shooter_profile | 10 | 0.999999 |

## Defender Profile Sanity
| defenderId | defenderName | overallDfgPct | overallDfgA | overallNormalFgPct | overallPctPlusminus | raw_probability | calibrated_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 203110 | Draymond Green | 0.453000 | 808.000000 | 0.491000 | -0.038000 | 0.305948 | 0.296552 |
| 203497 | Rudy Gobert | 0.440000 | 1223.000000 | 0.489000 | -0.050000 | 0.305301 | 0.296552 |
| 202695 | Kawhi Leonard | 0.528000 | 631.000000 | 0.472000 | 0.056000 | 0.299725 | 0.296552 |
| none | No defender selected |  |  |  |  | 0.347507 | 0.336406 |

## Findings
- The 2014/15 shooter prior count features are season-to-date and current-game-excluded; the modern adapter supplies multi-season/canonical production profile counts. That is a semantic mismatch even when values land inside min/max.
- Jokic's modern near-rim row combines elite/shrunk shooting rates, non-cold-start flags, and modern count semantics with a sparse 2014/15 near-rim open-defender region.
- Isotonic calibration amplifies the high raw HGB region at 6 ft rather than creating it from a low raw score.
- Selected Draymond/Rudy/Kawhi defender profiles differ numerically, but the HGB places this Curry scenario in the same leaves after preprocessing; no-defender differs because missing defender PT fields trigger missing indicators/imputation.

## Support Diagnostic Recommendation
Add explicit experimental-only support diagnostics before showing modern experimental probabilities without qualification: `supportStatus` as `IN_DISTRIBUTION`, `LIMITED_SUPPORT`, or `OUT_OF_DISTRIBUTION`; `supportWarnings`; and `outOfRangeFeatures`. Attempt-count semantic mismatch should trigger at least `LIMITED_SUPPORT`, and outside p1-p99 plus sparse nearest-neighborhood support should trigger `OUT_OF_DISTRIBUTION`.
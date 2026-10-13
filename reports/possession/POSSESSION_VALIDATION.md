# Possession Validation

Total possessions: `1347164`
Overall possessions per team game: `87.06`
Overall PPP: `1.094`
Shot-to-possession match rate: `1.000000`

| Season | Type | Possessions | Games | Poss/team game | PPP | TOV rate | OREB continuation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-21 | playoffs | 14317 | 85 | 84.22 | 1.100 | 0.142 | 0.134 |
| 2020-21 | regular_season | 189773 | 1080 | 87.86 | 1.079 | 0.156 | 0.130 |
| 2021-22 | playoffs | 14408 | 87 | 82.80 | 1.064 | 0.164 | 0.134 |
| 2021-22 | regular_season | 213581 | 1230 | 86.82 | 1.077 | 0.157 | 0.135 |
| 2022-23 | playoffs | 14142 | 84 | 84.18 | 1.091 | 0.150 | 0.137 |
| 2022-23 | regular_season | 214937 | 1230 | 87.37 | 1.102 | 0.160 | 0.136 |
| 2023-24 | playoffs | 13348 | 82 | 81.39 | 1.083 | 0.147 | 0.137 |
| 2023-24 | regular_season | 214537 | 1230 | 87.21 | 1.104 | 0.154 | 0.136 |
| 2024-25 | playoffs | 13931 | 84 | 82.92 | 1.096 | 0.167 | 0.143 |
| 2024-25 | regular_season | 215348 | 1230 | 87.54 | 1.098 | 0.162 | 0.140 |
| 2025-26 | playoffs | 14035 | 85 | 82.56 | 1.080 | 0.173 | 0.149 |
| 2025-26 | regular_season | 214807 | 1230 | 87.32 | 1.108 | 0.165 | 0.142 |

Validation issues:
```json
{
  "negative_duration": 7698,
  "missing_offense_team": 0,
  "implausible_points_gt_8": 168,
  "ordering_errors": 2015
}
```

Sanity note: derived possession counts are in plausible NBA range, but this remains an empirical parser rather than official possession scoring.
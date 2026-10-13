# Score Anomaly Audit

Generated: 2026-10-08T15:59:37Z

## Scope
This audit analyzes score decrease/correction events and made-field-goal post-score reconciliation exceptions without modifying the canonical dataset or pipeline.

## Anomaly Taxonomy
| Category                                | Score-decrease events |
| --------------------------------------- | --------------------- |
| replay_or_review_context                | 725                   |
| free_throw_score_correction             | 628                   |
| made_fg_with_score_correction           | 512                   |
| non_scoring_event_with_score_correction | 45                    |

## Made-FG Reconciliation Exceptions
| Category                        | Made-FG exceptions |
| ------------------------------- | ------------------ |
| made_fg_score_snapshot_mismatch | 1753               |
| made_fg_with_score_correction   | 512                |

By season/type:
| Season  | Season type    | Made-FG exceptions |
| ------- | -------------- | ------------------ |
| 2020-21 | playoffs       | 37                 |
| 2020-21 | regular_season | 410                |
| 2021-22 | playoffs       | 27                 |
| 2021-22 | regular_season | 384                |
| 2022-23 | playoffs       | 21                 |
| 2022-23 | regular_season | 288                |
| 2023-24 | playoffs       | 14                 |
| 2023-24 | regular_season | 367                |
| 2024-25 | playoffs       | 10                 |
| 2024-25 | regular_season | 319                |
| 2025-26 | playoffs       | 17                 |
| 2025-26 | regular_season | 371                |

By shot value:
| Shot value | Made-FG exceptions |
| ---------- | ------------------ |
| 2          | 1660               |
| 3          | 605                |

By period:
| Period | Made-FG exceptions |
| ------ | ------------------ |
| 1      | 498                |
| 2      | 589                |
| 3      | 593                |
| 4      | 575                |
| 5      | 10                 |

By Shot Detail action type:
| Action type                     | Made-FG exceptions |
| ------------------------------- | ------------------ |
| Jump Shot                       | 485                |
| Driving Layup Shot              | 252                |
| Pullup Jump shot                | 194                |
| Tip Layup Shot                  | 115                |
| Putback Layup Shot              | 105                |
| Layup Shot                      | 103                |
| Running Layup Shot              | 99                 |
| Driving Floating Jump Shot      | 96                 |
| Step Back Jump shot             | 71                 |
| Cutting Layup Shot              | 69                 |
| Running Dunk Shot               | 53                 |
| Cutting Dunk Shot               | 51                 |
| Driving Finger Roll Layup Shot  | 49                 |
| Running Jump Shot               | 43                 |
| Floating Jump shot              | 41                 |
| Fadeaway Jump Shot              | 40                 |
| Driving Dunk Shot               | 38                 |
| Driving Floating Bank Jump Shot | 33                 |
| Tip Dunk Shot                   | 31                 |
| Dunk Shot                       | 30                 |
| Alley Oop Dunk Shot             | 27                 |
| Turnaround Jump Shot            | 25                 |
| Turnaround Fadeaway shot        | 23                 |
| Reverse Layup Shot              | 21                 |
| Driving Reverse Layup Shot      | 20                 |

## Representative Score-Decrease Examples
| Season  | Type           | Game     | Period | Clock       | Action         | Before  | Reported after | Description                                          | Category                      |
| ------- | -------------- | -------- | ------ | ----------- | -------------- | ------- | -------------- | ---------------------------------------------------- | ----------------------------- |
| 2020-21 | regular_season | 22000002 | 2      | PT00M52.60S | Made Shot      | 54-56   | 53-56          | James 3' Running Layup (9 PTS) (Schroder 3 AST)      | made_fg_with_score_correction |
| 2020-21 | regular_season | 22000003 | 3      | PT10M49.00S | Made Shot      | 66-68   | 64-66          | Middleton 26' 3PT Jump Shot (11 PTS) (Holiday 2 AST) | made_fg_with_score_correction |
| 2020-21 | regular_season | 22000012 | 1      | PT04M57.00S | Made Shot      | 20-22   | 18-15          | Fultz 10' Jump Shot (4 PTS)                          | made_fg_with_score_correction |
| 2020-21 | regular_season | 22000012 | 1      | PT04M12.00S | Free Throw     | 27-29   | 18-19          | Adebayo Free Throw 1 of 2 (5 PTS)                    | free_throw_score_correction   |
| 2020-21 | regular_season | 22000013 | 1      | PT08M35.00S | Made Shot      | 14-11   | 10-11          | Westbrook 19' Jump Shot (4 PTS)                      | made_fg_with_score_correction |
| 2020-21 | regular_season | 22000015 | 4      | PT06M09.00S | Free Throw     | 90-122  | 88-119         | Williams Free Throw 2 of 2 (12 PTS)                  | free_throw_score_correction   |
| 2020-21 | regular_season | 22000022 | 4      | PT01M17.00S | Instant Replay | 95-102  | 91-102         | Instant Replay4th Period (9:21 PM EST)               | replay_or_review_context      |
| 2020-21 | regular_season | 22000023 | 1      | PT08M23.00S | Free Throw     | 21-13   | 9-9            | Grant Free Throw 2 of 2 (6 PTS)                      | free_throw_score_correction   |
| 2020-21 | regular_season | 22000031 | 1      | PT03M40.00S | Instant Replay | 13-43   | 5-22           | Instant Replay1st Period (3:59 PM EST)               | replay_or_review_context      |
| 2020-21 | regular_season | 22000033 | 2      | PT09M54.00S | Free Throw     | 113-120 | 35-32          | Birch Free Throw 1 of 2 (1 PTS)                      | free_throw_score_correction   |

## Representative Made-FG Exception Examples
| Season  | Type           | Game     | Event | Player            | Period | Clock       | Value | Before | Reported after | Description                                         | Category                        |
| ------- | -------------- | -------- | ----- | ----------------- | ------ | ----------- | ----- | ------ | -------------- | --------------------------------------------------- | ------------------------------- |
| 2020-21 | regular_season | 22000057 | 508   | Solomon Hill      | 3      | PT01M16.00S | 3     | 94-100 | 95-103         | S. Hill 24' 3PT Jump Shot (9 PTS) (Rondo 5 AST)     | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000171 | 297   | Bogdan Bogdanovic | 2      | PT01M40.00S | 2     | 58-48  | 60-50          | Bogdanovic 2' Alley Oop Layup (7 PTS) (Young 8 AST) | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000205 | 479   | John Collins      | 3      | PT04M30.00S | 2     | 73-60  | 75-61          | Collins 1' Alley Oop Dunk (15 PTS) (Huerter 6 AST)  | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000254 | 341   | De'Andre Hunter   | 3      | PT11M06.00S | 2     | 66-50  | 68-52          | Hunter 19' Pullup Jump Shot (16 PTS)                | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000254 | 356   | John Collins      | 3      | PT10M22.00S | 3     | 68-50  | 68-55          | Collins 27' 3PT Jump Shot (19 PTS) (Hunter 4 AST)   | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000254 | 557   | John Collins      | 4      | PT07M16.00S | 2     | 110-94 | 111-96         | Collins 16' Turnaround Jump Shot (23 PTS)           | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000254 | 579   | De'Andre Hunter   | 4      | PT06M13.00S | 2     | 111-94 | 115-98         | Hunter 2' Cutting Dunk Shot (25 PTS) (Hill 2 AST)   | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000268 | 153   | Rajon Rondo       | 1      | PT00M00.00S | 2     | 18-19  | 22-21          | Rondo 15' Pullup Jump Shot (2 PTS)                  | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000451 | 479   | Danilo Gallinari  | 3      | PT02M01.00S | 3     | 89-66  | 91-69          | Gallinari 29' 3PT Jump Shot (3 PTS) (Young 9 AST)   | made_fg_score_snapshot_mismatch |
| 2020-21 | regular_season | 22000451 | 494   | Trae Young        | 3      | PT00M31.00S | 2     | 91-66  | 96-71          | Young 10' Jump Shot (22 PTS) (Gallinari 1 AST)      | made_fg_score_snapshot_mismatch |

## Canonical Score Impact
The canonical dataset stores pre-shot score from the ordered running score before each PBP event. The anomaly classes are source-score corrections or transient/stale PBP score snapshots, not duplicate canonical shot rows or broken shot matching.

| Class                                         | Count |
| --------------------------------------------- | ----- |
| A_source_pbp_correction_behavior              | 1910  |
| B_harmless_transient_scoreboard_inconsistency | 1753  |
| C_ambiguous                                   | 0     |
| D_pipeline_error                              | 0     |

## Clutch Impact
| Metric                                             | Count |
| -------------------------------------------------- | ----- |
| anomalous canonical shots                          | 2265  |
| anomalous clutch shots                             | 72    |
| score diff may differ under immediate PBP snapshot | 2097  |
| uncertain clutch classifications                   | 21    |

## Remediation
No canonical pipeline change is recommended. The current validation should continue reporting scoreboard correction behavior as acceptable source behavior while preserving the verified pre-shot score fields. No `score_context_quality` column is introduced because this audit found no demonstrable pipeline errors; clutch-sensitive rows are documented here.

## Final Assessment
Score context is sufficiently reliable for modeling, with documented source-level scoreboard correction behavior.
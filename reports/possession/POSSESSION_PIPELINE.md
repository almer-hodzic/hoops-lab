# Possession Pipeline

Rules used:
- A possession starts at the first team event after no active possession.
- Made field goals end a possession.
- Turnovers end a possession.
- Defensive rebounds after missed shots end the prior possession and start the rebound team's possession.
- Offensive rebounds continue the current possession.
- Period end closes any open possession.
- Free throws are kept inside the active possession; free-throw-only endings are not asserted unless period/turnover/made-FG logic closes the possession.
- Jump balls and technical edge cases are handled deterministically by team-control sequence, and unusual cases remain in the dataset for validation.

PBP fields used: `gameId`, `actionNumber`, `clock`, `period`, `teamId`, `location`, `actionType`, `subType`, `isFieldGoal`, `scoreHome`, `scoreAway`, `description`.
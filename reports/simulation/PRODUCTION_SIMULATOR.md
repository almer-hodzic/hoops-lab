# Production Single-Game Simulator V1

Approved scope: **B. ACCEPTABLE FOR UNCERTAINTY-AWARE V1 WITH DOCUMENTED LIMITATIONS**.
Research baseline: `a7491448f1f298bc39a8b475ce08db9fe85e6c85`.
Frozen production version: `hoopslab-single-game-v1.0.0`; response schema version 1.

This is an uncertainty-aware statistical model, not a claim of objective historical truth.
No models were retrained, no calibration parameters were changed, and no holdout benchmark was rerun.
No Simulator UI, season simulator, or application redesign is included.

## Architecture And Startup

`SimulationEndpoints.cs` adds two thin .NET routes. They forward to the dedicated loopback-only
Python service in `scripts/simulation/simulation_service.py`, independently of Shot Lab's existing
inference service. All roster and simulation business rules live in Python, not in the proxy.
The .NET proxy applies only transport limits, cancellation, and unavailable/timeout error handling.

From the repository root, start the Python simulator:

```powershell
python scripts/simulation/simulation_service.py --port 8012
```

Then run the existing .NET Web application normally:

```powershell
dotnet run --project nba-clutch-agent/AiAgents.NbaClutchAgent.Web
```

`Simulation__BaseUrl` overrides the .NET upstream, default `http://127.0.0.1:8012`.
The Python service is an internal service, not an internet-facing HTTP deployment server.
It accepts four concurrent simulations, returns 503 when busy, limits requests to 64 KiB,
and applies a 10-second socket timeout. The .NET upstream timeout is 30 seconds.
Malformed requests return JSON 400, oversized bodies 413, unsupported content types 415,
and unavailable upstream service 503 (timeout 504). Unexpected engine failures return JSON 500;
they never silently become tied scores, substituted data, or an invented winner.

Canonical CSVs and production artifacts load and are verified once per process. Requests use
indexed in-memory PlayerSeason profiles and league environments. There is no network acquisition,
raw-cache access, research joblib loading, fitting, or automatic artifact export at startup.
Deploy the repository's canonical `data/historical/processed` data and committed production package
alongside the Python scripts; moving only the .NET binary is not a complete deployment.

## Frozen Artifacts And Provenance

All production files are committed under `data/simulation/production_v1/`, outside ignored research
`artifacts/`. The numerical package is approximately 1.64 MiB, plus small reference/verification JSON:

| File | Runtime purpose | Size |
| --- | --- | ---: |
| `config.json` | Version, fitted imputer/scaler/linear coefficients, rotation/usage config, domain references, thresholds, home/OT parameters, source hashes | 23,045 bytes |
| `defensive_profiles.csv` | Verified canonical-keyed supplemental source evidence absent from canonical PlayerSeason CSV | 1,279,461 bytes |
| `joint_residuals.csv` | All 6,149 centered joint roster-engine residual vectors; no outcomes/identities | 398,837 bytes |
| `overtime_residuals.csv` | 300 centered paired one-OT efficiency increments | 12,690 bytes |
| `reference_cases.json` | Six frozen research-reference inputs, intermediate features, strengths, expectations and seeded summaries | 52,562 bytes |
| `verification.json` | Actual API smoke-test and performance measurements; not a runtime dependency | Small JSON |

`export_production_simulator.py` reads only existing approved fitted objects:
`targeted_projection_frozen.joblib`, `defensive_continuity_v1.joblib`, and `game_final_v1.joblib`.
It asserts the approved final artifact SHA256 and selected configuration, then exports coefficients,
training-only profile priors, and the exact residual samples. No fit/calibration/evaluation is called.
Supplemental defensive fields are copied from the existing verified source export, with original
source minutes retained to check identity. The large original game/player raw datasets are not included.

The config fingerprints canonical CSVs, source artifacts, residual/supplemental CSVs, and the actual
approved numerical functions. Startup fails on mismatched data, artifact content, or kernel source.
Text SHA256 hashes normalize CRLF to LF to survive Git's Windows/Unix checkout settings; binary
source hashes remain byte-exact.
Runtime uses the existing research functions for rotation, usage normalization, aggregation, linear
anchor predictions, defensive prediction/uncertainty, extrapolation, game expectation, joint draws,
and regulation scoring. Their original files remain unchanged. `FrozenPipeline` is just read-only
imputer/scaler/linear inference from exported fitted values, checked against original sklearn objects.

The production draw adapter retains per-draw final/regulation scores and OT periods, which the
research summary API did not expose. It uses the shared regulation/joint-draw functions and exactly
the research tie/OT sequence, not a new noise or calibration formula. Seeded research parity tests
cover this adapter. Kernel fingerprints prevent an unnoticed research change from changing runtime.

Re-export commands are explicit development tools, never runtime dependencies:

```powershell
python scripts/simulation/export_production_simulator.py
python scripts/simulation/export_production_references.py
```

The latter runs inference on declared canonical fixtures only, not a historical outcome benchmark.
It uses the original fitted research objects and independent research profile construction, not the
production predictions as reference values. Keep the artifact/version relationship intentional when
making any future change; V1 calibration is closed.

## API Contracts

`GET /api/simulation/info` returns the model/schema version, artifact SHA256, supported target seasons,
venue options, draw limits, roster limits, starter slots, ironman size, calibration end, and disclaimer.

`POST /api/simulation/game` accepts `application/json`:

```json
{
  "teamA": {
    "name": "2015-16 Cavaliers",
    "starters": [
      {"slot": "PG", "playerId": "br:irvinky01", "season": "2015-16"},
      {"slot": "SG", "playerId": "br:smithjr01", "season": "2015-16"},
      {"slot": "SF", "playerId": "br:jamesle01", "season": "2015-16"},
      {"slot": "PF", "playerId": "br:loveke01", "season": "2015-16"},
      {"slot": "C", "playerId": "br:thomptr01", "season": "2015-16"}
    ],
    "bench": [
      {"playerId": "br:dellama01", "season": "2015-16"},
      {"playerId": "br:shumpim01", "season": "2015-16"},
      {"playerId": "br:mozgoti01", "season": "2015-16"}
    ]
  },
  "teamB": {
    "name": "2015-16 Warriors",
    "starters": [
      {"slot": "PG", "playerId": "br:curryst01", "season": "2015-16"},
      {"slot": "SG", "playerId": "br:thompkl01", "season": "2015-16"},
      {"slot": "SF", "playerId": "br:barneha02", "season": "2015-16"},
      {"slot": "PF", "playerId": "br:greendr01", "season": "2015-16"},
      {"slot": "C", "playerId": "br:bogutan01", "season": "2015-16"}
    ],
    "bench": [
      {"playerId": "br:iguodan01", "season": "2015-16"},
      {"playerId": "br:livinsh01", "season": "2015-16"},
      {"playerId": "br:speigma01", "season": "2015-16"}
    ]
  },
  "targetSeason": "2016-17",
  "venue": "neutral",
  "simulationCount": 10000,
  "seed": 12345,
  "ironmanMode": false
}
```

The above is a complete eight-versus-eight request. Additional runnable `request` objects in
`reference_cases.json` include a named twelve-player Warriors roster versus the named eight-player
Cavaliers roster. The five-player case is separate, not the primary test.

The existing Team Builder identity format is also accepted unchanged: `teamName`, `schemaVersion: 1`,
`starters` keyed by PG/SG/SF/PF/C, and a bench array with optional sequential BENCH1...BENCH7 slots.
Entry display labels (`playerName`, `team`, `position`) may be passed but are ignored for inference.
Canonical `playerId` + `season` alone resolve skills. Caller-supplied statistics and unknown fields
are rejected. Canonical traded-player season totals remain totals; their source context uses the
same minutes-weighted split mapping as the approved research, not the last franchise's context.

Normal mode requires 8-12 named players per team, exactly five unique PG/SG/SF/PF/C starter slots,
and no duplicate canonical player even across different seasons within that team. The same player
may appear on both opposing teams for counterfactual/mirror comparisons. Each team projects 240
minutes: normal starter bounds 20-38, bench bounds 4-28, using approved balanced-prior water filling.
`ironmanMode: true` applies to both teams and requires exactly five each, no bench, 48 minutes each.
It always returns HIGH uncertainty and explicit nonstandard-hypothetical/no-fatigue warnings.
Five-player teams never silently enter normal mode. Unknown identities/seasons, malformed slots,
and rosters above twelve are rejected. Draw count defaults to 10,000; integer range 1,000-100,000.
Seed is optional/null or an integer in 0...9007199254740991; booleans are not integers in this contract.

Response structure:

| Field | Meaning |
| --- | --- |
| `modelVersion`, `targetSeason`, `venue`, `simulationCount`, `seed`, `ironmanMode` | Versioned run identity; returned seed permits replay |
| `teamA`, `teamB` | Name, target-era projected ORtg/DRtg/pace, deterministic expected efficiency/points, win probability |
| Team `rotation` | Canonical ID/name/season/slot, projected minutes, normalized usage%, possession share |
| Team `uncertainty`, `uncertaintyReasons`, `defensiveCoverage`, `extrapolation` | Uncertainty category/reasons and compact domain-distance evidence |
| Team `diagnostics`, `rotationDepth`, `projectedMinutes` | Factual intermediate indicators, not arbitrary overall ratings |
| `leagueEnvironment` | Canonical target-season ORtg, pace, TS% |
| `game.expectedPace`, `expectedScore`, `expectedMargin`, `expectedTotal` | Deterministic regulation/pre-simulation expectations |
| `game.simulatedMeanScore`, `medianScore` | Monte Carlo final-score summaries including OT; distinct from expectations |
| `game.scoreIntervals`, `marginIntervals` | Equal-tailed 50/80/90/95 final-score and A-minus-B margin intervals |
| `game.regulationMarginIntervals` | Corresponding regulation-only margin intervals |
| `game.scoreDistributions` | Exact compact marginal integer-score/margin histograms as `{value,count}` arrays; counts sum to draws |
| `game.overtimeProbability`, `overtimePeriodDistribution`, `finalTies` | Regulation tie/OT incidence, OT period counts, zero final ties |
| `game.sampleOutcome` | One actual simulated untied final score, its regulation score and OT period count |
| `metadata` | Source commit/hash, NumPy version, input mode, frozen model choices, MC probability SE, limitations |

Score and margin interval bounds are `[lower, upper]`. Marginal medians/interval endpoints need not
form an actual paired game; `sampleOutcome` does. Histogram bins are marginal, not a returned joint
draw table. A win probability and its complement sum to one; no draw probability remains after OT.

## Selected Profiles And Target Era

Supported target environments are all 47 canonical LeagueSeason rows: **1979-80 through 2025-26**.
There is no invented neutral era. Neutral is only a venue option.

Chosen historical versions are held at their selected canonical age/skill, as in the approved
custom-lineup adapter; they are not aged to the target era. History MPG uses the approved .5-decay
weighting of positive-minute seasons within the selected three-year window, never later seasons.
Availability comes from the selected season's games/schedule. Null skills use frozen <=2013
position/pooled priors, while defensive availability is assessed before imputation. Unverified
defensive supplemental identity is treated as missing, not silently trusted.

The approved aggregate uses minutes weights, role-shrunk usage (alpha=1, shrink=.10), usage-weighted
relative TS/3PAr/FTr, source pace relative to its league, and other transferable profile indicators.
For selected versions there is no target-season performance fetch. Target league ORtg/pace become
the additive anchors, preserving roster strength deviations exactly as approved era transport.
This is descriptive counterfactual historical simulation, not a claim that a historical target's
environment or the frozen 2019 calibration was known prospectively in 1980.

Custom rosters have returning_minutes_share=0 and retained_defensive_context=0 even if every member
played for the same actual franchise. Source defensive context and origin diversity are uncertainty
evidence only; they do not transfer the old team's defensive quality into a new custom roster.
The approved DRtg model is retained unchanged, no new tail correction or gamma adjustment.

## Expectation, Venue And Monte Carlo

Let L be target league ORtg, P the mean of projected team paces. Neutral expectations:
`A_eff = L + (A_ORtg-L) + (B_DRtg-L)`; vice versa for B; points = `P*eff/100`.

At either home venue, orient the actual home team as research home, adding the approved common
offset 0.2937025939580691 per 100 to both efficiencies and half the prospective home-margin effect
2.694038660326973 to home / subtracting half from away. These are the frozen trailing-five-year
parameters ending 2018-19, not a hardcoded +3 points. Neutral removes both venue adjustments.
The final orientation is mapped back to Team A/Team B in the response.

Each simulation jointly resamples one centered `(pace, home_eff, away_eff)` vector from the approved
roster-engine empirical sampler, scales by target league pace/ORtg, and uses the approved covariance
product-mean correction. Regulation scores are rounded and floored at zero by the original kernel.
The calibrated close-game tie atom (extra=0.16227381103843122, thinning=0) applies only to the
approved 1-3-point rounded margin band. Every remaining tie receives paired empirical five-minute
OT increments scaled to the target environment. Repeat up to 20 OTs, then fail loudly if still tied.
There is no arbitrary independent noise, random-winner tie break, or uncertainty variance multiplier.

Default 10,000 draws gives worst-case Monte Carlo probability standard error 0.5 percentage points.
Per-request NumPy RNG instances prevent concurrent requests sharing random state. Same payload/seed,
artifact, data and software environment gives identical output. No seed produces a fresh random
53-bit seed, returned for replay. NumPy version and artifact hash are returned; cross-version RNG/
floating-point identity is not promised. Seeded draws agree with original research summary values;
complementary probability subtraction may differ at the last floating-point bit (<1e-15).

## Uncertainty And Explainability

LOW/MODERATE/HIGH retain the approved evidence/coverage semantics, not confidence probabilities.
HIGH is triggered by low continuity (<.35), weighted missing defensive evidence >.20, diverse
source contexts beyond frozen training quantiles, or strong extrapolation (>1 IQR beyond range
or >15% of features outside). The original moderate criteria (.65 continuity, .05 missingness,
source-context p75 or any outside feature) remain unchanged. Offense/pace high domain risk also
raises the team warning; ironman always forces HIGH. All submitted custom teams currently lack
certified continuity and therefore report HIGH, including plausible same-franchise selections.
This is intentional: the API cannot certify lineup/scheme continuity from chosen identities alone.

Labels never change strengths, means or noise. Outcome intervals reflect calibrated game noise and
observed roster-engine forecast error, not every possible synthetic-roster interaction uncertainty.
Diagnostics expose shooting efficiency relative to source league, spacing/3P volume, FTr,
playmaking/turnover rates, offensive and defensive rebound rates, DBPM evidence, usage concentration/
compression, rotation HHI/depth, missingness, context dispersion and zero inherited continuity.
No LLM prose, chemistry score, overall rating or invented offense/defense 0-100 score is generated.

## Parity And Verification

All six original-research reference cases pass. Strengths, intermediate features, minutes and
deterministic expectations agree within **1e-9 absolute**. Same-seed empirical summaries, means,
margin intervals and OT incidence agree (probability complement tolerance 1e-15):

| Case | Target / venue | A ORtg / DRtg / pace | Expected A-B regulation score |
| --- | --- | --- | --- |
| A: 12-player Warriors vs 8-player Cavaliers | 2016-17 / A home | 115.840 / 107.616 / 99.882 | 112.395 - 105.959 |
| B: 8-player mixed-era custom vs Warriors | 1990-91 / B home | 121.262 / 101.554 / 96.260 | 116.854 - 108.171 |
| C: Cavaliers vs Warriors | 2016-17 / neutral | 112.459 / 108.190 / 93.674 | 106.979 - 110.807 |
| D: Same teams | 2016-17 / A home | 112.459 / 108.190 / 93.674 | 108.567 - 109.788 |
| E: Fixed-seed early-era transport | 1979-80 / neutral | 112.380 / 104.155 / 106.556 | 114.869 - 110.777 |
| F: Curry/Jordan/LeBron/Duncan/Shaq mirror ironman | 2016-17 / neutral | 124.959 / 101.706 / 93.616 | 109.654 - 109.654 |

F uses Curry 2015-16, Jordan 1990-91, LeBron 2012-13, Duncan 2002-03 and Shaq 1999-00,
exactly 48 minutes each with HIGH uncertainty. Primary full-roster validation uses cases A-E.

Verified commands/results:

```text
python -m unittest discover -s scripts/simulation -p test_*.py
96 passed: original 63 research tests + 33 production tests
python scripts/simulation/test_production_simulation_api.py
PASS: real .NET -> Python; six cases; identity/roster/draw validation; 413/415; upstream-down 503
python scripts/historical/validate_full_historical_dataset.py
PASS: 20,526 player seasons, 3,818 players, 1,314 teams, 47 leagues; zero errors
python scripts/historical/test_historical_player_api.py
PASS: Historical Players, canonical seasons, missing data and Compare
node scripts/historical/test_team_builder_logic.mjs
PASS
node --check nba-clutch-agent/AiAgents.NbaClutchAgent.Web/wwwroot/app.js
node --check nba-clutch-agent/AiAgents.NbaClutchAgent.Web/wwwroot/teamBuilder.js
PASS
python scripts/inference/test_inference.py
11 passed: existing Shot Lab production inference regression tests
dotnet build nba-clutch-agent/AiAgents.NbaClutchAgent.sln --no-restore
PASS: zero warnings, zero errors
```

Production tests also cover all 47 target eras, both venue orientations, seed replay/random behavior,
eight/twelve-player rotations and usage conservation, malformed input, concurrent determinism,
traded-season context, interval nesting, all final margins nonzero, histogram counts, HIGH-label
invariance, checkout-line-ending portability/tampering rejection, and no per-request CSV parsing
or runtime model loading/fitting.
The existing frontend, Team Builder module, historical datasets, Shot Lab inference/model files,
original simulation scripts/artifacts and old benchmark reports remain byte-for-byte unchanged.
No new dataset downloads, research fits, or outcome evaluations were performed.

## Performance

Actual end-to-end local .NET endpoint, 30 warm requests, 12 vs 8 players, 10,000 draws:
median **253.5 ms**, p95 **645.6 ms**. One 100,000-draw request: **730.3 ms**.
Python cold startup including verification/imports/CSV indexing: **11.35 s**, outside request latency.
Python working set: **320.910 MiB before**, **321.223 MiB after 30**, **325.355 MiB after max draws**;
observed range 320.848-325.355 MiB. Warm-request working-set increase was 0.313 MiB in this short run;
this is not an unlimited-duration leak or load-test guarantee. NumPy native threads are capped at one
by the service. Measurements and acquisition-free rerunnable test are in `verification.json` and
`test_production_simulation_api.py`. This run includes local host scheduling variation, not an SLA.

## Known V1 Limitations And Next Step

- Defense remains imperfect, approved only for uncertainty-aware V1; no further tuning was done.
- Calibration ends 2018-19; the accepted 2025-26 game holdout does not refresh coefficients or noise.
- Prior holdout showed central interval undercoverage, about one-point home bias and OT overprediction;
  accepted 95% coverage does not establish perfect calibration or precise extreme probabilities.
- Roster-engine noise was calibrated using first-game roster proxies and lagged profiles, not certified
  preseason availability or arbitrary all-time superteams. Selected historical versions are a documented
  application adapter, not a newly calibrated custom-roster model.
- Mixed-era, neutral, unusual rotations and ironman outputs extrapolate; no injury, fatigue, scheme,
  matchup assignment, on/off interaction, coaching or chemistry model exists.
- Canonical historical league environments are descriptive full-season contexts, not pregame forecasts.
  Frozen 2019 venue/noise calibration is reused across target eras, not independently retuned by era.
- Empirical margin/score distributions retain source home/away residual structure even for neutral
  games; no separately calibrated neutral or exact mirror-symmetry model is claimed.
- Historical possession estimation and time-proportional overtime scaling remain approximate.
- Probability MC error and statistical outcome intervals are not calibrated synthetic-roster confidence.
- Service readiness requires matching committed artifacts/canonical data; dataset upgrades deliberately
  fail version checks until explicitly validated and versioned. Research joblib/cache files are not needed.

Recommended next step: review/approve this production API and its frozen parity evidence, then merge
this feature through the normal integration workflow. Only afterward authorize the separate Simulator
UI task to consume these contracts and surface HIGH/extrapolation limitations explicitly. No merge or
push is part of this implementation task.

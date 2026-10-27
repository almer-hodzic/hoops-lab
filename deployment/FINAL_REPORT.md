# HoopsLab public V1 deployment review — 2026-10-27

Prepared locally. No commit, push, or deployment. Existing V2, MovieShop, and Portfolio source were not modified.

1. **Workspace:** `C:/Users/Lenovo/Desktop/hoops-lab-deploy`.
2. **Base:** HEAD remains exactly `86a1821e8874d49a72f5300a8f6d7ace1150fe33`; branch `deployment/railway-v1`. A single-branch public main clone was clean before edits. No V2 branch was fetched/merged; no unpublished V2 source/artifacts were copied.
3. **Runtime:** Web and its three library dependencies target .NET 10, running ASP.NET Core/.NET 10.0.14. Release build/publish: zero warnings/errors. Removed unused Swagger/OpenAPI package references after restore reported vulnerable Microsoft.OpenApi 2.0.0. Console stays exactly at original .NET 9 and is not packaged; its original public source is incomplete (`Infrastructure.Dataset` missing). Full solution build therefore fails independently of deployment Web behavior. See Microsoft migration guidance: https://learn.microsoft.com/en-us/aspnet/core/migration/90-to-100 . No installed host SDK was changed.
4. **Artifact provenance:** PASS. All 24 source archives match public V1 hashes; all 1,368,990 canonical rows reconstructed exactly; selected V1 model rebuilt from 843,622 training rows matches frozen predictions on all canonical rows within 1e-12; calibration, profiles, metadata, columns and policy matched; all 219,158 analysis CSV rows reproduced. Existing bundle object serialization is not byte-identical to retraining; full numerical parity is established. Detailed evidence/generators: `PROVENANCE.md`.
5. **Artifact SHA-256:** Full table below and `runtime-artifacts.json`. Binary hashes checked exactly; CSV/JSON have additional LF-normalized startup hashes to survive cross-platform Git checkouts. Source and destination copies were checked. Legacy shots.tsv checksum was inspected, but its generation lineage could not be established; it is excluded along with its unused legacy consumers.
6. **Python:** CPython 3.12.10. Exact pins: numpy 2.4.6; pandas 3.0.3; scikit-learn 1.9.0; scipy 1.17.1; joblib 1.5.3; threadpoolctl 3.6.0; pyarrow 25.0.1. Import/dependency closure: lxml 6.1.1, narwhals 2.22.1, python-dateutil 2.9.0.post0, six 1.17.0. All versions come from the verified local environment. Linux amd64 CPython 3.12 wheel hashes pinned; pip dependency check passes. No host package installations/upgrades.
7. **Docker:** One Linux amd64 image, multistage .NET build plus official portable ASP.NET runtime and Python. Base images and Dockerfile frontend pinned by digest; .NET lockfiles, Python wheel hashes, native package versions pinned. Final UID/GID 10001. Only public .NET port exposed. Image ID `sha256:5709c12749e3362aad434fe190b498b30e38d5cb4712355462385c107107c385`; local image size 1,172,494,169 bytes (~1.09 GiB). Package repository index timestamps are not frozen; no byte-for-byte OS-layer claim.
8. **Supervisor:** Python PID 1 validates required file checksums, starts both Python services, waits up to 180 seconds for both, then launches .NET. Process exit fails container; SIGTERM/SIGINT forwards to process groups, waits 15 seconds, kills/reaps stragglers. Child-failure and graceful-stop checks passed.
9. **Runtime data:** Three inference files; verified analysis CSV; public V1 historical processed CSVs/metadata; public V1 production simulation config/arrays/reference JSON; original runtime Python modules; published Web/wwwroot. Layout `/app/{scripts,data,artifacts,web,deployment}`. No raw research archives, training joblibs, legacy shots.tsv, writable Q-table state, database, secrets, or unpublished V2 work.
10. **Endpoint safety:** `/api/run`, `/api/agent/*`, `/api/baseline` blocked for every HTTP verb; legacy runner/worker not registered. API concurrency 4, shared API budget 240/min, expensive POST budget 20/min, 64 KiB body cap. 404/429/413 verified. Python services listen only on 127.0.0.1:8011/8012; container publishes only .NET. Upstream 5xx and transport faults return generic messages; raw exception sanitizing checked.
11. **Readiness:** `/ping` .NET liveness; `/ready` checks required file paths and both Python `/ping` responses. Full SHA verification before startup, plus unchanged simulator kernel/data integrity validation. Docker continuously checks readiness; Railway uses it as a deployment gate (https://docs.railway.com/deployments/healthchecks).
12. **Tests:** Linux inference 11/11 PASS; production simulator 33/33 PASS; original historical API assertions PASS against real container (only process-launch lifecycle adapted); Team Builder Node logic, JS syntax, full historical validation and prototype validation PASS. Linux broader simulation/research suite: 96 run, 91 passed, 4 skipped, 1 error due to missing raw 1989 advanced HTML cache. Host suite: 93 passed/3 errors (two missing research fixtures and Windows HTTP connection abort). Defense scripts fail on absent ignored research parquet inputs. Original simulation API harness cannot run against published layout; it hardcodes net9 debug paths and legacy behavior, and is superseded by the new full-container check. There are no .NET unit-test projects; successful dotnet test is a no-test invocation. Exact outputs in `test-results.json`.
13. **Container:** Build PASS; running and healthy at `http://127.0.0.1:18080`. Local review only. Docker container `hoopslab-v1-review`; one CPU, 2 GiB memory limit. Provider PORT=9123 was separately verified; temporary test container removed.
14. **Features:** Homepage/assets, Player Explorer, historical detail, Compare, Shot Lab profiles/model/inference, historical decision pagination, and single-game simulation PASS by HTTP/API. Existing Team Builder JS logic PASS. Browser rendered QA and browser-only localStorage interactions remain unverified: Edge app approval timed out. No UI/source design changes.
15. **Idle RAM:** 713.06 MiB after feature warmup.
16. **One inference RAM:** sampled peak 714.13 MiB; after 713.92 MiB; request 0.050 seconds.
17. **One simulation RAM:** sampled peak 714.21 MiB; after 713.44 MiB; 10,000 draws; request 0.133 seconds.
18. **Startup:** 11.515 seconds cold-process startup including container creation to readiness; clean restart 9.531 seconds. Excludes image download/build. Memory is Linux cgroup memory.current including charged cache/sampler overhead, sampled every 20 ms for 3 seconds during one operation. No destructive stress/load test. The tested 2 GiB limit gives useful headroom; 512 MiB is below observed idle use. 1 GiB/concurrent traffic capacity was not established.
19. **Railway variables:** provider PORT (default local 8080), ASPNETCORE_ENVIRONMENT=Production. Supervisor derives ASPNETCORE_URLS=http://0.0.0.0:$PORT. Optional ProductionInference__BaseUrl must equal http://127.0.0.1:8011; optional Simulation__BaseUrl must equal http://127.0.0.1:8012. APP_ROOT set internally. No secrets/database. Leave start-command override unset. `/ready`, 300-second startup health timeout, on-failure restart in railway.json. Consider 20-second deployment drain. No Railway service/account was changed.
20. **Files:** 8 modified tracked files and 23 new files, listed below. Review exports are ignored local artifacts.
21. **Exact diff:** `review/full.patch` includes every tracked edit and untracked addition, including binary model/profile additions and the complete analysis CSV. `review/source.patch` excludes only the four bulky generated inference/analysis assets (their exact hashes are in the manifest). `review/diff-summary.txt` gives per-file counts. No files staged; no commits made.
22. **Remaining blockers/limits:** Manual browser review remains blocked by app approval timeout. Strict all-research-tests-green is blocked by missing public research fixtures and obsolete legacy harness assumptions. The unused public Console cannot build from its incomplete public source. Core production Web/container checks pass; these limitations are not concealed as a full-solution/all-tests pass. Push/deployment are intentionally held pending your review.

## Exact packaged-file hashes (inspected bytes)

| File | SHA-256 |
| --- | --- |
| `artifacts/inference/production_inference_bundle.joblib` | `b6ad864066e097a260dc88e4fee92c031895cab239fe3eb3e4c80ccfcd70efa9` |
| `artifacts/inference/player_profiles.parquet` | `4a97b5aa52b3f1a1f0ffae177ea60c5bfe766c7f5d413233eb4f40f00c53977d` |
| `artifacts/inference/player_metadata.json` | `ddaead2c5d4b6a3806c54b10d84f2890b616b6491939fed92dd71d98addb44d9` |
| `data/analysis/decision_2025_26_regular.csv` | `76237d3466b8f57fd56840ea5d088fb47769624aa69369cd8a53610331337417` |
| `data/historical/processed/player_seasons.csv` | `fc7da096c7274af19de867dbdc92f862529e03a87899c831f001c93da5185e6d` |
| `data/historical/processed/player_team_splits.csv` | `15975dc1a122df5065be1a9a40400822d057dcaa6b1bdd3aa3e850e51a17b52f` |
| `data/historical/processed/team_seasons.csv` | `2a0ce558a2acf0d6b1e0c6063cb05a3b7f2e931e98126b2f0bf16414614755b1` |
| `data/historical/processed/league_seasons.csv` | `5630d6733447ccbfdb74ad5e6dbdaaa3d5d5a46daa94a0a62758b8b21500bb4b` |
| `data/simulation/production_v1/config.json` | `7412c18370219f400070bfe6f96c6ac0f0ea5bb2d67d5203c4237bae464fd341` |
| `data/simulation/production_v1/defensive_profiles.csv` | `30ef2bbdcbd4e1bc9f749e9db2b257f97bc6bbe69dbc7cd544d81118d2869c7e` |
| `data/simulation/production_v1/joint_residuals.csv` | `65e565992b13ca986884fcef46f792f619d59c1ea63c488d70e16567ab20f590` |
| `data/simulation/production_v1/overtime_residuals.csv` | `2085ed92917e6f6c3329953a9378d52da792f7fa34f20dcf938a6565084cffde` |

## Exact changed-file inventory

- Modified: `.gitignore`
- Modified: `nba-clutch-agent/AiAgents.NbaClutchAgent.Application/AiAgents.NbaClutchAgent.Application.csproj`
- Modified: `nba-clutch-agent/AiAgents.NbaClutchAgent.Domain/AiAgents.NbaClutchAgent.Domain.csproj`
- Modified: `nba-clutch-agent/AiAgents.NbaClutchAgent.Infrastructure/AiAgents.NbaClutchAgent.Infrastructure.csproj`
- Modified: `nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj`
- Modified: `nba-clutch-agent/AiAgents.NbaClutchAgent.Web/Program.cs`
- Modified: `nba-clutch-agent/AiAgents.NbaClutchAgent.Web/Simulation/SimulationEndpoints.cs`
- Modified: `scripts/inference/inference_service.py`
- Added: `.dockerignore`
- Added: `Dockerfile`
- Added: `artifacts/inference/player_metadata.json`
- Added: `artifacts/inference/player_profiles.parquet`
- Added: `artifacts/inference/production_inference_bundle.joblib`
- Added: `data/analysis/decision_2025_26_regular.csv`
- Added: `deployment/FINAL_REPORT.md`
- Added: `deployment/PROVENANCE.md`
- Added: `deployment/README.md`
- Added: `deployment/requirements.lock`
- Added: `deployment/requirements.txt`
- Added: `deployment/runtime-artifacts.json`
- Added: `deployment/start.py`
- Added: `deployment/test-results.json`
- Added: `deployment/validate_container.py`
- Added: `deployment/validation-results.json`
- Added: `nba-clutch-agent/AiAgents.NbaClutchAgent.Application/packages.lock.json`
- Added: `nba-clutch-agent/AiAgents.NbaClutchAgent.Domain/packages.lock.json`
- Added: `nba-clutch-agent/AiAgents.NbaClutchAgent.Infrastructure/packages.lock.json`
- Added: `nba-clutch-agent/AiAgents.NbaClutchAgent.Web/Deployment/PublicDemo.cs`
- Added: `nba-clutch-agent/AiAgents.NbaClutchAgent.Web/packages.lock.json`
- Added: `nba-clutch-agent/Directory.Build.props`
- Added: `railway.json`

# HoopsLab public V1 deployment

Isolated branch: `deployment/railway-v1`, based only on `86a1821e8874d49a72f5300a8f6d7ace1150fe33`. No push or deployment has been performed.

See `PROVENANCE.md` before replacing runtime assets. No database, writable model state, secret, or persistence volume is required.

## Architecture

One Linux amd64 container. ASP.NET Core serves the existing static frontend and public API. Python inference binds only `127.0.0.1:8011`; Python simulation binds only `127.0.0.1:8012`. Only the .NET HTTP port is published. Final process runs as UID/GID 10001. No new application/animation framework or frontend build is introduced.

The .NET 10 SDK and ASP.NET runtime images and CPython 3.12.10 image are pinned by digest. .NET package lockfiles and Python wheel SHA-256 hashes lock application dependencies. Debian native dependency versions are also pinned. Repository indexes still come from the live Debian repository; byte-for-byte OS-layer reproducibility is not claimed, and a removed pinned package must trigger an explicit dependency refresh rather than silently selecting a newer version. Rebuild deliberately when updating base digests/security dependencies, then rerun parity checks.

The .NET 10 portable official runtime is copied into Debian 12 with its native dependencies; official .NET 10 images use Ubuntu by default. Debian 12 remains supported for .NET 10. Source: https://learn.microsoft.com/en-us/dotnet/core/install/linux-debian . Migration guidance: https://learn.microsoft.com/en-us/aspnet/core/migration/90-to-100 . There is no application behavior refactor.

## Local build/run

```powershell
docker build --platform linux/amd64 -t hoopslab-v1:local .
docker run -d --name hoopslab-v1-review -p 127.0.0.1:8080:8080 -e PORT=8080 hoopslab-v1:local
```

Only map the .NET port. Never publish 8011 or 8012. Do not override the image entrypoint.

## Startup and health

`deployment/start.py` is PID 1. It validates every packaged artifact's SHA-256 (binary exact; CSV/JSON normalized to LF for cross-platform Git checkouts) before launching Python children, waits for both `/ping` responses, then starts .NET. Startup timeout: 180 seconds. Any required process exit (including exit code 0) fails the container and terminates siblings. SIGTERM/SIGINT stops all process groups, waits up to 15 seconds and kills/reaps stragglers. No shell-managed background jobs.

`/ping`: .NET liveness. `/ready`: checks all required runtime paths plus successful `ok:true` health responses from both Python services. Full checksum verification happens before startup; the simulator also verifies its frozen kernel/data contract. Docker checks readiness continuously; Railway healthchecks gate deployment but do not monitor continuously after release (https://docs.railway.com/deployments/healthchecks).

## Public demo limits

Legacy `/api/run`, `/api/agent/*` and `/api/baseline` return 404 for every verb. Runner/background worker/Q-table dependencies are not registered, so no training or state writes happen. Web Dataset/shots.tsv is deliberately excluded.

API concurrency: 4 with no queue. Shared API budget: 240/minute; combined POST Shot Lab + simulation budget: 20/minute. Limits return 429. These deliberately shared portfolio-demo budgets avoid trusting spoofable forwarded IPs, and may need tuning for legitimate traffic. Request bodies are limited to 65,536 bytes, including chunked bodies. Upstream 5xx/transport exceptions return generic messages; internal errors go to container logs. Existing simulation validation still limits draw counts and payload shape.

## Runtime data

- All three verified `artifacts/inference/` files.
- Verified `data/analysis/decision_2025_26_regular.csv` (historical decision browser).
- Public V1 `data/historical/processed/` canonical/prototype CSVs and metadata.
- Public V1 `data/simulation/production_v1/` frozen configuration, CSV arrays, reference/verification JSON.
- V1 inference core/service, simulation Python modules and their historical Python imports.
- Published .NET Web app with existing wwwroot.

Layout is `/app/{scripts,data,artifacts,web,deployment}`. The historical service discovers `/app/data` by walking ancestors of `/app/web`. No raw research archives, training joblibs, V2 source, development secrets, Git metadata, bin/obj, or legacy Dataset are in the Docker context.

## Railway setup (not performed)

Use repository root / Dockerfile, Linux amd64, one replica, no database or volume. `railway.json` selects `/ready`, a 300-second deployment health timeout and on-failure restart (up to 3 retries). Leave Start Command unset to retain the supervisor entrypoint. Select memory based on the measured validation report with headroom; do not choose a tier from simulator-only research numbers.

| Variable | Value / behavior |
| --- | --- |
| `PORT` | Provider supplies the public HTTP port; local default 8080. Must not be 8011/8012. Supervisor binds .NET to 0.0.0.0 on this port. |
| `ASPNETCORE_ENVIRONMENT` | Production (image default) |
| `ASPNETCORE_URLS` | Derived by supervisor from PORT; do not set a conflicting value. |
| `ProductionInference__BaseUrl` | Optional; if set must be exactly http://127.0.0.1:8011 |
| `Simulation__BaseUrl` | Optional; if set must be exactly http://127.0.0.1:8012 |
| `APP_ROOT` | Set internally by supervisor; do not override. |
| `RAILWAY_DEPLOYMENT_DRAINING_SECONDS` | Consider 20 seconds for graceful request termination. |

No fake credentials or secrets are required.

## Existing research checks

Some public V1 research checks depend on ignored/non-public cached raw files and research datasets. Report missing fixtures explicitly; do not fetch/copy V2 research artifacts to make these tests pass. The deployment image intentionally excludes those research inputs. Production numerical reference tests, inference parity, historical validation, API tests, and lifecycle/isolation checks are separately reported in the local validation report.

## Public source limitation

The original public V1 Console host cannot build: its `Infrastructure.Dataset` namespace and CsvShotStream source are absent from public Git. The original Web legacy training wiring referenced the same missing sources; deployment removes that unused wiring. Docker builds/publishes the actual Web project and its three library dependencies on .NET 10. The legacy Console is not packaged or executed; its source/target remains unchanged at .NET 9, and a full solution build is not claimed to pass. No missing implementation was copied from V2. There are no .NET test projects in the public solution; `dotnet test` is a successful no-test invocation, not a .NET unit-test suite.

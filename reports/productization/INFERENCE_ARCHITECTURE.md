# Production Inference Architecture

## Selected Approach

Production uses a local Python HTTP inference service backed by the frozen sklearn artifacts, with the .NET Web project exposing stable application routes as a thin proxy.

ONNX/native .NET inference was not selected because the production pipeline depends on sklearn `ColumnTransformer`, `OrdinalEncoder`, `HistGradientBoostingClassifier`, and sigmoid calibration. Exporting every preprocessing and calibration step would introduce parity risk. The selected approach preserves exact Python predictions and keeps the .NET application clean.

## Production Model

- Shot model: `historical_shooter_profile`
- Model family: HistGradientBoosting classifier
- Model version: `shot-quality-v1.1`
- Policy version: `decision-v1`
- Replacement xPPS: loaded from `artifacts/decision/decision_policy_v1.joblib`
- REVIEW threshold: loaded from `artifacts/decision/decision_policy_v1.joblib`
- Possession Policy v2: research/context only, not used for production recommendations

## Artifacts

- `artifacts/inference/production_inference_bundle.joblib`
- `artifacts/inference/player_profiles.parquet`
- `artifacts/inference/player_metadata.json`

The bundle contains the frozen shooter model, the v1 sigmoid calibrator, model columns, production metadata, and Decision Policy v1 replacement/threshold values. Player profiles are built from the frozen historical feature definitions and current available canonical data.

## Feature Contract

`POST /api/decision/analyze` accepts legitimate pre-shot inputs:

- `playerId`
- `shotDistance`, `locX`, `locY`
- `shotValue`
- `shotType`, `actionType`
- optional `shotZoneBasic`, `shotZoneArea`, `shotZoneRange`
- `period`, `secondsRemaining`, `scoreDiff`
- `isHome`, `isOvertime`
- optional `gameState`

The caller does not supply historical shooter statistics. The inference service resolves those from `player_profiles.parquet`, with cold-start fallback to the same global priors used by modeling.

## API Routes

- `POST /api/decision/analyze`
- `GET /api/players?q=&limit=`
- `GET /api/players/{id}/profile`
- `GET /api/analysis/decisions?page=&pageSize=&player=&recommendation=&clutchOnly=`
- `GET /api/model`

The Python service exposes the same paths on its local port. The .NET Web project proxies these routes so future UI work can call the existing application host.

## Failure Behavior

The service returns `400` for malformed requests, invalid shot value, invalid clock/period, invalid distance, or invalid JSON. Unknown players are allowed only through explicit cold-start fallback and are marked with `COLD_START_PROFILE`. Missing model/profile artifacts fail service startup rather than silently substituting values.

## Parity Testing

`scripts/inference/test_inference.py` compares production inference against direct Python model prediction using the same loaded artifacts with probability tolerance at 12 decimal places. Tests also cover TAKE, REVIEW, DEFER, cold-start behavior, invalid request handling, probability range, xPPS math, margin math, and threshold exposure.

## Local Startup

Start the inference service:

```powershell
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe scripts/inference/inference_service.py --port 8011
```

Then start the .NET Web app:

```powershell
dotnet run --project nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj
```

## Known Limitations

- Python inference service must be running for production decision endpoints.
- Player profiles are fixed at the current artifact cutoff and should be regenerated when canonical data changes.
- Zone inference is simple if callers omit explicit NBA zone fields; the future simulator should send zone fields when available.
- Historical analysis currently exposes Decision Policy v1 2025-26 regular-season CSV data with pagination.

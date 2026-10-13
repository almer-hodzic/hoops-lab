# Project Documentation

## Architecture Overview

The project has four layers:

1. Data pipeline: raw NBA archives to canonical parquet datasets.
2. Modeling: shot-quality probability models and decision-policy artifacts.
3. Inference: Python service loads sklearn/joblib artifacts.
4. Web app: .NET application proxies API calls and serves the Decision Lab frontend.

## Folder Structure

- `data/raw/`: raw NBA archives.
- `data/processed/`: canonical datasets.
- `data/analysis/`: generated analysis outputs.
- `artifacts/models/`: trained model artifacts.
- `artifacts/inference/`: production inference bundle and player profiles.
- `artifacts/decision/`: Decision Policy artifacts.
- `scripts/data/`: data audit/build scripts.
- `scripts/modeling/`: model training scripts.
- `scripts/decision/`: decision-policy scripts.
- `scripts/inference/`: Python inference service.
- `nba-clutch-agent/`: .NET/Web application.
- `reports/`: audit, modeling, decision, and productization reports.

## Prerequisites

- Python 3.12 with packages from project requirements files.
- .NET SDK 9.
- Existing generated artifacts in `artifacts/` and `data/processed/`.

## Run Python Inference Service

From the project root:

```powershell
python scripts/inference/inference_service.py --host 127.0.0.1 --port 8011
```

## Run .NET Web App

From the project root:

```powershell
dotnet run --project nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj
```

## Key API Endpoints

- `GET /api/model`
- `GET /api/players?q=...`
- `GET /api/players/{id}/profile`
- `POST /api/decision/analyze`
- `GET /api/analysis/decisions`
- `POST /api/experimental/defense/analyze`
- `GET /api/experimental/defense/contract`
- `GET /api/defenders?q=...`

## Production Model Flow

User context is converted into canonical pre-shot features. The production inference bundle predicts `p_make`; Decision Policy computes `shot_xpps`, `replacement_xpps`, `decision_margin`, and TAKE/REVIEW/DEFER.

## Experimental Defensive Flow

Advanced Context calls the V1.1 defensive tracking experiment. It uses closest defender distance, shot clock, dribbles, touch time, and safe shooter rates. Defender profile display is context-only and does not alter V1.1 probability. OOD support hides absolute experimental probability in the UI.

## Important Artifact Locations

- Production inference: `artifacts/inference/production_inference_bundle.joblib`
- Production selected shot model: `artifacts/models/shooter_skill_v11_selected_model.joblib`
- Decision Policy v1: `artifacts/decision/decision_policy_v1.joblib`
- Defensive V1.1: `artifacts/models/experimental_defensive_shot_quality_v11.joblib`
- Metrics summary: `submission/metrics_summary.json`

## Limitations

The model is predictive, not causal. Playoff calibration drift exists. Defensive context is trained on 2014/15 tracking data and remains experimental. Possession-continuation/counterfactual modeling is feasible but not production-ready in the current artifact set.

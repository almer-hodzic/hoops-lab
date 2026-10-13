# ML NBA Agent Workspace

This workspace separates the existing NBA Clutch Agent application from newly downloaded datasets and audit outputs.

## Structure

- `nba-clutch-agent/`: existing .NET application source.
- `data/raw/`: immutable downloaded archives, grouped by dataset type and season type.
- `data/interim/`: future temporary normalized data; currently intentionally empty.
- `data/processed/`: future model-ready data; currently intentionally empty.
- `data/metadata/`: machine-readable audit summaries.
- `scripts/data/`: rerunnable data audit scripts.
- `reports/data-audit/`: human-readable audit reports.

## Raw Data Policy

Do not edit, overwrite, or delete files under `data/raw/`. Audit scripts read compressed archives directly and write generated outputs only to `reports/` and `data/metadata/`.

## Run The Audit

```powershell
python scripts/data/audit_nba_data.py
```

No third-party Python packages are required.

## Next Stage

Review the audit reports, add regular-season archives if needed, then implement a canonical interim pipeline. Modeling and frontend changes are intentionally out of scope for this audit stage.

## Canonical Dataset Pipeline

Install data dependencies:

```powershell
python -m pip install -r scripts/data/requirements.txt
```

Run the audit:

```powershell
python scripts/data/audit_nba_data.py
```

Build canonical processed datasets:

```powershell
python scripts/data/build_canonical_dataset.py
```

Outputs are written to `data/processed/`, `data/interim/`, `data/metadata/`, and `reports/data-pipeline/`. Raw archives under `data/raw/` are immutable. The temporal split is train `2020-21` through `2023-24`, validation `2024-25`, and test `2025-26`.

## Production Inference

Build or refresh production inference artifacts:

```powershell
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe scripts/inference/build_production_inference_artifacts.py
```

Run deterministic inference tests:

```powershell
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe scripts/inference/test_inference.py
```

Start the local Python inference service:

```powershell
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe scripts/inference/inference_service.py --port 8011
```

Start the .NET app, which proxies production inference routes:

```powershell
dotnet run --project nba-clutch-agent/AiAgents.NbaClutchAgent.Web/AiAgents.NbaClutchAgent.Web.csproj
```

Production recommendation uses Shooter Skill v1.1 plus Decision Policy v1. Possession Policy v2 remains research/context only.

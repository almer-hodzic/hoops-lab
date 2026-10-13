# Experimental Defensive Context Integration

The Decision Lab Advanced Context feature now uses Defensive Shot Quality V1.1 as its active experimental runtime model. This remains research-only: it does not replace the production Shot Quality model, does not change Decision Policy v1, and cannot alter TAKE / REVIEW / DEFER.

## Production Path

- `POST /api/decision/analyze`
- Modern 2020-21 through 2025-26 shot-quality model
- Historical shooter-profile features
- Decision Policy v1 replacement benchmark and threshold
- Primary recommendation remains TAKE / REVIEW / DEFER

## Experimental Path

- `POST /api/experimental/defense/analyze`
- Serialized artifact: `artifacts/models/experimental_defensive_shot_quality_v11.joblib`
- Active runtime model: `defensive_shot_quality_v11`
- Selected candidate: `V11_A_hgb_regularized_sigmoid`
- Variant: tracking plus shooter rates only
- Calibration: sigmoid
- Training domain: 2014-15 shot-level tracking data
- Status: `EXPERIMENTAL`

V1 remains preserved as the research predecessor in `artifacts/models/experimental_defensive_shot_quality_v1.joblib`, but Advanced Context analysis no longer loads it.

## Feature Contract

V1.1 uses:

- `shot_distance`
- `shot_value`
- `period`
- `seconds_remaining_period`
- `is_home`
- `closest_defender_distance`
- `shot_clock`
- `dribbles`
- `touch_time`
- `shot_clock_missing`
- `touch_time_missing`
- `touch_time_invalid`
- `distance_bucket`
- `shooter_prior_fg_pct_shrunk`
- `shooter_prior_2p_pct_shrunk`
- `shooter_prior_3p_pct_shrunk`
- `shooter_distance_prior_fg_pct_shrunk`

Explicitly excluded:

- `shooter_prior_fga`
- `shooter_prior_2pa`
- `shooter_prior_3pa`
- `shooter_distance_prior_attempts`
- `defender_pt_overall_d_fg_pct`
- `defender_pt_overall_d_fga`
- `defender_pt_overall_normal_fg_pct`
- `defender_pt_overall_pct_plusminus`

## Why V1.1

The V1 research model used isotonic calibration plus shooter attempt counts and defender PT profile fields. V1.1 was selected because sigmoid calibration reduced extreme near-rim behavior, attempt counts were not semantically safe to adapt from modern multi-season profiles into a 2014-15 season-to-date tracking model, and defender profile fields were weak and unstable as probability inputs.

Closest defender distance remains a model feature. Defender identity and defender profile statistics are display context only.

## Modern Adapter

The runtime maps only semantically defensible modern shooter rates into V1.1:

- `shooter_prior_fg_pct_shrunk`
- `shooter_prior_2p_pct_shrunk`
- `shooter_prior_3p_pct_shrunk`
- `shooter_distance_prior_fg_pct_shrunk`

Missing shooter rates use the documented cold-start population fallback. The API exposes `shooterColdStart` and `mappedShooterFeatures` as diagnostics.

## Support-Aware Display

The API returns `supportStatus`, `supportWarnings`, `outOfRangeFeatures`, and `sparseContext`.

- `IN_DISTRIBUTION`: show experimental P(make) and the production comparison.
- `LIMITED_SUPPORT`: show experimental P(make) with a concise caution.
- `OUT_OF_DISTRIBUTION`: hide the absolute experimental probability in the normal UI and show an outside-training-support explanation.

Internal probabilities are kept under diagnostics for auditing and are not used as headline UI values when support is out of distribution.

## Pressure Sensitivity

Pressure sensitivity now uses V1.1 at 1, 2, 4, 6, and 8 feet. Each point returns defender distance, probability when displayable, and support status. The UI presents a value row, labels limited support, and marks unsupported points as OOD instead of drawing a misleading continuous curve.

## Domain-Shift Limitation

The model was trained on 2014-15 tracking data and is useful for exploring associations among defensive pressure, shot clock, dribbles, touch time, and make probability. It is not validated as a modern production probability model and does not affect the production recommendation.

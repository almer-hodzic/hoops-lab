# Public V1 artifact provenance gate

Base: `86a1821e8874d49a72f5300a8f6d7ace1150fe33` (2026-10-15).
Audit performed locally on 2026-10-27 before deployment source edits. Only public V1 code was executed; the existing development workspace was read-only. No development source was copied.

## Evidence and reproduction

1. All 24 shot-detail/play-by-play archives matched the SHA-256 values committed in `data/metadata/dataset_summary.json` at the base commit.
2. Ran the public V1 `build_canonical_dataset.process_pair` for each season/type using those verified archives. All 1,368,990 canonical rows matched the existing `shots_enriched.parquet` exactly, with dtype differences ignored but exact cell values required.
3. Used public V1 `train_shooter_skill_v11.build_historical_features`, the `hist_gradient_boosting / historical_shooter_profile` spec, and the original fixed seed. Refit that selected estimator on 843,622 regular-season training rows (2020-21 through 2023-24).
4. Compared rebuilt and frozen model probabilities on ALL 1,368,990 canonical rows, in chunks of 25,000, with absolute tolerance 1e-12 and zero relative tolerance: PASS. Object serialization hashes differed; byte-identical retraining is not claimed.
5. Refit the sigmoid calibrator on the late 2024-25 regular-season validation split. Coefficients and intercept matched the bundle within 1e-12.
6. Regenerated player profiles with the public V1 generator in memory: all profile cells matched exactly, player metadata JSON matched structurally, and global priors matched. Model columns matched the selected V1 spec. Replacement value matched the V1 training mean; threshold matched the committed Decision Policy V1 metrics (0.15).
7. Recomputed all 219,158 analysis CSV rows using the V1 model/calibrator, global replacement policy and threshold: all columns matched (numeric tolerance 1e-12).
8. The five generating/runtime source files inspected in the local development workspace were byte-equivalent to public V1 after CRLF normalization: canonical builder, shooter trainer, Decision Policy V1 builder, production inference builder, and inference core. These source files were NOT copied.
9. Bundle metadata identifies `shot-quality-v1.1`, `decision-v1`, generated `2026-10-11T10:22:29Z`, before the pinned public commit. This metadata is supporting evidence, not independent cryptographic attestation.
10. The unchanged public V1 inference suite passed all 11 tests against the local artifacts via explicit constructor path arguments (in-memory defaults substitution only; no source edit or artifact copy for that test).

Result: V1 generation lineage and numerical/runtime compatibility verified. Only then were the three inference artifacts and analysis CSV copied, with source and destination SHA-256 checked. No Season/Dynasty artifacts or unpublished V2 source are included.

## Generators

| Artifact | Public V1 generator / inputs |
| --- | --- |
| `production_inference_bundle.joblib` | `scripts/inference/build_production_inference_artifacts.py`: frozen shooter model from `scripts/modeling/train_shooter_skill_v11.py`, Decision Policy V1 from `scripts/decision/build_decision_policy_v1.py`, canonical shots from `scripts/data/build_canonical_dataset.py` |
| `player_profiles.parquet` | Same inference builder, verified canonical shots and V1 historical feature definitions |
| `player_metadata.json` | Same inference builder, verified canonical shots |
| `decision_2025_26_regular.csv` | `scripts/decision/build_decision_policy_v1.py`, verified canonical shots, frozen V1 shooter and sigmoid calibration |
| Web `Dataset/shots.tsv` | No producing script or verifiable source lineage found in public V1. NOT copied or packaged. Only legacy baseline/training used it; those endpoints and runner registration are removed/blocked. |

## Checksums

The authoritative packaged-file SHA-256 values and byte sizes are in `runtime-artifacts.json`. The `sha256` fields record exact inspected byte hashes, including Windows checkout line endings. CSV/JSON additionally record `text_sha256` for LF-normalized startup verification, so Linux Git checkouts remain valid. Binary model/profile hashes always require exact bytes. The simulator additionally checks its original newline-normalized data and numerical-kernel fingerprints at startup.

Inspected but not packaged upstream inputs:

- Shooter model: `c6190aa059e71e605672431148b2a3115eacfb75f663f1462c412f8f9f9fa98d`
- Decision Policy V1: `81b0a194ac433c613a61712cbd71f586ed5ec43427d1df18d4594a6fd7cc7f26`
- Canonical shots: `5e0030726a8cca44ddb1510d95881d2795963632a890411f9ec551408ed8791d`
- Legacy shots.tsv (excluded): `007478123b01355d060b1736b026e5db0b9f3dfe24f99312cad0bd2660caf985`

## Python compatibility

Windows working environment: CPython 3.12.10. The serialized sklearn version is 1.9.0, matching installed sklearn. Runtime package versions are preserved in `requirements.txt`; Linux amd64 CPython 3.12 wheel hashes are pinned in `requirements.lock`. The lock intentionally targets Railway Linux amd64. No newest-version substitution was made.

## Reproducing the audit

Use a clean checkout of the pinned commit and the same Python versions (plus the training script's matplotlib dependency). Treat the external artifact/source-data directory as read-only and disable bytecode writes. Verify each raw archive against the committed manifest, reconstruct each canonical partition with `process_pair`, then refit only the selected shooter spec and late-validation sigmoid calibrator. Compare full-dataset predictions and regenerate profiles/metadata and the analysis CSV as described above. The normal inference test alone proves runtime compatibility, not the complete generation lineage; retain these data-reproduction checks when replacing artifacts.

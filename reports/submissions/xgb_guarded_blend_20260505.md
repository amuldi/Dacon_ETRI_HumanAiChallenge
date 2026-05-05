# XGB Guarded Blend 2026-05-05

- Current best: `lgb_temporal_s4b650.csv` public `0.5829008297`
- XGB run: `xgb_targetwise_histmix_guarded_20260505`
- Method: train target-wise XGBoost, then blend into current best only when OOF blend improves the target.
- S4 is frozen to avoid damaging the public-proven S4 beta axis.

## Self-Critique

- XGBoost adds model diversity, but the dataset has only 450 train rows and many features, so raw XGB can overfit.
- A direct XGB submission is not acceptable under the 'do not drop' requirement.
- The guarded blend can still drop public if OOF does not match public, so the first file is capped at 3%.

## OOF Scores

| Target | Current | XGB raw |
|---|---:|---:|
| Q1 | `0.576384` | `0.592363` |
| Q2 | `0.570476` | `0.615270` |
| Q3 | `0.576026` | `0.615885` |
| S1 | `0.508579` | `0.545422` |
| S2 | `0.547219` | `0.581090` |
| S3 | `0.526192` | `0.558885` |
| S4 | `0.652198` | `0.634070` |

- Current mean: `0.565296`
- XGB raw mean: `0.591855`

## Guarded Candidates

| File | OOF mean | Selected weights | Decision |
|---|---:|---|---|
| `archive/2026-05-05_xgb_guard_blocked/submit_xgb_guard003_s4b650.csv` | `0.565296` | `none` | Blocked; no XGB target passed the guard. |
| `archive/2026-05-05_xgb_guard_blocked/submit_xgb_guard005_s4b650.csv` | `0.565296` | `none` | Blocked; no XGB target passed the guard. |

## Decision Rule

- XGB did not pass the guard and should not be uploaded.
- The generated guard files were archived, not kept in `submissions/ready/`.
- If we still want model diversity, the next attempt should use stronger feature selection or a different validation design before blending.

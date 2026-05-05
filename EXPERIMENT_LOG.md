# Experiment Log

## Current Best

| Date | File | Public | OOF | Experiment | Decision |
|---|---|---:|---:|---|---|
| 2026-05-04 | `lgb_temporal_s4b650.csv` | `0.5829008297` | `0.565296` | `toward_57_s4_ladder_20260504_b650` | current best |

## Public Score History

| Date | File | Public | Result |
|---|---|---:|---|
| 2026-04-25 | `submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv` | `0.5960566585` | original anchor |
| 2026-04-29 | `next_stable_tuned.csv` | `0.5956332255` | improved |
| 2026-04-29 | `lgb_stable_calibrated.csv` | `0.5946792872` | improved |
| 2026-04-30 | `lgb_calib_tight.csv` | `0.5944158654` | improved |
| 2026-05-01 | `lgb_57_antis4_probe.csv` | `0.5943866101` | small improved |
| 2026-05-01 | `lgb_temporal_prior.csv` | `0.5886545849` | large improved |
| 2026-05-02 | `lgb_temporal_targetwise.csv` | `0.5863944910` | improved |
| 2026-05-02 | `lgb_temporal_push.csv` | `0.5880629390` | failed, archived |
| 2026-05-02 | `lgb_temporal_s4half.csv` | `0.5856099175` | improved |
| 2026-05-03 | `lgb_temporal_s4b065.csv` | `0.5853942690` | improved |
| 2026-05-03 | `lgb_temporal_s4b085.csv` | `0.5851199626` | improved |
| 2026-05-03 | `lgb_temporal_s4b110.csv` | `0.5847975097` | improved |
| 2026-05-04 | `lgb_temporal_s4b130.csv` | `0.5845552904` | improved |
| 2026-05-04 | `lgb_temporal_s4b650.csv` | `0.5829008297` | improved but flattened |
| 2026-05-04 | `lgb_temporal_s4up650.csv` | `0.5854286839` | failed; stop S4 up-only |
| 2026-05-05 | `submit_s4down650_fixed.csv` | `0.5838666368` | failed; stop S4 directional probes |
| 2026-05-05 | `submit_qshead160_s4b650.csv` | `0.5837781861` | failed; stop Q/S recovery |
| 2026-05-05 | `submit_s4sym800_s4b650.csv` | `0.5845567721` | failed; stop S4 extension |

## Current Submission Folder

```text
submissions/ready/
  lgb_temporal_s4b130.csv
  lgb_temporal_s4b650.csv
```

Old ready CSV files were moved to:

```text
submissions/archive/2026-05-01_after_temporal_prior/
submissions/archive/2026-05-02_push_failed/
submissions/archive/2026-05-03_after_s4b110/
submissions/archive/2026-05-04_after_s4up650_failed/
submissions/archive/2026-05-05_down650_data_error/
submissions/archive/2026-05-05_s4_direction_failed/
submissions/archive/2026-05-05_qshead_failed/
submissions/archive/2026-05-05_s4_extension_failed/
submissions/archive/2026-05-05_xgb_guard_blocked/
```

## Latest Strategy

Stop post-processing. The current best remains `lgb_temporal_s4b650.csv`.

- S4 beta `6.50` is the current public best, but beta `8.00` failed public.
- `s4up650` and `s4down650` both failed public, so isolated S4 halves are stopped.
- `submit_qshead160_s4b650.csv` also failed public, so Q/S recovery is stopped.
- XGB guarded blend selected no non-zero weights, so model-diversity blend is blocked.
- Do not upload more post-hoc candidates today.

## Next Candidate

No next upload candidate is approved. Keep the best file and move to new validation/feature/model work.

| File | Beta | OOF | Public | Decision |
|---|---:|---:|---:|---|
| `lgb_temporal_s4b650.csv` | `6.50` | `0.565296` | `0.5829008297` | current best |
| `submit_s4sym800_s4b650.csv` | `8.00` | `0.569718` | `0.5845567721` | failed, archived |
| `submit_s4near950_s4b650.csv` | `6.50/9.50 gated` | `0.574614` |  | blocked, archived |
| `submit_s4sym950_s4b650.csv` | `9.50` | `0.574655` |  | blocked, archived |
| `submit_qshead160_s4b650.csv` | `S4 fixed` | `0.565063` | `0.5837781861` | failed, archived |
| `lgb_temporal_s4b130.csv` | `1.30` | `0.559702` | `0.5845552904` | previous best backup |
| `lgb_temporal_s4up650.csv` | `6.50 up-only` | `0.562956` | `0.5854286839` | failed, archived |
| `submit_s4down650_fixed.csv` | `6.50 down-only` | `0.562185` | `0.5838666368` | failed, archived |
| `lgb_temporal_push.csv` |  | `0.559537` | `0.5880629390` | failed, do not submit |

<!-- s4_public_anchored_20260505:start -->

## 2026-05-05 S4 Public-Anchored Reset

- Latest failed: `submit_qshead160_s4b650.csv` public `0.5837781861`.
- Self-critique: Q/S recovery was OOF-led and contradicted public feedback, so it is stopped.
- Corrected strategy: only the symmetric S4 beta axis has public support; up/down halves and Q/S moves are not supported.
- `submit_s4sym800_s4b650.csv` failed public at `0.5845567721`.
- Stop condition triggered: stop S4/QS post-processing and move to new model/features.

<!-- s4_public_anchored_20260505:end -->

<!-- xgb_guarded_blend_20260505:start -->

## 2026-05-05 XGB Guarded Blend

- Trained target-wise `xgboost.XGBClassifier` with the same histmix feature views.
- Raw XGB OOF mean: `0.591855`, worse than current `0.565296`.
- Per-target guard selected no non-zero blend weights; S4 stayed frozen by rule.
- Generated guard files were moved to `submissions/archive/2026-05-05_xgb_guard_blocked/`.
- Decision: do not upload XGB blend files unless a future XGB run passes the guard.

<!-- xgb_guarded_blend_20260505:end -->

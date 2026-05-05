# Stop Post-Hoc Submissions 2026-05-05

- Current best: `lgb_temporal_s4b650.csv` public `0.5829008297`
- Latest failed: `submit_s4sym800_s4b650.csv` public `0.5845567721`
- Decision: stop all remaining post-processing submissions.

## Failed Axes

| Axis | Evidence | Decision |
|---|---|---|
| S4 up-only | `lgb_temporal_s4up650.csv` public `0.5854286839` | stop |
| S4 down-only | `submit_s4down650_fixed.csv` public `0.5838666368` | stop |
| Q/S head push | `submit_qshead160_s4b650.csv` public `0.5837781861` | stop |
| S4 extension | `submit_s4sym800_s4b650.csv` public `0.5845567721` | stop |
| XGB guarded blend | raw XGB OOF `0.591855`; all selected weights `0` | blocked |

## Self-Critique

- I kept probing around the same post-hoc directions after `b650`.
- The public scores now show that `b650` was a narrow optimum, not a direction that could be safely extended.
- OOF improvements have repeatedly failed public, so OOF-only post-processing is no longer a reliable selector.

## Next Work

- Do not upload `submit_s4near950_s4b650.csv` or `submit_s4sym950_s4b650.csv`.
- Keep `lgb_temporal_s4b650.csv` as the only approved current-best file.
- Next improvement attempt must be a new validation/feature/model experiment, and it should produce a file only if a guard passes before submission.

# Submission Report: temporal_prior_v1

- Public result: `0.5886545849`
- Previous public best: `0.5943866101`
- Base run: `public_lgb_targetwise_57_antis4_probe_v1`
- Prior config: `{'kind': 'power', 'max_days': 7, 'power': 2.0, 'subject_weight': 0.05, 'global_weight': 0.05}`
- Primary file: `submissions/ready/lgb_temporal_prior.csv`
- Safe file: `submissions/ready/lgb_temporal_prior_safe.csv`

## OOF Target Scores

| Target | Base | Prior only | Primary blend | Safe blend | Primary weight | Safe weight | Primary test mean | Safe test mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | `0.588133` | `0.665557` | `0.578292` | `0.579737` | `0.27` | `0.16` | `0.5101` | `0.5067` |
| Q2 | `0.576568` | `0.627125` | `0.571879` | `0.572572` | `0.25` | `0.15` | `0.6030` | `0.6072` |
| Q3 | `0.580365` | `0.666567` | `0.579040` | `0.579268` | `0.12` | `0.07` | `0.5763` | `0.5745` |
| S1 | `0.509637` | `0.653023` | `0.509569` | `0.509582` | `0.02` | `0.01` | `0.6771` | `0.6771` |
| S2 | `0.554083` | `0.640505` | `0.553740` | `0.553787` | `0.07` | `0.04` | `0.6295` | `0.6288` |
| S3 | `0.534961` | `0.609209` | `0.533588` | `0.533805` | `0.14` | `0.08` | `0.6377` | `0.6342` |
| S4 | `0.614036` | `0.719500` | `0.614036` | `0.614036` | `0.00` | `0.00` | `0.5450` | `0.5450` |

## Decision

- Primary OOF mean: `0.562878` vs base `0.565397`.
- This is the first non-calibration candidate with a meaningful OOF move after the latest public feedback.
- It can still fail public if leaderboard split is not temporally random within subject; submit primary first, then safe only if needed.

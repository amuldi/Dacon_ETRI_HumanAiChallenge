# Submission Report: temporal_targetwise_v1

- Current public best: `0.5886545849`
- Current best file: `submissions/ready/lgb_temporal_prior.csv`
- Primary file: `submissions/ready/lgb_temporal_targetwise.csv`
- S4 probe file: `submissions/ready/lgb_temporal_targetwise_s4.csv`

## OOF Target Scores

| Target | Base anti-S4 | Current best | Targetwise | Targetwise+S4 | Prior only | Primary w | S4 w | Primary mean | S4 mean | Config |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q1 | `0.588133` | `0.578292` | `0.576384` | `0.576384` | `0.693716` | `0.27` | `0.27` | `0.5094` | `0.5094` | `{'kind': 'power', 'max_days': 7, 'power': 3.0, 'subject_weight': 0.05, 'global_weight': 0.05}` |
| Q2 | `0.576568` | `0.571879` | `0.570476` | `0.570476` | `0.642853` | `0.26` | `0.26` | `0.6059` | `0.6059` | `{'kind': 'power', 'max_days': 5, 'power': 2.0, 'subject_weight': 0.05, 'global_weight': 0.05}` |
| Q3 | `0.580365` | `0.579040` | `0.576026` | `0.576026` | `0.714089` | `0.18` | `0.18` | `0.5817` | `0.5817` | `{'kind': 'power', 'max_days': 3, 'power': 3.0, 'subject_weight': 0.05, 'global_weight': 0.05}` |
| S1 | `0.509637` | `0.509569` | `0.508579` | `0.508579` | `0.715790` | `0.07` | `0.07` | `0.6777` | `0.6777` | `{'kind': 'power', 'max_days': 3, 'power': 3.0, 'subject_weight': 0.05, 'global_weight': 0.05}` |
| S2 | `0.554083` | `0.553740` | `0.547219` | `0.547219` | `0.561607` | `0.43` | `0.43` | `0.6287` | `0.6287` | `{'kind': 'exp', 'max_days': 21, 'tau': 45.0, 'subject_weight': 0.05, 'global_weight': 0.05}` |
| S3 | `0.534961` | `0.533588` | `0.526192` | `0.526192` | `0.532513` | `0.57` | `0.57` | `0.6497` | `0.6497` | `{'kind': 'exp', 'max_days': 90, 'tau': 45.0, 'subject_weight': 0.05, 'global_weight': 0.05}` |
| S4 | `0.614036` | `0.614036` | `0.614036` | `0.612934` | `0.634415` | `0.00` | `0.19` | `0.5450` | `0.5465` | `{'kind': 'exp', 'max_days': 21, 'tau': 45.0, 'subject_weight': 0.05, 'global_weight': 0.05}` |

## Decision

- Primary OOF mean: `0.559845` vs current best `0.562878`.
- S4 probe OOF mean: `0.559687`, but S4 is historically unstable on public.
- Submit the S4-frozen targetwise file first.

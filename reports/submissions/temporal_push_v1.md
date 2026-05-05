# Submission Report: temporal_push_v1

- Current public best: `0.5863944910`
- Current best file: `submissions/ready/lgb_temporal_targetwise.csv`
- Push file: `submissions/ready/lgb_temporal_push.csv`
- Push S4 file: `submissions/ready/lgb_temporal_push_s4.csv`

## OOF Target Scores

| Target | Prior v1 | Current best | Push | Push+S4 | Alpha | Push mean | Push+S4 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | `0.578292` | `0.576384` | `0.575612` | `0.575612` | `2.00` | `0.5087` | `0.5087` |
| Q2 | `0.571879` | `0.570476` | `0.569877` | `0.569877` | `2.00` | `0.6087` | `0.6087` |
| Q3 | `0.579040` | `0.576026` | `0.575384` | `0.575384` | `1.72` | `0.5855` | `0.5855` |
| S1 | `0.509569` | `0.508579` | `0.508496` | `0.508496` | `1.39` | `0.6779` | `0.6779` |
| S2 | `0.553740` | `0.547219` | `0.547176` | `0.547176` | `1.09` | `0.6286` | `0.6286` |
| S3 | `0.533588` | `0.526192` | `0.526180` | `0.526180` | `1.04` | `0.6502` | `0.6502` |
| S4 | `0.614036` | `0.614036` | `0.614036` | `0.613001` | `0.00` | `0.5450` | `0.5462` |

## Decision

- Push OOF mean: `0.559537` vs current best `0.559845`.
- This is a narrow continuation of the public-success direction, not a new model family.
- Submit `lgb_temporal_push.csv` first; S4 version only after confirming this direction still improves public.

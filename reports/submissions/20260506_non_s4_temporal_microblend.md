# 2026-05-06 non-S4 temporal microblend

## Decision

- Candidate file: `/Users/jsh/Desktop/output/대회/ ETRI_휴먼이해 인공지능 논문경진대회/submissions/ready/submit_20260506_non_s4_temporal_microblend.csv`
- Current anchor: `public_lgb_targetwise_toward57_s4b650_20260504`
- Source signal: `public_lgb_targetwise_temporal_push_s4_v1`
- Internal guard: `PASS`
- Candidate OOF mean: `0.565105309`
- Current OOF mean: `0.565296154`
- Mean delta: `-0.000190846`

## Why this is different from failed S4 pushes

- S4 is fixed at the current public-best prediction.
- Q1/Q2/Q3/S1/S2/S3 may move only when target OOF improves.
- Every moved target is capped to mean absolute test drift `<= 0.005`.
- The candidate name avoids blocked axes: no S4 extension, no S4 up/down, no QS head/micro/scal, no XGB guard003/005.

## Target weights

| Target | Weight | OOF delta | Test drift | Reason |
|---|---:|---:|---:|---|
| `Q1` | `0.793` | `-0.000708036` | `0.004994197` | selected |
| `Q2` | `0.238` | `-0.000206341` | `0.004986951` | selected |
| `Q3` | `0.253` | `-0.000283225` | `0.004993978` | selected |
| `S1` | `1.000` | `-0.000083005` | `0.004600512` | selected |
| `S2` | `0.963` | `-0.000042666` | `0.003191285` | selected |
| `S3` | `1.000` | `-0.000012647` | `0.001745118` | selected |
| `S4` | `0.000` | `0.000000000` | `0.000000000` | frozen |

## Self-critique

1. The full temporal-push direction already failed public, so this candidate must not reuse its S4 movement.
2. The expected lift is small. It is a safer 5.7-direction probe, not a guaranteed jump to 0.57.
3. If public worsens, the lesson is that non-S4 temporal OOF gains are also split-sensitive; next work should move to feature-stable retraining instead of post-hoc blending.

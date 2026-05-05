# Submission Report: temporal_s4_recovery_v1

- Current public best: `0.5863944910`
- Failed file: `lgb_temporal_push.csv` public `0.5880629390`
- Next file: `submissions/ready/lgb_temporal_s4half.csv`

| Target | Current | S4 half | S4 full | Half test mean | Full test mean |
|---|---:|---:|---:|---:|---:|
| Q1 | `0.576384` | `0.576384` | `0.576384` | `0.5094` | `0.5094` |
| Q2 | `0.570476` | `0.570476` | `0.570476` | `0.6059` | `0.6059` |
| Q3 | `0.576026` | `0.576026` | `0.576026` | `0.5817` | `0.5817` |
| S1 | `0.508579` | `0.508579` | `0.508579` | `0.6777` | `0.6777` |
| S2 | `0.547219` | `0.547219` | `0.547219` | `0.6287` | `0.6287` |
| S3 | `0.526192` | `0.526192` | `0.526192` | `0.6497` | `0.6497` |
| S4 | `0.614036` | `0.613205` | `0.612934` | `0.5458` | `0.5465` |

## Decision

- Stop Q/S over-push. It failed public.
- Try the S4-only half step next; it changes only S4 and keeps the proven Q/S targetwise predictions.

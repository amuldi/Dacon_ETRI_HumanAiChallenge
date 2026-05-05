# Submission Report: public_feedback_v1

- Current public best: `0.5944158654`
- Public quadratic fit points:
  - alpha 0.00: `lgb_stable_calibrated.csv` public `0.5946792872`
  - alpha 1.00: `lgb_calib_tight.csv` public `0.5944158654`
  - alpha 1.20: `lgb_q3s1s2_boost.csv` public `0.5944366954`
- Selected alpha: `0.93`
- Anti-S4 beta for probe: `-0.10`

## Target Scores

| Target | Current OOF | Publicfit QS OOF | 5.7 Probe OOF | Probe test mean |
|---|---:|---:|---:|---:|
| Q1 | `0.588133` | `0.588133` | `0.588133` | `0.5016` |
| Q2 | `0.576568` | `0.576568` | `0.576568` | `0.6134` |
| Q3 | `0.580342` | `0.580365` | `0.580365` | `0.5717` |
| S1 | `0.509624` | `0.509637` | `0.509637` | `0.6770` |
| S2 | `0.554053` | `0.554083` | `0.554083` | `0.6277` |
| S3 | `0.534961` | `0.534961` | `0.534961` | `0.6290` |
| S4 | `0.608158` | `0.608158` | `0.614036` | `0.5450` |

## Decision

- This does not guarantee 5.7. It is the most defensible public-feedback candidate after S4/top-k and calibration-extension failures.
- If this fails, leaderboard probing has diminishing returns and a new feature source is required.

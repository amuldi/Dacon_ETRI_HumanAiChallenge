# Submission Report: q3s1s2_boost_v1

- Current public best: `0.5944158654`
- Current best OOF: `0.564548`
- Primary OOF: `0.564539`
- Safe OOF: `0.564541`

## Changed Targets

| Target | Current OOF | Primary OOF | Safe OOF | Primary Delta | Primary test mean | Safe test mean |
|---|---:|---:|---:|---:|---:|---:|
| Q3 | `0.580342` | `0.580301` | `0.580317` | `-0.000042` | `0.5709` | `0.5712` |
| S1 | `0.509624` | `0.509636` | `0.509621` | `+0.000011` | `0.6739` | `0.6750` |
| S2 | `0.554053` | `0.554014` | `0.554025` | `-0.000038` | `0.6275` | `0.6276` |

## Decision

- CatBoost light OOF was weaker (`0.582719`), so no CatBoost blend was used.
- `S3/S4` micro calibration failed on public, so both targets are frozen at current best.
- This is a narrow public-success-direction probe, not a guaranteed 5.7 jump.

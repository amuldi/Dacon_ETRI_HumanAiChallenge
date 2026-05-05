# Submission Report: temporal_s4_beta_v1

- Current public best: `0.5847975097` from `lgb_temporal_s4b110.csv`
- Strategy: keep Q1/Q2/Q3/S1/S2/S3 fixed and isolate the S4 beta ladder.
- Ready folder now keeps only current best and meaningful backups.

| File | Beta | OOF mean | S4 OOF | Decision |
|---|---:|---:|---:|---|
| `lgb_temporal_s4b110.csv` | `1.10` | `0.559689` | `0.612947` | current best |
| `lgb_temporal_s4b100.csv` | `1.00` | `0.559687` | `0.612934` | conservative backup |
| `lgb_temporal_s4b085.csv` | `0.85` | `0.559690` | `0.612958` | previous public best backup |
| `lgb_temporal_s4b095.csv` | `0.95` | `0.559687` | `0.612937` | archived, superseded |
| `lgb_temporal_targetwise_s4.csv` | `1.00` | `0.559687` | `0.612934` | full S4, highest risk |

## Decision

- Keep `lgb_temporal_s4b110.csv` as current best.
- Keep `lgb_temporal_s4b100.csv` and `lgb_temporal_s4b085.csv` as backups.
- Older ladder files were moved to `submissions/archive/2026-05-03_after_s4b110/`.

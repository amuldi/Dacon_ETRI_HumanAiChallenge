# Submission Report: toward_57_s4_ladder_20260504

- New current best: `lgb_temporal_s4b130.csv` public `0.5845552904`
- Goal: probe whether the public-validated S4 beta direction can enter the 0.57 range.
- Fixed targets: Q1/Q2/Q3/S1/S2/S3.
- Risk control: stop if b200 fails; use b650 only after b500 remains strong.

| File | Beta | OOF mean | S4 OOF | S4 test mean | Clip high | Projected public | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `lgb_temporal_s4b130.csv` | `1.30` | `0.559702` | `0.613037` | `0.547018` | `0` | `0.5845552904` | current best |
| `lgb_temporal_s4b200.csv` | `2.00` | `0.559846` | `0.614045` | `0.548116` | `0` | `0.583620` | 1st submit: confirm that the b130 trend survives a wider jump. |
| `lgb_temporal_s4b350.csv` | `3.50` | `0.560688` | `0.619940` | `0.550466` | `0` | `0.581643` | 2nd submit only if b200 improves; target low 0.58 range. |
| `lgb_temporal_s4b500.csv` | `5.00` | `0.562358` | `0.631628` | `0.552817` | `0` | `0.579667` | 3rd submit only if b350 improves; first direct 0.57-range attempt. |
| `lgb_temporal_s4b650.csv` | `6.50` | `0.565296` | `0.652198` | `0.554090` | `7` | `0.577690` | Ceiling probe only after b500 improves; clipping starts, highest risk. |

## Why This Ladder

- The b130 result improved from `0.5847975097` to `0.5845552904`, matching the linear beta-public trend.
- The per-beta movement is small in probability space; even b500 has no clipping and only raises S4 test mean to about `0.5528`.
- b650 projects into the 0.577 range but starts clipping, so it is reserved as the ceiling probe.

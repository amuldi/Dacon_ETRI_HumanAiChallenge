# Submission Report: today_s4_extension_20260504

- Current best: `lgb_temporal_s4b110.csv` public `0.5847975097`
- Strategy: continue the proven S4-only beta axis with smaller steps beyond beta 1.10.
- Fixed targets: Q1/Q2/Q3/S1/S2/S3.

| File | Beta | OOF mean | S4 OOF | S4 test mean | Decision |
|---|---:|---:|---:|---:|---|
| `lgb_temporal_s4b110.csv` | `1.10` | `0.559689` | `0.612947` | `0.546705` | current best |
| `lgb_temporal_s4b115.csv` | `1.15` | `0.559691` | `0.612961` | `0.546783` | 1순위: b110에서 가장 작은 추가 연장 |
| `lgb_temporal_s4b120.csv` | `1.20` | `0.559694` | `0.612981` | `0.546862` | 2순위: b115가 개선될 때만 제출 |
| `lgb_temporal_s4b130.csv` | `1.30` | `0.559702` | `0.613037` | `0.547018` | 3순위: b120까지 개선될 때만 제출, 고위험 |

## Upload Logic

- First upload: `lgb_temporal_s4b115.csv`.
- Continue to `b120` only if `b115` beats `0.5847975097`.
- Continue to `b130` only if `b120` also improves.
- If `b115` is worse, stop extension and optionally test `lgb_temporal_s4b100.csv` as the conservative diagnostic.

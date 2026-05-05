# Submission Report: post_b650_strategy_20260504

- New current best: `lgb_temporal_s4b650.csv` public `0.5829008297`
- b650 improved, but the observed gain is far above the old linear projection and now looks saturated.
- A S4-only path probably cannot reach 0.57 by itself; we need directional S4 diagnostics plus a small Q/S calibration branch.

| File | Mode | Beta | OOF mean | S4 OOF | S4 test mean | Min | Max | Clip low | Clip high | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `lgb_temporal_s4b650.csv` | current | `6.50` | `0.565296` | `0.652198` | `0.554090` | `0.071653` | `0.980000` | `0` | `7` | current best |
| `lgb_temporal_s4up650.csv` | `S4 up` | `6.50` | `0.562956` | `0.635814` | `0.611238` | `0.221304` | `0.980000` | `0` | `7` | First submit: decomposes b650 and tests whether the public gain came from raising S4 only. |
| `lgb_temporal_s4b1200.csv` | `S4 symmetric` | `12.00` | `0.585696` | `0.794998` | `0.559847` | `0.020000` | `0.980000` | `21` | `17` | Second submit only if S4 still looks alive; symmetric plateau check. |
| `lgb_temporal_s4b650_qscal.csv` | `S4 b650 + Q/S cal` | `6.50` | `0.564909` | `0.652198` | `0.554090` | `0.071653` | `0.980000` | `0` | `7` | Non-S4 fallback: keep b650 S4 and add conservative OOF calibration on Q/S targets. |
| `lgb_temporal_s4down650.csv` | `S4 down` | `6.50` | `0.562185` | `0.630420` | `0.487833` | `0.071653` | `0.864348` | `0` | `0` | Diagnostic only if up650 fails; tests whether lowering S4 rows was the useful half. |
| `lgb_temporal_s4b1800.csv` | `S4 symmetric` | `18.00` | `0.613069` | `0.986608` | `0.564621` | `0.020000` | `0.980000` | `36` | `45` | Ceiling probe only if b1200 improves meaningfully; high clipping risk. |

## Interpretation

- `s4up650` and `s4down650` split the b650 move into its two halves.
- `s4b1200` checks whether symmetric S4 has any useful residual slope.
- `s4b650_qscal` keeps the current best S4 and only applies conservative OOF-fitted Q/S calibration.

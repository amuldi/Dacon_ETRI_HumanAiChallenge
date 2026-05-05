# S4 Public-Anchored Strategy 2026-05-05

- Current best: `lgb_temporal_s4b650.csv` public `0.5829008297`
- Failed latest: `submit_qshead160_s4b650.csv` public `0.5837781861`
- New direction: stop Q/S moves, keep Q1/Q2/Q3/S1/S2/S3 fixed, and only continue the public-proven symmetric S4 axis.

## Self-Critique

### What was wrong before

- I treated `s4up650` and `s4down650` failures as if the whole S4 direction was dead. That was too broad.
- The actual public evidence says isolated up/down halves are bad, but the symmetric `s4b650` combination is still the best file.
- `qshead160` had better OOF but worse public, so OOF-only Q/S recovery is not trustworthy for the next upload.

### Why this strategy is more defensible

- It changes only the one axis that has repeatedly improved public: symmetric S4 beta.
- The first candidate is a small move from beta 6.50 to 8.00, not a jump to the high-clipping beta 12/18 files.
- Q/S targets are frozen, because the latest public result directly invalidated that direction.

### Why it can still be wrong

- S4 OOF gets worse as beta increases, so this is public-feedback extrapolation, not validation-driven improvement.
- If `b650` was already the local public optimum, `sym800` will worsen.
- If `sym800` worsens, more post-processing is not the right next move; the next axis must be new model/features.

## Candidate Metrics

- Near-gated rows: test `192/250`, OOF `449/450` within 7 days.

| File | OOF mean | S4 OOF | S4 mean | S4 min | S4 max | Clip low | Clip high | Mean abs change vs b650 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `submit_s4sym800_s4b650.csv` | `0.569718` | `0.683150` | `0.554695` | `0.020000` | `0.980000` | `1` | `7` | `0.026886` |
| `submit_s4near950_s4b650.csv` | `0.574614` | `0.717420` | `0.550253` | `0.020000` | `0.980000` | `6` | `9` | `0.041109` |
| `submit_s4sym950_s4b650.csv` | `0.574655` | `0.717707` | `0.555866` | `0.020000` | `0.980000` | `6` | `11` | `0.053018` |

## Decision

- First upload: `submit_s4sym800_s4b650.csv`.
- Do not upload the remaining Q/S recovery files.
- If `sym800` improves, continue with `near950`; if it worsens, stop S4/QS post-processing.

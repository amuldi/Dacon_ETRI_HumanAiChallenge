# Submission Report: qs_recovery_20260505

- Current best: `lgb_temporal_s4b650.csv` public `0.5829008297`
- Failed latest: `submit_s4down650_fixed.csv` public `0.5838666368`
- Strategy: freeze S4 and probe a constrained non-S4 temporal-prior axis.

| File | OOF mean | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `submit_qshead160_s4b650.csv` | `0.565063` | `0.575781` | `0.570034` | `0.575512` | `0.508507` | `0.547219` | `0.526192` | `0.652198` | First upload: push only Q1/Q2/Q3/S1; S2/S3/S4 fixed after public failures. |
| `submit_qshead135_s4b650.csv` | `0.565138` | `0.575982` | `0.570187` | `0.575658` | `0.508527` | `0.547219` | `0.526192` | `0.652198` | Safer fallback if head160 is too aggressive. |
| `submit_qsmicro_s4b650.csv` | `0.565132` | `0.575982` | `0.570187` | `0.575658` | `0.508527` | `0.547191` | `0.526183` | `0.652198` | Small all-Q/S push; use only if head-only direction improves. |
| `submit_qscal_s4b650_fixed.csv` | `0.564909` | `0.574967` | `0.570421` | `0.575815` | `0.508320` | `0.546829` | `0.525813` | `0.652198` | Conservative fallback: keep b650 S4 and apply OOF-fitted Q/S calibration. |

## Rationale

- Full Q/S over-push failed earlier, so S2/S3 are frozen in the first two files.
- The first file moves only Q1/Q2/Q3/S1 where OOF still improves along the temporal axis.
- All output CSVs use fixed-format sample keys and 10-decimal probabilities to avoid submission Data Error.

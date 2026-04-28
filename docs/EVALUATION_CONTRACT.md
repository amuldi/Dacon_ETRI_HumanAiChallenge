# Evaluation Contract

## Split Families

### `group_time`

- Primary model-selection split.
- For each subject, sort rows by `sleep_date`.
- Use a rolling-origin validation layout:
  - first 40%: history only
  - next 20%: `fold_id = 0`
  - next 20%: `fold_id = 1`
  - last 20%: `fold_id = 2`
- Only rows in the later 60% are primary scored rows.
- Outer-fold training excludes future rows from the same subject.

### `group`

- Secondary guardrail split.
- 5-fold whole-subject holdout via `GroupKFold`.
- Used to detect subject leakage or over-specialization.

## Metric

The implemented metric is equal-weight macro binary log-loss over the 7 targets:

- `Q1`, `Q2`, `Q3`, `S1`, `S2`, `S3`, `S4`

Each target uses clipped probabilities and standard binary log-loss. The overall score is the arithmetic mean of the 7 per-target losses.

## Calibration

- Per-target Platt scaling on fold-internal calibration splits.
- `group_time`: calibration split is the latest 20% inside each outer training fold.
- `group`: calibration split is a group-aware holdout inside the outer training fold.
- The exported OOF tables include both raw and calibrated probabilities.

## Current Baseline Outputs

- `artifacts/oof/oof_predictions_group_time.parquet`
- `artifacts/oof/oof_predictions_group.parquet`
- `artifacts/experiments/baseline_group_time.json`
- `artifacts/experiments/baseline_group.json`

## Current Scores

- `group_time` calibrated mean log-loss: `0.6850430377`
- `group` calibrated mean log-loss: `0.7633749255`


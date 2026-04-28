# Data Contract

## Row Grain

- Training labels live at one row per `subject_id`, `sleep_date`, `lifelog_date`.
- The enforced rule is `lifelog_date = sleep_date - 1 day`.
- The generated feature table keeps one row per `subject_id`, `sleep_date`, `lifelog_date`.

## Source Files

- Labels: `data/ch2026_metrics_train.csv`
- Submission template: `data/ch2026_submission_sample.csv`
- Raw modalities: `data/ch2025_data_items/*.parquet`

## Modality Rules

- Scalar modalities are aggregated directly to day:
  - `mACStatus`, `mActivity`, `mLight`, `mScreenStatus`, `wLight`, `wPedo`
- Nested modalities are summarized per row first, then aggregated to day:
  - `mAmbience`, `mBle`, `mGps`, `mUsageStats`, `mWifi`, `wHr`
- Daily features may contain NaNs for genuinely unobserved modalities.
- Every modality emits a `__record_count` feature and a `__missing_day` flag.

## Leakage Rules

- Features for a target row may only use raw events from that exact `lifelog_date` or earlier.
- Personal history features are built from prior days only via shifted rolling or expanding windows.
- `group_time` validation excludes future rows from the same subject during outer-fold training.

## Current Artifacts

- Schema contract report: `reports/contracts/schema_contract.md`
- Machine-readable schema contract: `artifacts/contracts/schema_contract.json`
- Daily feature table: `artifacts/features/daily_feature_table.parquet`


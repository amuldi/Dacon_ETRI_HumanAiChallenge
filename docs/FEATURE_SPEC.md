# Feature Spec

## Current Daily Table

- Unit: `1 row = 1 subject_id, sleep_date`
- Join keys: `subject_id`, `sleep_date`, `lifelog_date`
- Output: numeric features + missingness flags only
- Current artifact target: `artifacts/features/daily_feature_table.parquet`

## Implemented Feature Families

- Same-day modality summaries
  - per-modality row count, coverage ratio, bucket coverage
  - modality-specific aggregates such as screen-on, charging, pedometer, heart-rate, GPS, WiFi, BLE, usage, ambience, light
- Time-bucket behavior summaries
  - overnight / morning / afternoon / evening shares or aggregates where supported
- Missingness and observation quality
  - `miss_<prefix>_any`
  - `miss_days_since_last_<prefix>`
  - per-modality bucket utilization, rows-per-bucket, rows-per-covered-hour
  - global observed modality count, missing modality count, mean coverage
- Cross-modality context
  - screen-charge coupling
  - mobility-cardio and mobility-extent interactions
  - usage-screen density
  - night screen vs charge / light interactions
- Subject-relative temporal context on curated anchors
  - `__lag1`
  - `__expanding_mean`
  - `__roll_mean_{3,7,14}`
  - `__roll_std_{7,14}`
  - `__delta_{3,7,14}`
  - `__delta_expanding`
  - `__zscore_{7,14}`
  - `__ratio_{3,14}`
  - `__recent_vs_history_gap`
  - `__recent_vs_history_ratio`
  - `__slope_{7,14}`
  - `__cv_14`

## Current Notes

- The table intentionally over-generates candidate tabular features so the baseline can later prune to a smaller feature view.
- The strongest current OOF baseline still depends heavily on smoothed subject prior, so these features are best treated as a candidate pool rather than a proven winning subset.
- If a future model family is added, prefer reusing the same daily table and selecting a narrower feature view instead of creating a separate preprocessing branch.

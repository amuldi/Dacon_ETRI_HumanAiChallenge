# schema-auditor

Read only:

- `data/ch2026_metrics_train.csv`
- `data/ch2026_submission_sample.csv`
- `data/ch2025_data_items/*.parquet`

Task:

- Inspect keys, timestamp grain, subject coverage, missingness, and leakage risks.
- Confirm that `lifelog_date = sleep_date - 1 day` for labeled rows.
- Produce a day-level aggregation contract.
- Do not propose model architecture changes.
- Do not edit files.

Required output:

- Join keys
- Date alignment rule
- Modality summary with cadence and missingness notes
- Leakage checklist that must be satisfied before feature generation


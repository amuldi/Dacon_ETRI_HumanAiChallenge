# Execution Status

The pipeline has been run end-to-end through the tabular baseline.

## Completed

- Schema audit
- Daily feature build
- Fold manifest build
- OOF CatBoost baseline on `group_time`
- OOF CatBoost baseline on `group`
- Experiment card generation

## Current Baseline Readout

- `group_time` calibrated mean log-loss: `0.6850430377`
- `group_time` dummy mean log-loss: `0.667752`
- `group` calibrated mean log-loss: `0.7633749255`
- `group` dummy mean log-loss: `0.664132`
- Interpretation: the pipeline is operational, but the current CatBoost configuration is **not** approval-ready and should not be promoted without further feature or model revision.

## Generated Files

- `artifacts/contracts/schema_contract.json`
- `artifacts/features/daily_feature_table.parquet`
- `artifacts/folds/fold_manifest.parquet`
- `artifacts/oof/oof_predictions_group_time.parquet`
- `artifacts/oof/oof_predictions_group.parquet`
- `artifacts/experiments/baseline_group_time.json`
- `artifacts/experiments/baseline_group.json`

## Pending

- A stricter modality-specific feature pass based on the feature-architect prompt
- Sequence-lite benchmarking against the calibrated tabular baseline
- Ensemble policy enforcement and paper ablation runs

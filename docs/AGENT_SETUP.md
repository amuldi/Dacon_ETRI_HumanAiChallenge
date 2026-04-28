# Agent Setup

This workspace is organized around five roles instead of file ownership.

| Agent | Responsibility | First Output |
|---|---|---|
| `schema-auditor` | raw schema, time alignment, missingness, leakage | data contract |
| `feature-architect` | daily feature schema and priority | feature spec |
| `baseline-modeler` | OOF CatBoost baseline | first OOF report |
| `sequence-lite` | short-window lightweight sequence experiments | sequence report |
| `validation-paper` | folds, calibration, experiment cards, paper hypotheses | evaluation contract |

## Shared Interfaces

- `daily_feature_table`
  - row grain: `subject_id, sleep_date, lifelog_date`
  - remaining columns: numeric features, missingness flags, targets for train rows
- `fold_manifest`
  - keys plus `split_scheme`, `fold_id`, `role`
- `oof_prediction_table`
  - keys, targets, raw probabilities, calibrated probabilities
- `experiment_card`
  - model family, feature view, split scheme, calibrated score, target scores, acceptance flag

## Operating Order

1. Run `schema-auditor` and `validation-paper`.
2. Merge both into the project contracts.
3. Run `feature-architect`.
4. Build `daily_feature_table`.
5. Run `baseline-modeler`.
6. Open `sequence-lite` only if it beats the calibrated tabular baseline.


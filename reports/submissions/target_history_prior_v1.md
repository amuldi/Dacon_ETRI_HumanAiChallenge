# Submission Report: target_history_prior_v1

- Anchor public: `0.5956332255`
- Anchor OOF: `0.568769`
- Primary OOF: `0.568769`
- Conservative OOF: `0.568769`

## Primary Selection

| Target | Source | Weight | OOF | Prior OOF | Config |
|---|---|---:|---:|---:|---|
| Q1 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.589710 | 0.635262 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| Q2 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.579744 | 0.599062 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.0, 'future_scale': 1.0}` |
| Q3 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.585794 | 0.626914 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S1 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.516004 | 0.591955 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S2 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.560431 | 0.573016 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S3 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.538894 | 0.554240 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S4 | `target_history_prior_v1` | 0.02 | 0.610807 | 0.644167 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |

## Conservative Selection

| Target | Source | Weight | OOF | Prior OOF | Config |
|---|---|---:|---:|---:|---|
| Q1 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.589710 | 0.635262 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| Q2 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.579744 | 0.599062 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.0, 'future_scale': 1.0}` |
| Q3 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.585794 | 0.626914 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S1 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.516004 | 0.591955 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S2 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.560431 | 0.573016 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S3 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.538894 | 0.554240 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |
| S4 | `public_lgb_targetwise_guarded_v2_stable_tuned` | 0.00 | 0.610809 | 0.644167 | `{'max_days': 28, 'power': 1.25, 'subject_weight': 0.15, 'future_scale': 1.0}` |

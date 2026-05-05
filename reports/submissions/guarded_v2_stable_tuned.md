# Submission Report: guarded_v2_stable_tuned

- Current best public: `0.5960566585`
- Current best OOF: `0.572196`
- Primary OOF: `0.568769`
- Conservative OOF: `0.569143`

## Primary Selection

| Target | Candidate | Features | OOF | Source |
|---|---|---:|---:|---|
| Q1 | `top300` | 300 | 0.589710 | guarded_v2_stable_tuned |
| Q2 | `top250` | 250 | 0.579744 | guarded_v2_stable_tuned |
| Q3 | `top250` | 250 | 0.585794 | guarded_v2_stable_tuned |
| S1 | `top300` | 300 | 0.516004 | guarded_v2_stable_tuned |
| S2 | `top400` | 377 | 0.560431 | guarded_v2_stable_tuned |
| S3 | `top300` | 300 | 0.538894 | guarded_v2_stable_tuned |
| S4 | `baseline` | 411 | 0.610809 | public_lgb_targetwise_histmix_guarded_v1 |

## Conservative Selection

| Target | Candidate | Features | OOF | Source |
|---|---|---:|---:|---|
| Q1 | `baseline` | 556 | 0.590830 | public_lgb_targetwise_histmix_guarded_v1 |
| Q2 | `top250` | 250 | 0.579744 | guarded_v2_stable_tuned |
| Q3 | `baseline` | 365 | 0.587291 | public_lgb_targetwise_histmix_guarded_v1 |
| S1 | `top300` | 300 | 0.516004 | guarded_v2_stable_tuned |
| S2 | `top400` | 377 | 0.560431 | guarded_v2_stable_tuned |
| S3 | `top300` | 300 | 0.538894 | guarded_v2_stable_tuned |
| S4 | `baseline` | 411 | 0.610809 | public_lgb_targetwise_histmix_guarded_v1 |

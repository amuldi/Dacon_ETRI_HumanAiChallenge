# Experiment Log

## 2026-04-27 Softblend Reproduce

Current best public score: `0.5960566585`  
Current best submission: `submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv`

Validation artifact note: the cached OOF files used for this assembly are marked `public_stratified`. The requested submissions preserve the current best public pipeline and do not retrain models.

## OOF Summary

| Experiment | OOF mean | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| histmix_guarded_v1_reproduce | 0.572196 | 0.590830 | 0.585800 | 0.587291 | 0.524007 | 0.562788 | 0.543849 | 0.610809 |
| softblend_w090 | 0.572147 | 0.590830 | 0.585544 | 0.587205 | 0.524007 | 0.562788 | 0.543849 | 0.610809 |
| softblend_w085 | 0.572139 | 0.590830 | 0.585496 | 0.587198 | 0.524007 | 0.562788 | 0.543849 | 0.610809 |
| softblend_w095 | 0.572166 | 0.590830 | 0.585646 | 0.587236 | 0.524007 | 0.562788 | 0.543849 | 0.610809 |

## Target-Wise View

| Target | histmix_guarded_v1_reproduce | softblend_w090 | softblend_w085 | softblend_w095 |
|---|---|---|---|---|
| Q1 | public_core | public_core | public_core | public_core |
| Q2 | public_hist411 | 0.90 public_hist411 + 0.10 public_core | 0.85 public_hist411 + 0.15 public_core | 0.95 public_hist411 + 0.05 public_core |
| Q3 | public_hist365 | 0.90 public_hist365 + 0.10 public_core | 0.85 public_hist365 + 0.15 public_core | 0.95 public_hist365 + 0.05 public_core |
| S1 | public_core | public_core | public_core | public_core |
| S2 | public_core | public_core | public_core | public_core |
| S3 | public_core | public_core | public_core | public_core |
| S4 | public_hist411 | public_hist411 | public_hist411 | public_hist411 |

## Generated Files

| Experiment | Submission | Stability |
|---|---|---|
| softblend_w090 | `submissions/archive/softblend_initial/submission_softblend_w090.csv` | `logs/stability/stability_softblend_w090.csv` |
| softblend_w085 | `submissions/archive/softblend_initial/submission_softblend_w085.csv` | `logs/stability/stability_softblend_w085.csv` |
| softblend_w095 | `submissions/archive/softblend_initial/submission_softblend_w095.csv` | `logs/stability/stability_softblend_w095.csv` |
| histmix_guarded_v1_reproduce | `artifacts/submissions/submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv` | `logs/stability/stability_histmix_guarded_v1_reproduce.csv` |

Automatic experiment log: `logs/experiments.csv`

## Recommended Submission Order

1. `softblend_w090`
2. `softblend_w085`
3. `softblend_w095`
4. `histmix_guarded_v1_reproduce`

## 2026-04-28 Public Score Memo

| Submission | Public score | Delta vs best | Decision |
|---|---:|---:|---|
| `submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv` | 0.5960566585 | 0.0000000000 | Current best anchor |
| `submission_public_lgb_v4_softblend_w090.csv` | 0.5962890684 | +0.0002324099 | Worse; stop softblend direction |
| `submission_public_lgb_v5_public_lgb_targetwise_histmix_guarded_v1_subject_holdout.csv` | 0.5968333841 | +0.0007767256 | Worse; do not promote |

Interpretation: the subject-holdout retrain improves validation realism but does not improve the public leaderboard as a standalone submission. The first softblend probe also worsened public score, so moving Q2/Q3 back toward `public_core` is not supported by public feedback. Keep the original public-stratified `histmix_guarded_v1` family as the scoring anchor.

## Clean Submission Folder

Short, readable candidate files are collected under `submissions/ready/`.

| Rank | File | OOF mean | Public score | Notes |
|---:|---|---:|---:|---|
| 0 | `00_best_guarded.csv` | 0.572196 | 0.5960566585 | Reference current best |
| 1 | `next_guarded_s2s3.csv` | 0.572191 |  | Next low-risk probe: only S2/S3 tiny holdout blend |
| 2 | `01_soft_w090.csv` | 0.572147 | 0.5962890684 | Submitted and worse; stop this direction |
| 3 | `02_soft_w090_s2s3.csv` | 0.572142 |  | Demoted because it inherits failed Q2/Q3 softblend |
| 4 | `03_soft_w085_s2s3.csv` | 0.572134 |  | Demoted because it moves farther in failed softblend direction |
| 5 | `04_soft_w085.csv` | 0.572139 |  | Do not submit for now |
| 6 | `05_soft_w095.csv` | 0.572166 |  | Do not submit for now |

Recommended next submission order:

1. `submissions/ready/next_guarded_s2s3.csv`
2. stop if it does not beat `0.5960566585`

Additional logs:

- `logs/public_scores.csv`
- `logs/candidate_scores.csv`
- `submissions/README.md`
- `logs/README.md`
- `PROJECT_STRUCTURE.md`

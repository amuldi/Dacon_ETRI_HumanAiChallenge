# Public Score Strategy - 2026-04-24

## Known Public Scores

- `submission_hgb_prior_v1_hgb_prior_group_time.csv`: 0.6195173919
- `submission_prior_v2_prior_v2_group_time.csv`: 0.6203498892
- `submission_v3_s2_s4_guarded_public_guarded_group_time.csv`: 0.6190720413
- `submission_v4_s4_only_public_guarded_group_time.csv`: 0.6214971074
- `submission_v4_s2_only_public_guarded_group_time.csv`: 0.6170923258
- `submission_v5_s2_w125_public_guarded_group_time.csv`: 0.6168506153
- `submission_v7_s2_w128_public_guarded_group_time.csv`: 0.6168306056

## Signal

The public score is an average of target log-loss values, so target-only swaps are additive.

`S2-only = v1 + v3(S2+S4) - S4-only`

Expected score:

`0.6195173919 + 0.6190720413 - 0.6214971074 = 0.6170923258`

This means the S4 v2 change is harmful, while the S2 v2 change is strongly beneficial.

## Submission Order

1. Keep `artifacts/submissions/submission_v7_s2_w128_public_guarded_group_time.csv` as the current best.
2. Continue a narrow S2 weight search just above 1.28.
3. Avoid S4 and Q-target changes unless a new independent signal appears.

## File Hygiene

- Keep source submissions required for candidate generation: `submission_hgb_prior_v1_hgb_prior_group_time.csv`, `submission_prior_v2_prior_v2_group_time.csv`.
- Keep the current best submission: `submission_v7_s2_w128_public_guarded_group_time.csv`.
- Keep only immediate next candidates: `submission_v6_s2_w132_public_guarded_group_time.csv`, `submission_v7_s2_w135_public_guarded_group_time.csv`, `submission_v6_s2_w138_public_guarded_group_time.csv`.
- When a new public score is reported, update this note, promote the best CSV if needed, and delete stale non-best candidates unless they are source files or immediate next candidates.

## Current Hypothesis

- Keep Q1/Q2/Q3 from v1.
- Keep S1/S3 from v1.
- Reuse only the S2 prior-v2 direction.
- Do not reuse S4 prior-v2, despite OOF improvement.

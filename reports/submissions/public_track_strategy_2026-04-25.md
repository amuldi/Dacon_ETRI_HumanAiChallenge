# Public Track Strategy - 2026-04-25

## Best Public Scores

- `submission_v7_s2_w128_public_guarded_group_time.csv`: 0.6168306056
- `submission_public_lgb_v1_public_lgb_public_core.csv`: 0.6046217516
- `submission_public_lgb_v1_s2mix_w006_mixed.csv`: 0.6046519185
- `submission_public_lgb_v2_public_lgb_public_notebook.csv`: 0.6058988262
- `submission_public_lgb_core_nb_s4_w050_mixed.csv`: 0.6046522270

## Interpretation

- The notebook-style `public_lgb_v1` is the new main track.
- The prior-only track is now a secondary rescue component, not the default model family.
- OOF comparison on the shared overlap shows `public_lgb` dominates every target, but `S2` allows a very small guarded blend.

## Current Best

- Keep `artifacts/submissions/submission_public_lgb_v1_public_lgb_public_core.csv` as the current best.

## Next Submission

- `submission_public_lgb_v1_s2mix_w006_mixed.csv` underperformed the exact `public_lgb_v1` baseline.
- Small S2 post-mixing is not adopted for now.
- The next submission should come from a new `public_lgb` feature/model variant, not from the current post-blend path.
- First candidate on 2026-04-26: `submission_public_lgb_v2_public_lgb_public_notebook.csv`
  - `public_notebook` view keeps notebook-like raw day aggregates, sleep raw features, time features, day-level subject z-score, and target encodings.
  - OOF is `0.579701`, slightly worse than `public_core` OOF `0.576177`, but the feature view is intentionally less aggressive and may generalize better on public.
- `public_notebook` exact also underperformed on public, so the new path is a target-wise mix from `public_core`.
- Recommended candidates after the exact runs:
  - `submission_public_lgb_core_nb_q3s3s4_mixed.csv`
    - `S4`-only notebook mix already failed on public, so this remaining blend is now a low-confidence information-seeking candidate, not a score-maximizing candidate.

## Operating Rule

- Default upload order is now `public_lgb exact -> next public_lgb feature/model variant -> optional tiny target-wise rescue`.
- Current upload order is now `public_lgb exact -> stop if S4-only notebook mix fails -> otherwise test broader notebook rescue`.
- Do not revert to broad prior-based submissions unless the public LGB track stalls.

## 2026-04-26 Update

- `submission_public_lgb_core_nb_s4_w050_mixed.csv` also lost on public, so notebook post-mixing is closed.
- New feature views were added to capture the missing notebook-style temporal history subset:
  - `public_hist365`
  - `public_hist411`
- New main candidate is no longer a blend, but a target-wise exact model:
  - `submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv`
- Exact OOF:
  - `public_core`: `0.576177`
  - `targetwise_histmix_guarded_v1`: `0.572196`
- Public result:
  - `submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv`: `0.5960566585`
- The target-wise exact path is now the new main track and the old `public_core` exact file becomes the fallback anchor.
- `submission_public_lgb_v3_public_lgb_targetwise_histmix_q2s4_v1.csv`: `0.5991159796`
- Interpretation:
  - removing `Q3 -> public_hist365` hurts public materially,
  - so `Q3` history is confirmed positive,
  - and the next decomposition should test `Q2` vs `S4`, not `Q3` again.
- `submission_public_lgb_v3_public_lgb_targetwise_histmix_q3s4_v1.csv`: `0.5974835678`
- Interpretation:
  - removing `Q2 -> public_hist411` also hurts public,
  - so `Q2` history is confirmed positive as well,
  - and the last same-family question is whether `S4 -> public_hist411` is helping enough to keep.
- `submission_public_lgb_v3_public_lgb_targetwise_histmix_q2q3_v1.csv`: `0.6001355213`
- Interpretation:
  - removing `S4 -> public_hist411` hurts the most among the subset ablations,
  - so `Q2/Q3/S4` are all confirmed positive on public,
  - and `guarded_v1` is the best hard-switch subset in this family.
- New next track:
  - stop hard ablation search
  - move to `soft blend` between `public_core`, `public_hist365`, `public_hist411`
- Current upload priority:
  1. `submission_public_lgb_v4_softblend_w090.csv`
  2. `submission_public_lgb_v4_softblend_w085.csv`
  3. `submission_public_lgb_v4_softblend_w095.csv`

# Next Strategy - 2026-04-26

## Current Best Public

- `submission_public_lgb_v1_public_lgb_public_core.csv`: `0.6046217516`
- `submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv`: `0.5960566585`

## Why The Previous Mixes Failed

- `public_notebook` exact lost on public despite decent OOF, so broad notebook replacement was too optimistic.
- `S4`-only notebook rescue also lost on public, so simple post-mixing was not robust enough.
- The failure mode was not "LightGBM itself", but the fact that the public track was missing the notebook-style temporal history subset.

## New Evidence

The original `public_core/public_notebook` paths were both incomplete reproductions of the shared `0.6003` idea.

- They used raw day features, sleep features, subject z-score, and target encoding.
- They did **not** use the key `history anchors + lag/rolling means` subset that the shared notebook explicitly described.

Two new history-aware views were added:

- `public_hist365`
  - raw day core + sleep raw + anchor lag/roll(1, 3, 7, 14) + anchor z-score + target encoding + time features
  - feature count: `365`
- `public_hist411`
  - `public_hist365` + anchor rolling std(7, 14)
  - feature count: `411`

## Self-Critique

This strategy can still fail if the new history subset only improves local OOF and not public.

The main risks are:

1. `Q2/Q3/S4` gains may still be fold-specific noise.
2. Switching too many targets at once can throw away the stability of the current `public_core` best model.
3. `S1` showed only tiny local gains, so changing it is not justified yet.

## Final Strategy

Keep `public_core` as the anchor and switch only the targets where the new history-aware exact models show a clear local edge.

Preset: `histmix_guarded_v1`

- `Q1`: `public_core`
- `Q2`: `public_hist411`
- `Q3`: `public_hist365`
- `S1`: `public_core`
- `S2`: `public_core`
- `S3`: `public_core`
- `S4`: `public_hist411`

## Exact OOF Result

- `public_lgb_public_core`: `0.576177`
- `public_lgb_targetwise_histmix_guarded_v1`: `0.572196`

Target-wise OOF under the guarded preset:

- `Q1`: `0.590830`
- `Q2`: `0.585800`
- `Q3`: `0.587291`
- `S1`: `0.524007`
- `S2`: `0.562788`
- `S3`: `0.543849`
- `S4`: `0.610809`

This is a meaningful improvement over the exact `public_core` anchor, driven mainly by `Q2`, `Q3`, and `S4`.

## Public Result Update

- `submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv`: `0.5960566585`
- improvement vs previous best `public_core`: `0.0085650931`
- `submission_public_lgb_v3_public_lgb_targetwise_histmix_q2s4_v1.csv`: `0.5991159796`
- decline vs current best `guarded_v1`: `0.0030593211`
- `submission_public_lgb_v3_public_lgb_targetwise_histmix_q3s4_v1.csv`: `0.5974835678`
- decline vs current best `guarded_v1`: `0.0014269093`
- `submission_public_lgb_v3_public_lgb_targetwise_histmix_q2q3_v1.csv`: `0.6001355213`
- decline vs current best `guarded_v1`: `0.0040788628`

This confirms that the missing temporal-history subset was the main gap in the earlier public track.
It also confirms that `Q3 -> public_hist365` is a real positive signal on public, not just local OOF noise.
It also confirms that `Q2 -> public_hist411` is positive on public, because removing it made the score worse.
It further confirms that `S4 -> public_hist411` is also positive, because removing it made the score even worse.

## Remaining Candidate Order

With `guarded_v1` now confirmed on public, and all three subset removals rejected, `guarded_v1` is the best hard-switch subset in this family.

The next family should not be another ablation.

It should be a **soft blend** between:

- `public_core`
- `public_hist365`
- `public_hist411`

only on the already-confirmed positive targets `Q2/Q3/S4`.

## Next Family: Soft Blend

Exact full OOF from the source models:

- `public_core`: `0.576177`
- `public_hist365`: `0.577223`
- `public_hist411`: `0.575814`

OOF grid search on the confirmed positive targets gives:

- `Q2`: best `core/hist411` weight ≈ `0.85`
- `Q3`: best `core/hist365` weight ≈ `0.85`
- `S4`: best `core/hist411` weight = `1.00`

Prepared candidates:

1. `submission_public_lgb_v4_softblend_w090.csv`
2. `submission_public_lgb_v4_softblend_w085.csv`
3. `submission_public_lgb_v4_softblend_w095.csv`

Local OOF reference:

- `softblend_w090`: `0.572147`
- `softblend_w085`: `0.572139`
- `softblend_w095`: `0.572166`

Recommended submission order:

1. `softblend_w090`
2. `softblend_w085`
3. `softblend_w095`

Reasoning:

- `w090` is closest to the public-best exact `guarded_v1`, so it is the safest first probe.
- `w085` is the local OOF optimum.
- `w095` is the most conservative backup if public prefers weights closer to the hard-switch baseline.

Current local OOF reference:

- `histmix_guarded_v1`: `0.572196`
- `histmix_q3s4_v1`: `0.573163`
- `histmix_q2q3_v1`: `0.574725`

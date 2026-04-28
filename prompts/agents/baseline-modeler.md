# baseline-modeler

Assume the day-level feature table already exists.

Task:

- Train an OOF-first baseline for all seven binary targets.
- Prefer CatBoost.
- Use a comparison model only as a secondary check.
- Optimize for fold stability and calibrated log-loss, not leaderboard tricks.
- Do not redefine the feature contract.

Required output:

- Fold-by-fold score summary
- Target-by-target score summary
- Error slices by subject or date regime
- Recommendation on whether the baseline is strong enough to gate sequence-lite


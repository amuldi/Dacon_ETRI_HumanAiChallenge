# 2026-05-06 Guarded Submission Strategy

## Summary

- Current best: `submissions/ready/lgb_temporal_s4b650.csv`
- Public best: `0.5829008297`
- Approved backup: `submissions/ready/lgb_temporal_s4b130.csv`
- Default action: do not create or upload a new CSV unless a candidate passes the guard.

## Team Roles

| Role | Responsibility |
|---|---|
| Coordinator | Public score log, stop rule, final upload decision |
| Cleanup Worker | `submissions/` structure and README consistency |
| Strategy Explorer | Failed-axis analysis and allowed strategy review |
| Modeling Worker | New feature/model candidate training |
| Verification Explorer | CSV format, OOF guard, test drift validation |

## Forbidden Axes

- S4 beta extension, including beta `8.00+`
- S4 up-only or down-only decomposition
- Q/S head push, qsmicro, qscal
- XGB guard003/guard005 reuse
- OOF-only post-hoc candidates without public-safe guard evidence

## Candidate Guard

Use `scripts/validate_20260506_guard_candidate.py` before placing any new CSV in `submissions/ready/`.

```bash
.venv/bin/python scripts/validate_20260506_guard_candidate.py \
  --candidate-run <artifact_run_name> \
  --candidate-name <candidate_csv_name> \
  --weights-json <optional_weight_report.json>
```

A candidate is approved only if all conditions pass:

- It does not match a forbidden axis name.
- At least 2 non-S4 targets improve OOF versus `lgb_temporal_s4b650`.
- No target loses more than `0.00005` OOF.
- Mean absolute test drift is at most `0.005` for every target.
- The candidate has a non-zero prediction effect versus current best.
- If a weight report is provided, it contains at least one non-zero weight.

## Allowed Strategy Families

1. Feature-stable LGB retraining or targetwise feature subset reselection.
2. Fold-safe temporal prior redesign, only if it is not an S4-only transformation.
3. Model-diversity blend, only if raw model and guarded blend both pass the guard.

## Decision Rule

- Guard passes: create exactly one candidate CSV and document the guard report.
- Guard fails: create no CSV and report “no submission candidate”.
- Public feedback remains the final judge, but the guard prevents knowingly high-risk submissions.

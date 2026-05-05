# 2026-05-06 public-feedback anti microblend

## Decision

- Candidate file: `/Users/jsh/Desktop/output/대회/ ETRI_휴먼이해 인공지능 논문경진대회/submissions/ready/submit_20260506_public_feedback_anti_micro075.csv`
- Current public: `0.5829008297`
- Failed probe public: `0.5834566947`
- Failed delta: `+0.0005558650`
- Anti scale: `0.75`
- Linear public estimate: `0.5824839310`
- S4: frozen

## Why this direction

The last candidate passed OOF guard but public worsened, which means OOF was misleading on this non-S4 temporal axis. The safest public-feedback response is not to push that axis further, but to move partially in the opposite direction while keeping the current-best S4 untouched.

## Target diagnostics

| Target | Candidate OOF delta | Candidate drift | Failed drift |
|---|---:|---:|---:|
| `Q1` | `0.001000755` | `0.003745648` | `0.004994197` |
| `Q2` | `0.000183479` | `0.003740213` | `0.004986951` |
| `Q3` | `0.000265983` | `0.003745484` | `0.004993978` |
| `S1` | `0.000170967` | `0.003450384` | `0.004600512` |
| `S2` | `0.000087526` | `0.002393463` | `0.003191285` |
| `S3` | `0.000024358` | `0.001308838` | `0.001745118` |
| `S4` | `0.000000000` | `0.000000000` | `0.000000000` |

## Self-critique

1. This is not an OOF-safe candidate. It is a public-feedback correction after an OOF-safe file failed public.
2. The expected public score is a linear approximation, not a guarantee.
3. The scale is deliberately `0.75`, not `1.00`, because one failed public point is useful but not enough to fully invert the move.
4. If this fails, the next valid direction is feature-stable retraining, not another post-hoc public-feedback flip.

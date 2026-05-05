# 2026-05-06 public-feedback anti microblend (2.25)

## Decision

- Candidate file: `/Users/jsh/Desktop/output/대회/ ETRI_휴먼이해 인공지능 논문경진대회/submissions/ready/submit_20260506_public_feedback_anti_micro225.csv`
- Current public: `0.5829008297`
- Failed probe public: `0.5834566947`
- Failed delta: `+0.0005558650`
- Anti scale: `2.25`
- Linear public estimate: `0.5816501335`
- 3-point quadratic public estimate: `0.5823114601`
- 3-point quadratic vertex scale: `2.573`
- S4: frozen

## Why this direction

The last candidate passed OOF guard but public worsened, which means OOF was misleading on this non-S4 temporal axis. The first public-feedback anti move at scale 0.75 improved public from 0.5829008297 to 0.5826026306. Using the three observed points, the quadratic public estimate places the local best near scale 2.57, so this candidate tests a stronger but still sub-vertex scale while keeping S4 untouched.

## Target diagnostics

| Target | Candidate OOF delta | Candidate drift | Failed drift |
|---|---:|---:|---:|
| `Q1` | `0.004174948` | `0.011236943` | `0.004994197` |
| `Q2` | `0.000631319` | `0.011220640` | `0.004986951` |
| `Q3` | `0.000935792` | `0.011236451` | `0.004993978` |
| `S1` | `0.000805655` | `0.010351151` | `0.004600512` |
| `S2` | `0.000403743` | `0.007180390` | `0.003191285` |
| `S3` | `0.000110930` | `0.003926515` | `0.001745118` |
| `S4` | `0.000000000` | `0.000000000` | `0.000000000` |

## Self-critique

1. This is not an OOF-safe candidate. It is a public-feedback correction after an OOF-safe file failed public.
2. The public score estimate is based on only three submissions, so it can overfit the public leaderboard.
3. The scale is extrapolated from public feedback and still relies on a low-data approximation.
4. If this fails, the next valid direction is feature-stable retraining, not another post-hoc public-feedback flip.

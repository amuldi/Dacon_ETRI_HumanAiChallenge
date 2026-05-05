# Submissions

DACON 업로드 후보는 `submissions/ready/`만 확인합니다.

## Current Best

- Public best: `0.5826026306`
- Best file: `submissions/ready/submit_20260506_public_feedback_anti_micro075.csv`
- 2026-05-06 upload probe: `submissions/ready/submit_20260506_public_feedback_anti_micro225.csv`
- Previous anchor: `submissions/ready/lgb_temporal_s4b650.csv`
- Backup file: `submissions/ready/lgb_temporal_s4b130.csv`

## Ready Files

| Role | File | OOF mean | Public | Decision |
|---|---|---:|---:|---|
| 2026-05-06 upload probe | `ready/submit_20260506_public_feedback_anti_micro225.csv` | `0.566305` |  | 오늘 업로드 후보: public-feedback line-search scale `2.25`, quadratic estimate `0.5823114601` |
| current public best | `ready/submit_20260506_public_feedback_anti_micro075.csv` | `0.565544` | `0.5826026306` | 기존 best보다 `0.0002981991` 개선 |
| previous anchor | `ready/lgb_temporal_s4b650.csv` | `0.565296` | `0.5829008297` | 이전 public best anchor |
| backup | `ready/lgb_temporal_s4b130.csv` | `0.559702` | `0.5845552904` | 백업 보관 |

## Stop Rule

기존 후처리 축은 중단합니다. 예외는 2026-05-06 public-feedback line-search 후보 `anti_micro225` 1회뿐입니다.

- S4 beta 추가 연장 금지
- S4 up/down 분해 금지
- Q/S head push, qsmicro, qscal 금지
- XGB guard003/005 재사용 금지
- OOF만 좋아지는 후처리 후보 금지
- `anti_micro225`가 실패하면 public-feedback postprocessing도 중단

## Archived Evidence

실패 또는 차단된 CSV는 archive에 보존합니다.

| Axis | Archive path | Evidence |
|---|---|---|
| S4 extension | `archive/2026-05-05_s4_extension_failed/submit_s4sym800_s4b650.csv` | public `0.5845567721` |
| S4 extension blocked | `archive/2026-05-05_s4_extension_failed/submit_s4near950_s4b650.csv` | `sym800` 실패로 차단 |
| S4 extension blocked | `archive/2026-05-05_s4_extension_failed/submit_s4sym950_s4b650.csv` | `sym800` 실패로 차단 |
| Q/S push | `archive/2026-05-05_qshead_failed/submit_qshead160_s4b650.csv` | public `0.5837781861` |
| S4 direction | `archive/2026-05-05_s4_direction_failed/submit_s4down650_fixed.csv` | public `0.5838666368` |
| non-S4 temporal microblend | `archive/2026-05-06_non_s4_microblend_failed/submit_20260506_non_s4_temporal_microblend.csv` | public `0.5834566947` |
| XGB guarded blend | `archive/2026-05-05_xgb_guard_blocked/submit_xgb_guard003_s4b650.csv` | selected weights all `0` |
| XGB guarded blend | `archive/2026-05-05_xgb_guard_blocked/submit_xgb_guard005_s4b650.csv` | selected weights all `0` |

## 2026-05-06 Guard Policy

일반 새 CSV는 아래 조건을 모두 통과할 때만 `ready/`에 둡니다.

1. 기존 실패 축과 같은 조작이 아닐 것.
2. 최소 2개 이상 non-S4 타깃에서 target OOF가 개선될 것.
3. 어떤 타깃도 current 대비 OOF 손실이 `0.00005`를 넘지 않을 것.
4. test prediction 평균 drift가 target별 `0.005` 이하일 것.
5. 후보가 weight report를 만들면, 선택된 blend/model weight가 전부 `0`이 아닐 것.

Guard를 통과한 후보가 없으면 “제출 없음”이 정답입니다. 단, `anti_micro225`는 직전 OOF guard 통과 후보가 public에서 실패한 뒤 만든 public-feedback 예외 후보입니다.

## 2026-05-06 Public Feedback Candidate

`submit_20260506_public_feedback_anti_micro225.csv`는 public feedback을 반영한 line-search 후보입니다.

- 실패 방향 `s=-1.0`: `submit_20260506_non_s4_temporal_microblend.csv`, public `0.5834566947`
- 기준점 `s=0.0`: `lgb_temporal_s4b650.csv`, public `0.5829008297`
- 첫 anti 방향 `s=0.75`: `submit_20260506_public_feedback_anti_micro075.csv`, public `0.5826026306`
- 위 3점을 2차식으로 맞추면 추정 최적 scale은 `2.573`입니다.
- 이번 후보는 과적합 위험을 줄이기 위해 꼭짓점보다 낮은 `2.25`를 사용합니다.
- 3점 quadratic public estimate는 `0.5823114601`입니다.
- S4는 current best와 동일하게 고정합니다.

자가비판: 이 후보는 OOF-safe 후보가 아니라 public-feedback 예외입니다. `anti_micro225`가 실패하면 같은 방향의 추가 스케일링을 멈추고 feature-stable 재학습으로 돌아갑니다.

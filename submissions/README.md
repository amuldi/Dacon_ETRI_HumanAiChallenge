# Submissions

DACON 업로드 후보는 `submissions/ready/`만 확인합니다.

## Current Best

- Public best: `0.5829008297`
- Best file: `submissions/ready/lgb_temporal_s4b650.csv`
- Backup file: `submissions/ready/lgb_temporal_s4b130.csv`

## Ready Files

| Role | File | OOF mean | Public | Decision |
|---|---|---:|---:|---|
| current best | `ready/lgb_temporal_s4b650.csv` | `0.565296` | `0.5829008297` | 업로드 기준 파일 |
| backup | `ready/lgb_temporal_s4b130.csv` | `0.559702` | `0.5845552904` | 백업 보관 |

## Stop Rule

현재 후처리 축은 모두 중단합니다.

- S4 beta 추가 연장 금지
- S4 up/down 분해 금지
- Q/S head push, qsmicro, qscal 금지
- XGB guard003/005 재사용 금지
- OOF만 좋아지는 후처리 후보 금지

## Archived Evidence

실패 또는 차단된 CSV는 archive에 보존합니다.

| Axis | Archive path | Evidence |
|---|---|---|
| S4 extension | `archive/2026-05-05_s4_extension_failed/submit_s4sym800_s4b650.csv` | public `0.5845567721` |
| S4 extension blocked | `archive/2026-05-05_s4_extension_failed/submit_s4near950_s4b650.csv` | `sym800` 실패로 차단 |
| S4 extension blocked | `archive/2026-05-05_s4_extension_failed/submit_s4sym950_s4b650.csv` | `sym800` 실패로 차단 |
| Q/S push | `archive/2026-05-05_qshead_failed/submit_qshead160_s4b650.csv` | public `0.5837781861` |
| S4 direction | `archive/2026-05-05_s4_direction_failed/submit_s4down650_fixed.csv` | public `0.5838666368` |
| XGB guarded blend | `archive/2026-05-05_xgb_guard_blocked/submit_xgb_guard003_s4b650.csv` | selected weights all `0` |
| XGB guarded blend | `archive/2026-05-05_xgb_guard_blocked/submit_xgb_guard005_s4b650.csv` | selected weights all `0` |

## 2026-05-06 Guard Policy

새 CSV는 아래 조건을 모두 통과할 때만 `ready/`에 둡니다.

1. 기존 실패 축과 같은 조작이 아닐 것.
2. 최소 2개 이상 non-S4 타깃에서 target OOF가 개선될 것.
3. 어떤 타깃도 current 대비 OOF 손실이 `0.00005`를 넘지 않을 것.
4. test prediction 평균 drift가 target별 `0.005` 이하일 것.
5. 후보가 weight report를 만들면, 선택된 blend/model weight가 전부 `0`이 아닐 것.

Guard를 통과한 후보가 없으면 “제출 없음”이 정답입니다.

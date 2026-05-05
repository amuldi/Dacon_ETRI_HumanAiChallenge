# Project Structure

자주 보는 위치만 먼저 보면 됩니다. DACON 업로드 후보는 `submissions/ready/`만 기준으로 합니다.

## Quick View

```text
.
  submissions/            # 제출 CSV 관리
  logs/                   # public score, 후보 순위, 실험 로그
  reports/submissions/    # 제출 후보별 상세 리포트
  scripts/                # 재현/학습/후보 생성 스크립트
  src/                    # 재사용 Python 패키지 코드
  artifacts/              # OOF/test prediction/model 산출물
  data/                   # 대회 제공 데이터
  docs/directives/        # 구현 지시서와 진단 문서
```

## Submission Folder

```text
submissions/
  README.md
  ready/
    submit_20260506_public_feedback_anti_micro225.csv # 2026-05-06 upload probe
    submit_20260506_public_feedback_anti_micro075.csv # current public best, 0.5826026306
    lgb_temporal_s4b650.csv          # previous anchor, public 0.5829008297
    lgb_temporal_s4b130.csv          # backup, public 0.5845552904
  archive/
    2026-05-06_non_s4_microblend_failed/ # OOF guard-pass but public failed
    2026-05-05_s4_extension_failed/  # S4 추가 연장 실패/차단
    2026-05-05_qshead_failed/        # Q/S recovery 실패/차단
    2026-05-05_s4_direction_failed/  # S4 down-only 진단 실패
    2026-05-05_xgb_guard_blocked/    # XGB guarded blend 차단
    2026-05-04_after_s4up650_failed/ # S4 up-only 및 고위험 ladder
    2026-05-03_after_s4b110/         # 이전 temporal/S4 중간 후보
    2026-05-02_push_failed/          # temporal push 실패
    2026-05-01_after_temporal_prior/ # 이전 calibration/temporal 후보
    2026-04-29_cleanup/              # 과거 softblend/guarded 후보
```

## Score Logs

```text
logs/
  public_scores.csv       # 실제 DACON public 점수
  candidate_scores.csv    # 현재 후보 우선순위와 차단 기록
  experiments.csv         # 자동 실험 로그
  stability/              # seed/fold 안정성 상세
```

## Current Best

| File | Public | OOF | Notes |
|---|---:|---:|---|
| `submissions/ready/submit_20260506_public_feedback_anti_micro225.csv` |  | `0.566305` | 2026-05-06 upload probe, quadratic estimate `0.5823114601` |
| `submissions/ready/submit_20260506_public_feedback_anti_micro075.csv` | `0.5826026306` | `0.565544` | current public best |
| `submissions/ready/lgb_temporal_s4b650.csv` | `0.5829008297` | `0.565296` | previous anchor |
| `submissions/ready/lgb_temporal_s4b130.csv` | `0.5845552904` | `0.559702` | backup |

## Rules

- DACON 업로드 후보는 `submissions/ready/`만 사용합니다.
- 실패한 제출 CSV는 `submissions/archive/`에 보존하고 재제출하지 않습니다.
- `artifacts/`, `logs/`, `reports/`, `scripts/`는 재현과 분석에 필요하므로 정리 대상에서 제외합니다.
- 일반 새 CSV는 guard를 통과할 때만 생성합니다. 통과 후보가 없으면 제출하지 않습니다.
- `anti_micro225`는 2026-05-06 public-feedback 예외 후보입니다. 실패하면 이 postprocessing 방향은 중단합니다.

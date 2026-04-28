# Project Structure

사람이 자주 보는 위치와 파이프라인 내부 산출물을 분리해서 보면 됩니다.

## Quick View

```text
.
  submissions/            # 사람이 제출 파일을 고르는 곳
  logs/                   # public score와 후보 우선순위
  EXPERIMENT_LOG.md       # 실험 흐름 요약
  scripts/                # 실행 스크립트
  src/                    # Python 패키지 코드
  artifacts/              # 학습/예측 원본 산출물, 가급적 직접 수정하지 않음
  reports/                # 자동 생성 리포트
  data/                   # 원본/제공 데이터
  configs/                # 실험 설정
  docs/directives/        # 구현 지시서와 진단 문서
```

## Submission Area

```text
submissions/
  README.md
  ready/
    00_best_guarded.csv        # 현재 best 복사본
    next_guarded_s2s3.csv      # 다음 제출 후보
    01_soft_w090.csv           # 제출 완료, public 악화
    02_soft_w090_s2s3.csv      # 보류
    03_soft_w085_s2s3.csv      # 보류
    04_soft_w085.csv           # 보류
    05_soft_w095.csv           # 보류
  archive/
    softblend_initial/         # 초기에 생성한 긴 이름 softblend 파일
```

현재 제출할 파일:

```text
submissions/ready/next_guarded_s2s3.csv
```

## Score Logs

```text
logs/
  public_scores.csv            # 실제 public 점수 기록
  candidate_scores.csv         # 후보별 OOF/public/판단
  experiments.csv              # 자동 실험 로그
  stability/                   # seed/fold 안정성 상세
```

## Source Of Truth

- 실제 public best는 `logs/public_scores.csv` 기준으로 판단합니다.
- 제출 후보 우선순위는 `logs/candidate_scores.csv`와 `submissions/README.md` 기준으로 판단합니다.
- `artifacts/`는 모델/OOF/test prediction 원본 보관소입니다. 제출할 때 직접 고르지 않습니다.

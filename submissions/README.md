# Submissions

제출할 파일은 기본적으로 `submissions/ready/`만 보면 됩니다.  
`artifacts/submissions/`는 파이프라인 원본 산출물 보관용이고, DACON에 올릴 때는 여기의 짧은 파일명을 사용합니다.

## Folder Tree

```text
submissions/
  README.md
  ready/
    00_best_guarded.csv
    next_guarded_s2s3.csv
    01_soft_w090.csv
    02_soft_w090_s2s3.csv
    03_soft_w085_s2s3.csv
    04_soft_w085.csv
    05_soft_w095.csv
  archive/
    softblend_initial/
      submission_softblend_w085.csv
      submission_softblend_w090.csv
      submission_softblend_w095.csv
```

## What To Submit

| Priority | File | Status | Public | OOF | Decision |
|---:|---|---|---:|---:|---|
| 0 | `ready/00_best_guarded.csv` | current best copy | 0.5960566585 | 0.572196 | 기준 파일 |
| 1 | `ready/next_guarded_s2s3.csv` | next candidate |  | 0.572191 | 다음 제출 |
| 2 | `ready/01_soft_w090.csv` | submitted | 0.5962890684 | 0.572147 | 실패, 중단 |
| 3 | `ready/02_soft_w090_s2s3.csv` | demoted |  | 0.572142 | softblend 실패 방향 포함 |
| 4 | `ready/03_soft_w085_s2s3.csv` | demoted |  | 0.572134 | softblend 실패 방향 포함 |
| 5 | `ready/04_soft_w085.csv` | hold |  | 0.572139 | 제출 보류 |
| 6 | `ready/05_soft_w095.csv` | hold |  | 0.572166 | 제출 보류 |

## Current Recommendation

```text
submit: submissions/ready/next_guarded_s2s3.csv
stop if public score is not below 0.5960566585
```

## Naming

- `best_guarded`: 현재 public best인 hard-switch target map.
- `guarded_s2s3`: 현재 best를 유지하고 S2/S3만 subject-holdout 예측을 아주 작게 섞은 후보.
- `soft_w090`, `soft_w085`, `soft_w095`: Q2/Q3를 core 쪽으로 softblend한 이전 후보. `soft_w090` public 실패 후 우선순위에서 내림.

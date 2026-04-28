# Logs

점수 판단에는 아래 두 파일만 먼저 보면 됩니다.

```text
logs/
  README.md
  public_scores.csv       # DACON public score를 직접 기록
  candidate_scores.csv    # 지금 제출 후보와 우선순위
  experiments.csv         # 실험 생성 시 자동 기록된 OOF 상세
  stability/
    stability_*.csv       # target/fold/seed 안정성 상세 로그
```

## Read Order

1. `public_scores.csv`: 실제 leaderboard 점수.
2. `candidate_scores.csv`: 다음에 어떤 파일을 제출할지.
3. `experiments.csv`: OOF와 target-wise 구성 확인.
4. `stability/`: seed/fold별 디버깅이 필요할 때만 확인.

## Current Best

```text
submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv
public score: 0.5960566585
```

## Failed Public Probes

```text
submission_public_lgb_v4_softblend_w090.csv
public score: 0.5962890684

submission_public_lgb_v5_public_lgb_targetwise_histmix_guarded_v1_subject_holdout.csv
public score: 0.5968333841
```

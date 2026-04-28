# 구현 지시서 모음

작성일: 2026-04-27  
기준 공개 점수: **0.5960566585** (histmix_guarded_v1)

---

## 파일 목록

| 파일 | 내용 | 우선순위 |
|------|------|---------|
| [T1_fix_cv.md](T1_fix_cv.md) | CV 수정 — subject-stratified holdout | ★★★★★ 최우선 |
| [T2_trainonly_zscore.md](T2_trainonly_zscore.md) | z-score 누수 제거 (train-only 계산) | ★★★★☆ |
| [T3_per_target_params.md](T3_per_target_params.md) | 타겟별 LGB 하이퍼파라미터 | ★★★★☆ |
| [T4_blend_optimizer.md](T4_blend_optimizer.md) | OOF 기반 타겟별 블렌딩 가중치 최적화 | ★★★☆☆ |
| [T5_feature_stability.md](T5_feature_stability.md) | 피처 안정성 분석 및 stable subset 구성 | ★★★☆☆ |
| [DIAGNOSIS.md](DIAGNOSIS.md) | 전체 진단 결과 및 리스크 목록 | 참고 |

---

## 실행 순서

```
T1 → T2 → T3 → T4 → T5
```

**T1을 반드시 먼저 실행.** T1 완료 후 OOF 수치를 새 기준선으로 삼고 이후 실험 판단.

---

## 핵심 진단 요약

| 문제 | 심각도 | 현상 |
|------|--------|------|
| StratifiedKFold subject 누수 | 🔴 CRITICAL | OOF 0.572 vs 실제 ~0.60 |
| test/train 날짜 혼재 | 🔴 CRITICAL | test의 ~55%가 train 기간과 겹침 |
| z-score transductive 누수 | 🔴 HIGH | test 데이터가 train z-score 기준에 영향 |
| folds.py group_time 미사용 | 🟡 MEDIUM | 정의만 되어있고 학습에 연결 안됨 |
| softblend S4 미적용 | 🟡 MEDIUM | Q2/Q3만 혼합, S4 hist411 효과 미탐색 |

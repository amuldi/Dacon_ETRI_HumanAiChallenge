# 전체 진단 결과

작성일: 2026-04-27

---

## 데이터 구조 핵심 사실

```
subject 수       : 10명 (id01 ~ id10)
train rows       : 450 (subject당 평균 45행)
test  rows       : 250 (subject당 평균 25행)
sleep_date       : lifelog_date + 1일 (항상 고정)
타겟             : Q1, Q2, Q3, S1, S2, S3, S4 (모두 이진)
평가지표         : 7개 타겟 Binary Log-Loss 평균
```

### 타겟 분포

| 타겟 | 레이블 비율 (양성) | OOF 난이도 |
|------|--------------------|-----------|
| S1   | 0.682 | 쉬움 (0.524) |
| S3   | 0.662 | 쉬움 (0.544) |
| S2   | 0.651 | 보통 (0.563) |
| Q3   | 0.600 | 보통 (0.587) |
| Q2   | 0.562 | 보통 (0.586) |
| Q1   | 0.496 | 어려움 (0.591) |
| S4   | 0.560 | 가장 어려움 (0.611) |

### train/test 날짜 혼재 구조

```
id01: train=[2024-06-26, 2024-08-31]  test=[2024-07-30, 2024-09-14]
      test 중 14행이 train 기간 내, 13행이 이후
id04: train=[2024-07-31, 2024-10-26]  test=[2024-09-09, 2024-10-29]
      test 중 24행이 train 기간 내,  3행이 이후
```

**결론:** test의 ~55%는 train과 같은 달력 기간에 위치.  
대회 split은 시간순이 아닌 **날짜 랜덤 샘플링** 방식.

---

## 현재 최적 파이프라인

```
피처 테이블 (1,135개 컬럼):
  ├── raw modality 일별 집계          831개
  ├── sleep-window 피처               41개
  ├── subject z-score (transductive) 258개  ← 누수 있음
  └── target encoding (shift+rolling)  35개

학습:
  StratifiedKFold(n_splits=5, shuffle=True)  ← 누수 있음
  × 8 seeds → 평균
  타겟별 피처 뷰 (histmix_guarded_v1):
    Q1,S1,S2,S3 → public_core  (556 피처)
    Q2,S4       → public_hist411 (411 피처)
    Q3          → public_hist365 (365 피처)

결과:
  OOF mean (StratifiedKFold): 0.5722  ← 낙관적 (누수)
  공개 점수:                  0.5961
  group_time CV (측정):       0.6099  ← 비관적 (방향 반대)
```

---

## 문제 및 리스크 목록

### 🔴 CRITICAL-1: StratifiedKFold subject 완전 누수

```python
# 검증 결과:
Fold 0: tr_subj=10  va_subj=10  overlap=10  ← 전 subject 양쪽 등장
Fold 1: tr_subj=10  va_subj=10  overlap=10
...
```

10개 subject가 매 fold의 train/val 양쪽에 있음 → subject별 행동 패턴이 val로 직접 누수.  
**OOF 0.5722는 신뢰 불가. 진짜 예상 성능은 ~0.595~0.600.**

### 🔴 CRITICAL-2: test/train 날짜 혼재

test가 순수하게 train 이후 데이터가 아님.  
→ group_time(forward-chaining) CV는 실제 대회 구조와 불일치.  
→ **올바른 CV: 각 subject의 날짜를 비시간순 랜덤 holdout (test 비율 ~36% 반영)**

### 🔴 HIGH: z-score transductive 누수

```python
# 현재 코드 (public_lgb.py):
mu = output.groupby("subject_id")[col].transform("mean")  # train+test 전체 사용
```

- test가 subject당 ~35%이므로 train z-score의 평균 기준점이 test 공변량에 의해 이동
- 실측: d_mscreen_on_min 기준 평균 이동 12.07, 최대 35.6 (원단위)
- 258개 z-score 피처 전체에 해당

### 🟡 MEDIUM: folds.py group_time 미연결

`build_group_time_manifest()`가 구현되어 있으나 `_train_public_lgb_with_target_views()`에서 사용되지 않음.  
OOF parquet에 `split_scheme = "public_stratified"`로 하드코딩.

### 🟡 MEDIUM: softblend S4 미탐색

현재 softblend:
- Q2: guarded ×w + core ×(1-w), w=0.85/0.90/0.95
- Q3: 동일
- S4: **변경 없음** (guarded = hist411 그대로)

OOF 탐색 결과 S4는 `w=1.0` (hist411 단독)이 최적 — 하지만 hist365, core와의 2-model blend는 미탐색.

### 🟡 MEDIUM: seed variance 미추적

summary JSON에 seed별 점수가 없음 (`seed_scores` 키 없음).  
재실행 없이는 불안정 타겟 식별 불가.

### 🟢 LOW: OOF over-confidence (교정 미적용)

```
Q3: 레이블 평균 0.600  예측 평균 0.616  (+2.7% 과잉 예측)
S1: 레이블 평균 0.682  예측 평균 0.700  (+2.6% 과잉 예측)
S3: 레이블 평균 0.662  예측 평균 0.675  (+2.0% 과잉 예측)
```

최종 클리핑만 적용, 확률 교정(Platt/Isotonic) 미적용.

---

## 실험 우선순위 테이블

| # | 실험 | 예상 공개점수 | 리스크 | 소요 시간 |
|---|------|-------------|--------|---------|
| T1 | CV 수정 (subject holdout) | ~0.594~0.597 | 낮음 | 2h |
| T2 | train-only z-score | ~0.594~0.597 | 낮음 | 1h |
| T3 | 타겟별 파라미터 (S4 강화) | ~0.590~0.594 | 낮음 | 3h |
| T4 | 블렌딩 가중치 최적화 | ~0.592~0.594 | 낮음 | 1h |
| T5 | 피처 안정성 분석 | ~0.588~0.592 | 보통 | 4h |

---

## 제출 권장 순서

```
현재 베스트: 0.5961

1차  T1 실행 → subject_holdout OOF 확인 후 제출
2차  T4 실행 → 최적 블렌딩 제출
3차  T3 실행 → S4 개선된 모델 제출
4차  T2+T3 조합 제출
5차  (여유 시) T5 후 stable subset 재학습 제출
예비  2차+4차 앙상블 블렌딩 제출
```

**⚠️ T1 이전에 T4를 실행하면 StratifiedKFold 누수 OOF(0.572)로 최적화됨 → 과적합 위험.**

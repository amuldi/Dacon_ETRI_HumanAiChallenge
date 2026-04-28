# T3 — 타겟별 LGB 하이퍼파라미터

우선순위: ★★★★☆  
예상 소요: 3시간  
예상 공개점수 향상: +0.003~0.008  
선행 조건: T1 완료 후 실행 권장 (T2는 선택)

---

## 목적

모든 타겟에 동일한 LGB 파라미터를 사용 중.  
타겟별 난이도 차이가 크므로 각 타겟에 맞는 정규화 강도 적용.

**타겟별 특성:**
```
S4 (OOF 0.611): 가장 어려움. subject 분산 큼 (id01=0.760, id09=0.731)
                → 과적합 중 → 최대 정규화 필요
Q1 (OOF 0.591): 레이블 균형 (0.496), 어려움 → feature 다양성 필요
S1 (OOF 0.524): 가장 쉬움 → 용량 더 허용
S2,S3 (OOF 0.56~0.54): 중간 → 현재 파라미터 유지
```

---

## 파일 1: 신규 생성

**경로:** `src/etri_human_challenge/lgb_target_params.py`

```python
"""타겟별 LightGBM 하이퍼파라미터 설정.

설계 원칙:
  - 어려운 타겟(S4, Q1): num_leaves 줄이고, reg_lambda 올리고,
    feature_fraction 줄여 앙상블 다양성 확보
  - 쉬운 타겟(S1): num_leaves 올리고, reg 줄여 충분한 용량 허용
  - n_estimators: 항상 넉넉히 설정, early_stopping이 최적점 탐색
"""
from .public_lgb import PUBLIC_LGB_PARAMS

TARGET_LGB_PARAMS: dict[str, dict] = {
    # Q1: 균형 데이터 (0.496), 어려움 — 피처 다양성으로 과적합 방지
    "Q1": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":       31,
        "learning_rate":    0.015,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.7,
        "reg_alpha":        0.5,
        "reg_lambda":       3.0,
        "min_child_samples": 25,
        "n_estimators":     3000,
    },

    # Q2: hist411 사용, 현재 파라미터가 거의 최적
    "Q2": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":       31,
        "learning_rate":    0.02,
        "feature_fraction": 0.6,
        "reg_alpha":        0.3,
        "reg_lambda":       2.0,
        "n_estimators":     2500,
    },

    # Q3: hist365 사용, Q2와 동일 설정
    "Q3": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":       31,
        "learning_rate":    0.02,
        "feature_fraction": 0.6,
        "reg_alpha":        0.3,
        "reg_lambda":       2.0,
        "n_estimators":     2500,
    },

    # S1: 가장 쉬움 (0.524) — 용량 확장
    "S1": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":       63,
        "learning_rate":    0.025,
        "feature_fraction": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_estimators":     2000,
    },

    # S2, S3: 표준 파라미터, 트리 수만 증가
    "S2": {**PUBLIC_LGB_PARAMS, "n_estimators": 2000},
    "S3": {**PUBLIC_LGB_PARAMS, "n_estimators": 2000},

    # S4: 가장 어려움 (0.611), subject 분산 큼 — 최대 정규화
    "S4": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":        15,
        "learning_rate":     0.01,
        "feature_fraction":  0.5,
        "bagging_fraction":  0.6,
        "reg_alpha":         1.0,
        "reg_lambda":        5.0,
        "min_child_samples": 30,
        "n_estimators":      4000,
    },
}
```

---

## 파일 2: 수정

**경로:** `src/etri_human_challenge/public_lgb.py`

### 2-A. 함수 시그니처 수정

```python
def _train_public_lgb_with_target_views(
    *,
    run_name: str,
    target_feature_views: dict[str, str],
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    persist: bool = True,
    cv_scheme: str = "public_stratified",
    use_target_params: bool = False,   # ← 추가
) -> dict[str, Any]:
```

### 2-B. params 생성 로직 교체

기존 코드 (L45 근방):
```python
params = {**PUBLIC_LGB_PARAMS, "random_state": int(seed)}
```

교체:
```python
if use_target_params:
    from .lgb_target_params import TARGET_LGB_PARAMS
    base = TARGET_LGB_PARAMS.get(target, PUBLIC_LGB_PARAMS)
    params = {**base, "random_state": int(seed)}
else:
    params = {**PUBLIC_LGB_PARAMS, "random_state": int(seed)}
```

### 2-C. 공개 API 함수에 파라미터 전달

`train_public_lgb()` 및 `train_public_lgb_targetwise()` 함수 시그니처에 각각 `use_target_params: bool = False` 추가 후 내부 호출 시 전달.

### 2-D. 새 프리셋 추가

`PUBLIC_LGB_TARGETWISE_PRESETS` 딕셔너리에 추가:

```python
"histmix_guarded_v1_tuned": {
    "Q2": "public_hist411",
    "Q3": "public_hist365",
    "S4": "public_hist411",
},
```

---

## 파일 3: 신규 스크립트

**경로:** `scripts/train_public_lgb_tuned.py`

```python
#!/usr/bin/env python3
"""타겟별 하이퍼파라미터로 histmix_guarded_v1_tuned 학습."""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

from etri_human_challenge.public_lgb import (
    train_public_lgb_targetwise,
    make_public_lgb_targetwise_submission,
)

result = train_public_lgb_targetwise(
    preset_name="histmix_guarded_v1_tuned",
    default_feature_view="public_core",
    n_folds=5,
    seeds=[42, 1234, 9999, 7, 314, 2025, 777, 555],
    cv_scheme="subject_holdout",    # T1 완료 후
    use_target_params=True,         # T3 핵심
    persist=True,
)

print("=== Tuned OOF ===")
print(f"mean: {result['scores']['mean']:.6f}")
for t, v in result["scores"].items():
    if t not in ("mean", "std"):
        print(f"  {t}: {v:.6f}")

make_public_lgb_targetwise_submission(
    tag="public_lgb_v5_tuned",
    preset_name="histmix_guarded_v1_tuned",
    cv_scheme="subject_holdout",
    use_target_params=True,
)
```

---

## S4 파라미터 미세 탐색 (선택)

S4는 가장 불확실한 타겟이므로 핵심 파라미터를 추가 탐색할 수 있다.

```python
# 스크립트 추가 탐색 코드 (scripts/sweep_s4_params.py 별도 작성)
S4_SWEEP = [
    {"num_leaves": 7,  "reg_lambda": 5.0,  "learning_rate": 0.01},
    {"num_leaves": 15, "reg_lambda": 5.0,  "learning_rate": 0.01},  # ← T3 기본값
    {"num_leaves": 31, "reg_lambda": 5.0,  "learning_rate": 0.015},
    {"num_leaves": 15, "reg_lambda": 10.0, "learning_rate": 0.01},
    {"num_leaves": 15, "reg_lambda": 3.0,  "learning_rate": 0.015},
]
# 각 설정으로 S4만 학습하여 OOF 비교 후 best 선택
```

---

## 실행

```bash
cd "<repo_root>"
python3 scripts/train_public_lgb_tuned.py
```

---

## 검증 체크리스트

- [ ] `lgb_target_params.py`에 `TARGET_LGB_PARAMS` 딕셔너리가 7개 타겟 모두 포함하는가?
- [ ] S4 OOF가 기존 0.611보다 낮아졌는가? (0.600 이하 목표)
- [ ] S1 OOF가 기존 0.524 수준 유지되는가? (과적합 없이)
- [ ] 전체 OOF mean이 subject_holdout 기준선보다 낮아졌는가?

---

## 예상 결과

| 타겟 | 기존 OOF | 목표 OOF | 근거 |
|------|---------|---------|------|
| S4   | 0.611   | ~0.600  | num_leaves 축소 + reg_lambda 5.0 |
| Q1   | 0.591   | ~0.585  | feature_fraction 0.5로 다양성 확보 |
| S1   | 0.524   | ~0.520  | 용량 확장 (num_leaves 63) |
| Q2,Q3,S2,S3 | - | 유지 | 현재 파라미터 적절 |

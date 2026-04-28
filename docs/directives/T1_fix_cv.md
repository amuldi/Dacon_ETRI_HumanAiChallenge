# T1 — CV 수정: Subject-Stratified Holdout

우선순위: ★★★★★ **가장 먼저 실행**  
예상 소요: 2시간  
예상 공개점수: ~0.594~0.597

---

## 목적

현재 `StratifiedKFold`는 10개 전 subject가 매 fold의 train/val 양쪽에 등장.  
실제 대회 test는 각 subject 날짜의 ~36%를 비시간순 랜덤 선택.  
이를 재현하는 CV로 교체하여 신뢰 가능한 OOF 기준선을 확보한다.

**주의:** 이 작업 후 OOF 수치가 0.572 → ~0.595 로 올라간다.  
이는 성능 하락이 아니라 누수가 제거된 정직한 수치이므로 혼동하지 말 것.

---

## 파일 1: 신규 생성

**경로:** `src/etri_human_challenge/proper_cv.py`

```python
"""
실제 대회 구조를 재현하는 CV 스킴.

대회 test = 각 subject 날짜를 비시간순으로 ~36% 랜덤 선택.
→ 올바른 CV: 각 subject의 날짜를 n_folds 등분하여 순환 holdout.
   시간 순서 없음. subject 간 완전 독립.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator


def subject_stratified_holdout_iter(
    train_frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    subject별로 행을 n_folds에 균등 배정 후 순환 holdout.

    Parameters
    ----------
    train_frame : train split DataFrame. 반드시 reset_index(drop=True) 상태.
    n_folds     : fold 수 (기본 5).
    random_state: 재현성용 seed.

    Yields
    ------
    (train_idx, val_idx) : 정수 numpy 배열 쌍.

    보장사항
    --------
    - 각 행은 정확히 1번 val에 등장.
    - 같은 subject의 행이 한 fold에서 train/val에 동시 등장하지 않음.
    - 시간 순서 제약 없음 (대회 split 구조 반영).
    """
    rng = np.random.default_rng(random_state)
    frame = train_frame.reset_index(drop=True)
    fold_assignment = np.full(len(frame), -1, dtype=int)

    for subj in frame["subject_id"].unique():
        idx = frame.index[frame["subject_id"] == subj].to_numpy()
        shuffled = rng.permutation(idx)
        for fold_i, chunk in enumerate(np.array_split(shuffled, n_folds)):
            fold_assignment[chunk] = fold_i

    for fold_id in range(n_folds):
        val_idx   = np.where(fold_assignment == fold_id)[0]
        train_idx = np.where(fold_assignment != fold_id)[0]
        yield train_idx, val_idx
```

---

## 파일 2: 수정

**경로:** `src/etri_human_challenge/public_lgb.py`

### 2-A. 함수 시그니처 수정

`_train_public_lgb_with_target_views()` 파라미터에 `cv_scheme` 추가:

```python
def _train_public_lgb_with_target_views(
    *,
    run_name: str,
    target_feature_views: dict[str, str],
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    persist: bool = True,
    cv_scheme: str = "public_stratified",   # ← 추가
) -> dict[str, Any]:
```

### 2-B. splitter 선택 로직 교체

기존 코드 (L42 근방):
```python
splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
for train_idx, valid_idx in splitter.split(X_train, y):
```

교체:
```python
if cv_scheme == "subject_holdout":
    from .proper_cv import subject_stratified_holdout_iter
    fold_splits = list(
        subject_stratified_holdout_iter(train, n_folds=n_folds, random_state=int(seed))
    )
else:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    fold_splits = list(splitter.split(X_train, y))

for train_idx, valid_idx in fold_splits:
```

### 2-C. OOF split_scheme 필드 수정

기존 (L87 근방):
```python
oof_export["split_scheme"] = "public_stratified"
```

교체:
```python
oof_export["split_scheme"] = cv_scheme
```

### 2-D. 공개 API 함수 두 곳에도 파라미터 전달

`train_public_lgb()` 함수:
```python
def train_public_lgb(
    feature_view: str = "public_core",
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    persist: bool = True,
    cv_scheme: str = "public_stratified",   # ← 추가
) -> dict[str, Any]:
    return _train_public_lgb_with_target_views(
        ...
        cv_scheme=cv_scheme,   # ← 전달
    )
```

`train_public_lgb_targetwise()` 함수도 동일하게 `cv_scheme` 파라미터 추가 및 전달.

---

## 파일 3: 신규 스크립트

**경로:** `scripts/train_subject_holdout.py`

```python
#!/usr/bin/env python3
"""subject_holdout CV로 histmix_guarded_v1 재학습."""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

from etri_human_challenge.public_lgb import (
    make_public_lgb_targetwise_submission,
    train_public_lgb_targetwise,
)

result = train_public_lgb_targetwise(
    preset_name="histmix_guarded_v1",
    default_feature_view="public_core",
    n_folds=5,
    seeds=[42, 1234, 9999, 7, 314, 2025, 777, 555],
    cv_scheme="subject_holdout",
    persist=True,
)

print("=== subject_holdout OOF ===")
print(f"mean: {result['scores']['mean']:.6f}")
for t, v in result["scores"].items():
    if t not in ("mean", "std"):
        print(f"  {t}: {v:.6f}")

# 제출 파일도 동시 생성
make_public_lgb_targetwise_submission(
    tag="public_lgb_v5",
    preset_name="histmix_guarded_v1",
    cv_scheme="subject_holdout",
    persist=True,
)
```

---

## 실행

```bash
cd "<repo_root>"
python3 scripts/train_subject_holdout.py
```

---

## 검증 체크리스트

실행 후 확인 사항:

- [ ] OOF mean이 0.575~0.605 사이인가? (0.572 근처면 cv_scheme 미적용)
- [ ] `artifacts/oof/oof_predictions_public_lgb_targetwise_histmix_guarded_v1.parquet`의 `split_scheme` 컬럼이 `"subject_holdout"`인가?
- [ ] 제출 파일이 `artifacts/submissions/` 에 생성되었는가?
- [ ] 제출 후 공개점수가 0.595~0.598 사이인가?

---

## 기대 결과

| 지표 | 기존 | 예상 |
|------|------|------|
| OOF mean | 0.5722 (낙관적) | ~0.595~0.600 (신뢰 가능) |
| 공개 점수 | 0.5961 | 0.594~0.597 |
| OOF-공개 갭 | 0.024 (누수) | ~0.002~0.005 (정상) |

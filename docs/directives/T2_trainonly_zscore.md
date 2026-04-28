# T2 — z-score 누수 제거 (Train-Only Z-Score)

우선순위: ★★★★☆  
예상 소요: 1시간  
예상 공개점수 향상: +0.002~0.004  
선행 조건: T1 완료 후 실행 권장

---

## 목적

현재 subject z-score 258개는 train+test 전체 데이터로 계산됨.  
test 데이터(subject당 ~35%)가 train 피처의 정규화 기준에 영향을 미침.  
train 데이터만으로 평균/표준편차를 계산하고, 그 파라미터를 test에 적용하는 방식으로 수정.

**실측 누수 크기:**
```
d_mscreen_on_min 기준:
  평균 이동: mean=12.07, max=35.6 (원단위)
  표준편차 이동: mean=6.28, max=12.15
```

---

## 수정 파일

**경로:** `src/etri_human_challenge/public_lgb.py`  
**함수:** `_add_subject_zscores()`

### 기존 코드 (L294~L309)

```python
def _add_subject_zscores(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    numeric_cols = output.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set(TARGET_COLUMNS + ["dow", "month", "week", "is_weekend", "subject_num"])
    zscore_payload: dict[str, pd.Series] = {}
    zscore_cols: list[str] = []
    for col in numeric_cols:
        if col in excluded or "__" in col:
            continue
        mu = output.groupby("subject_id")[col].transform("mean")   # ← 문제: train+test 전체
        sig = output.groupby("subject_id")[col].transform("std").replace(0, np.nan)
        z_col = f"{col}__subj_z"
        zscore_payload[z_col] = (output[col] - mu) / sig
        zscore_cols.append(z_col)
    if zscore_payload:
        output = pd.concat([output, pd.DataFrame(zscore_payload, index=output.index)], axis=1)
    return output, zscore_cols
```

### 교체 코드

```python
def _add_subject_zscores(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    numeric_cols = output.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set(TARGET_COLUMNS + ["dow", "month", "week", "is_weekend", "subject_num"])
    zscore_payload: dict[str, pd.Series] = {}
    zscore_cols: list[str] = []

    # train 행만으로 subject별 통계 계산
    train_mask = output["split"] == "train"

    for col in numeric_cols:
        if col in excluded or "__" in col:
            continue

        # train 행에서만 평균/표준편차 계산
        train_stats = (
            output.loc[train_mask]
            .groupby("subject_id")[col]
            .agg(mu="mean", sig="std")
        )

        # train 파라미터를 train+test 전체에 적용
        joined = output[["subject_id"]].join(train_stats, on="subject_id")
        mu  = joined["mu"]
        sig = joined["sig"].replace(0.0, np.nan)

        z_col = f"{col}__subj_z"
        zscore_payload[z_col] = (output[col] - mu) / sig
        zscore_cols.append(z_col)

    if zscore_payload:
        output = pd.concat(
            [output, pd.DataFrame(zscore_payload, index=output.index)], axis=1
        )
    return output, zscore_cols
```

---

## 피처 테이블 재빌드

코드 수정 후 캐시된 피처 테이블을 재생성해야 한다.

### 방법 A: 기존 파일 덮어쓰기

```python
# Python 인터랙티브 또는 임시 스크립트
import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".vendor")
from etri_human_challenge.public_lgb import build_public_lgb_feature_table
build_public_lgb_feature_table(persist=True)
```

### 방법 B: 별도 파일로 저장 후 비교 (권장)

`build_public_lgb_feature_table()` 함수에 `suffix: str = ""` 파라미터를 추가:

```python
def build_public_lgb_feature_table(persist: bool = True, suffix: str = "") -> pd.DataFrame:
    ...
    if persist:
        fname = f"public_lgb_feature_table{suffix}.parquet"  # ← suffix 적용
        path = FEATURES_DIR / fname
        feature_table.to_parquet(path, index=False)
```

실행:
```bash
# train-only z-score 버전 저장
python3 -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.vendor')
from etri_human_challenge.public_lgb import build_public_lgb_feature_table
build_public_lgb_feature_table(persist=True, suffix='_trainz')
"
```

`public_lgb_feature_table_trainz.parquet`와 기존 `public_lgb_feature_table.parquet`를 각각 학습하여 OOF 비교 후 좋은 것 채택.

---

## 검증 코드

수정 전후 z-score 차이 확인:

```python
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.vendor')
import pandas as pd, numpy as np

# 수정 전 (기존 테이블)
old = pd.read_parquet("artifacts/features/public_lgb_feature_table.parquet")
# 수정 후 (재빌드)
new = pd.read_parquet("artifacts/features/public_lgb_feature_table_trainz.parquet")

col = "d_mscreen_on_min__subj_z"
train_mask = old["split"] == "train"
diff = (new.loc[train_mask, col] - old.loc[train_mask, col]).abs()
print(f"train z-score 변화: mean={diff.mean():.4f}  max={diff.max():.4f}")
# 기대값: mean > 0 (변화 있음), 수정 전보다 test 방향으로 치우치지 않음
```

---

## 검증 체크리스트

- [ ] `_add_subject_zscores()`에서 `train_mask = output["split"] == "train"` 라인이 있는가?
- [ ] `transform("mean")` 또는 `transform("std")`가 제거되었는가?
- [ ] 재빌드 후 피처 컬럼 수가 동일한가? (수정 전후 1135개)
- [ ] train 행의 z-score 값이 수정 전후 다른가? (diff mean > 0)
- [ ] test 행의 z-score 값도 계산되는가? (NaN 아님)

---

## 예상 결과

| 지표 | 예상 변화 | 이유 |
|------|----------|------|
| OOF mean | +0.001~0.003 상승 | z-score가 덜 정밀해짐 (정직해짐) |
| 공개점수 | +0.002~0.004 개선 | 실제 추론 환경과 정렬 |
| OOF-공개 갭 | 감소 | 누수 경로 하나 제거 |

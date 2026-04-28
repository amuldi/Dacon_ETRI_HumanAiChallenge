# T5 — 피처 안정성 분석 및 Stable Feature Subset

우선순위: ★★★☆☆  
예상 소요: 4시간 (분석 2h + 재학습 2h)  
예상 공개점수 향상: +0.003~0.007  
선행 조건: T1, T2 완료 후 실행

---

## 목적

현재 피처 수 대비 학습 샘플 수 비율:
```
public_hist411 (S4용): 411 피처 / fold당 ~360 train rows = 1.14  ← 위험
public_core   (Q1용): 556 피처 / fold당 ~360 train rows = 1.54  ← 매우 위험
```

seed별로 피처 중요도(gain) 분산이 큰 피처는 노이즈.  
안정적인 피처만 유지하면 과적합 감소, 특히 S4 개선 기대.

---

## 신규 스크립트

**경로:** `scripts/analyze_feature_stability.py`

```python
#!/usr/bin/env python3
"""
타겟별 피처 중요도를 seed 반복으로 수집하고 안정적인 subset을 선택한다.

출력:
  artifacts/features/feat_importance_{target}_{view}.parquet  # 중요도 행렬
  artifacts/features/stable_features_{target}.json            # 선택된 피처 목록
  reports/features/feature_stability_{target}.md              # 분석 리포트
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from etri_human_challenge.constants import TARGET_COLUMNS
from etri_human_challenge.paths import FEATURES_DIR, REPORT_FEATURES_DIR, ensure_runtime_dirs
from etri_human_challenge.proper_cv import subject_stratified_holdout_iter
from etri_human_challenge.public_lgb import (
    load_public_lgb_feature_table,
    get_public_lgb_feature_columns,
    PUBLIC_LGB_PARAMS,
)
from etri_human_challenge.utils import clip_probabilities, write_markdown


# 타겟별 피처 뷰 (현재 best 구성)
TARGET_VIEW = {
    "Q1": "public_core",
    "Q2": "public_hist411",
    "Q3": "public_hist365",
    "S1": "public_core",
    "S2": "public_core",
    "S3": "public_core",
    "S4": "public_hist411",
}

# 중요도 수집에 사용할 seed 수 (5개면 충분, 더 많을수록 정확)
STABILITY_SEEDS = [42, 1234, 9999, 7, 314]

# 비교할 Top-K 후보
TOP_K_CANDIDATES = [50, 100, 150, 200, 300, "all"]

# 안정성 기준: seed 간 CV (std/mean) ≤ 이 값인 피처를 '안정적'으로 판단
CV_THRESHOLD = 1.5


def collect_importances(
    X: pd.DataFrame,
    y: np.ndarray,
    train_frame: pd.DataFrame,
    seeds: list[int],
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    seed별로 LGB를 학습하고 fold-averaged gain importance를 수집.

    반환: DataFrame (index=피처명, columns=seed값, 값=gain importance)
    """
    imp_per_seed: dict[int, pd.Series] = {}

    for seed in seeds:
        fold_imp = pd.Series(0.0, index=X.columns, dtype=float)
        n_folds_actual = 0

        for tr_idx, va_idx in subject_stratified_holdout_iter(
            train_frame, n_folds=n_folds, random_state=seed
        ):
            params = {
                **PUBLIC_LGB_PARAMS,
                "random_state": seed,
                "n_estimators": 1000,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X.iloc[tr_idx],
                y[tr_idx],
                eval_set=[(X.iloc[va_idx], y[va_idx])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            fold_imp += pd.Series(
                model.feature_importances_, index=X.columns, dtype=float
            )
            n_folds_actual += 1

        imp_per_seed[seed] = fold_imp / max(n_folds_actual, 1)

    return pd.DataFrame(imp_per_seed)  # (n_features × n_seeds)


def select_stable_features(
    imp_df: pd.DataFrame,
    top_k: int | None,
    *,
    cv_threshold: float = CV_THRESHOLD,
) -> list[str]:
    """
    안정적인 피처를 top_k개 선택.

    안정성 기준: seed 간 CV ≤ cv_threshold  OR  mean importance ≥ 중앙값
    (OR 조건으로 중요한 피처가 불안정해도 유지)
    """
    mean_imp = imp_df.mean(axis=1)
    std_imp  = imp_df.std(axis=1)
    cv_imp   = std_imp / (mean_imp.abs() + 1e-9)

    stable_mask = (cv_imp <= cv_threshold) | (mean_imp >= mean_imp.median())
    candidates  = mean_imp[stable_mask].sort_values(ascending=False)

    if top_k is not None:
        return candidates.index.tolist()[:top_k]
    return candidates.index.tolist()


def evaluate_feature_subset(
    X_full: pd.DataFrame,
    y: np.ndarray,
    train_frame: pd.DataFrame,
    feature_subset: list[str],
    *,
    seed: int = 42,
    n_folds: int = 5,
) -> float:
    """주어진 피처 subset으로 OOF log-loss 계산 (단일 seed)."""
    oof = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in subject_stratified_holdout_iter(
        train_frame, n_folds=n_folds, random_state=seed
    ):
        model = lgb.LGBMClassifier(
            **{**PUBLIC_LGB_PARAMS, "random_state": seed, "n_estimators": 1000}
        )
        model.fit(
            X_full[feature_subset].iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X_full[feature_subset].iloc[va_idx], y[va_idx])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof[va_idx] = model.predict_proba(
            X_full[feature_subset].iloc[va_idx]
        )[:, 1]

    return float(log_loss(y, clip_probabilities(oof)))


def main() -> None:
    ensure_runtime_dirs()

    frame = load_public_lgb_feature_table(rebuild=False)
    train = frame[frame["split"] == "train"].reset_index(drop=True)

    summary_rows: list[dict] = []

    for target in TARGET_COLUMNS:
        view = TARGET_VIEW[target]
        feat_cols = get_public_lgb_feature_columns(frame, view)
        X = train[feat_cols].copy().replace([np.inf, -np.inf], np.nan)
        y = train[target].astype(int).values

        print(f"\n[{target}] ({view}, {len(feat_cols)}개 피처) 중요도 수집 중 ...")
        imp_df = collect_importances(X, y, train, STABILITY_SEEDS)

        # 저장
        imp_path = FEATURES_DIR / f"feat_importance_{target}_{view}.parquet"
        imp_df.to_parquet(imp_path)
        print(f"  중요도 행렬 저장: {imp_path.name}")

        # 안정성 통계
        mean_imp = imp_df.mean(axis=1)
        cv_imp   = imp_df.std(axis=1) / (mean_imp.abs() + 1e-9)
        n_stable = int((cv_imp <= CV_THRESHOLD).sum())
        print(f"  안정적 피처 수 (CV ≤ {CV_THRESHOLD}): {n_stable}/{len(feat_cols)}")

        # Top-K별 OOF 비교
        k_results: list[dict] = []
        for k in TOP_K_CANDIDATES:
            top_k_int = None if k == "all" else int(k)
            subset = select_stable_features(imp_df, top_k_int)
            ll = evaluate_feature_subset(X, y, train, subset, seed=42)
            print(f"  top-{str(k):5s}: {len(subset):4d}개  OOF={ll:.6f}")
            k_results.append({"k": k, "n": len(subset), "oof": ll})

        # 최적 K (OOF 최저점)
        best = min(k_results, key=lambda r: r["oof"])
        best_k_int = None if best["k"] == "all" else int(best["k"])
        stable_feats = select_stable_features(imp_df, best_k_int)

        # 저장
        feat_path = FEATURES_DIR / f"stable_features_{target}.json"
        feat_path.write_text(json.dumps(stable_feats, ensure_ascii=False, indent=2))
        print(f"  Stable subset 저장: {feat_path.name} ({len(stable_feats)}개)")

        # full 대비 개선
        full_oof = next(r["oof"] for r in k_results if r["k"] == "all")
        delta = full_oof - best["oof"]
        summary_rows.append({
            "target":     target,
            "view":       view,
            "full_feats": len(feat_cols),
            "best_k":     best["k"],
            "best_n":     best["n"],
            "full_oof":   full_oof,
            "best_oof":   best["oof"],
            "delta":      delta,
        })

        # 리포트
        lines = [
            f"# 피처 안정성 분석: {target}", "",
            f"- 뷰: `{view}`",
            f"- 전체 피처 수: {len(feat_cols)}",
            f"- 안정적 피처 수: {n_stable}",
            "",
            "## Top-K별 OOF", "",
            "| K | 피처 수 | OOF |",
            "|---|---------|-----|",
        ]
        for r in k_results:
            marker = " ← best" if r["k"] == best["k"] else ""
            lines.append(f"| {r['k']} | {r['n']} | {r['oof']:.6f}{marker} |")
        write_markdown(
            REPORT_FEATURES_DIR / f"feature_stability_{target}.md",
            "\n".join(lines),
        )

    # 전체 요약
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"{'타겟':5s} {'best_k':8s} {'n':6s} {'full_oof':10s} {'best_oof':10s} {'Δ':10s}")
    for r in summary_rows:
        print(f"{r['target']:5s} {str(r['best_k']):8s} {r['best_n']:6d} "
              f"{r['full_oof']:10.6f} {r['best_oof']:10.6f} {r['delta']:+10.6f}")


if __name__ == "__main__":
    main()
```

---

## T5 결과 활용: Stable Subset으로 재학습

T5 완료 후 `stable_features_{target}.json`이 생성되면 이를 새 피처 뷰로 등록하여 재학습.

### 피처 뷰 추가: `public_lgb.py`

`get_public_lgb_feature_columns()` 함수에 `"stable"` 뷰 분기 추가:

```python
if feature_view == "stable":
    import json
    from .paths import FEATURES_DIR
    # target 파라미터가 필요하므로 함수 시그니처에 target: str = "" 추가
    stable_path = FEATURES_DIR / f"stable_features_{target}.json"
    if stable_path.exists():
        stable_feats = json.loads(stable_path.read_text())
        return [c for c in stable_feats if c in frame.columns]
    # fallback: public_core
    return get_public_lgb_feature_columns(frame, "public_core")
```

### 새 프리셋 추가:

```python
"histmix_guarded_v1_stable": {
    "Q2": "public_hist411",   # stable subset 내에서도 hist411 효과 유지
    "Q3": "public_hist365",
    "S4": "public_hist411",
}
```

---

## 실행

```bash
cd "<repo_root>"
python3 scripts/analyze_feature_stability.py
```

---

## 검증 체크리스트

- [ ] 7개 타겟 모두 `stable_features_{target}.json` 생성되었는가?
- [ ] S4의 best_k가 411개 미만인가? (피처 축소 효과 있어야 함)
- [ ] 축소 후 OOF가 full 대비 같거나 낮아졌는가?
- [ ] 리포트에 Top-K 비교 테이블이 있는가?

---

## 예상 결과

| 타겟 | 전체 피처 수 | 예상 best_k | 예상 OOF 개선 |
|------|------------|------------|-------------|
| S4   | 411        | ~150~200   | -0.005~-0.010 |
| Q1   | 556        | ~200~300   | -0.003~-0.006 |
| S1   | 556        | ~100~200   | -0.001~-0.003 |
| Q2,Q3 | 411,365  | ~150       | -0.002~-0.004 |

---

## 주의사항

- **top-50 이하로 내리지 말 것.** 핵심 신호까지 제거되어 역효과.
- T5 분석은 subject_holdout OOF 기준으로 수행해야 신뢰 가능.
- stable subset이 full subset보다 나쁘면 현재 피처 수가 적절하다는 의미 → 채택 불필요.

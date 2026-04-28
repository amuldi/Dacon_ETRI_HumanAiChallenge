# T4 — OOF 기반 타겟별 블렌딩 가중치 최적화

우선순위: ★★★☆☆  
예상 소요: 1시간  
예상 공개점수 향상: +0.002~0.005  
선행 조건: **T1 완료 후 실행** (T1 OOF로 최적화해야 신뢰 가능)

---

## 목적

현재 softblend의 두 가지 문제:
1. Q2/Q3만 혼합, S4 hist411 블렌딩 미탐색
2. 가중치 0.85/0.90/0.95 중 3개만 수동 비교

201점 세밀 그리드로 모든 7개 타겟의 최적 블렌딩 가중치를 독립적으로 탐색.

**OOF 탐색 결과 (사전 측정):**
```
Q2: guarded(hist411) 0.85 + core 0.15 → loss=0.5855  (순수 guarded 0.5858 대비 -0.0003)
Q3: guarded(hist365) 0.85 + core 0.15 → loss=0.5872  (순수 guarded 0.5873 대비 -0.0001)
S4: guarded(hist411) 1.00 + core 0.00 → 혼합 불필요
```
→ 현재 softblend 효과가 미미한 이유: 이미 guarded가 최적에 가까움.  
→ **다른 model 쌍(hist411 vs hist365, tuned vs baseline)에서 더 큰 개선 가능성.**

---

## 신규 스크립트

**경로:** `scripts/optimize_target_blend.py`

```python
#!/usr/bin/env python3
"""
두 run의 OOF를 사용하여 타겟별 최적 블렌딩 가중치를 탐색하고
blended test 제출 파일을 생성한다.

사용 예:
  # guarded_v1 vs public_core
  python3 scripts/optimize_target_blend.py \
      --run-a public_lgb_targetwise_histmix_guarded_v1 \
      --run-b public_lgb_public_core \
      --tag guarded_core_opt

  # tuned 모델 vs hist411
  python3 scripts/optimize_target_blend.py \
      --run-a public_lgb_targetwise_histmix_guarded_v1_tuned \
      --run-b public_lgb_public_hist411 \
      --tag guarded_tuned_h411_opt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import (
    FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_SUBMISSIONS_DIR,
    SUBMISSIONS_DIR, ensure_runtime_dirs,
)
from etri_human_challenge.utils import clip_probabilities, write_json, write_markdown


GRID_POINTS = 201  # 0.000, 0.005, ..., 1.000


def _load_oof_predictions(run_name: str) -> pd.DataFrame:
    """OOF parquet에서 타겟 예측값만 추출."""
    path = OOF_DIR / f"oof_predictions_{run_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"OOF 파일 없음: {path}")
    df = pd.read_parquet(path)
    # 예측 컬럼: Q1_public_lgb, Q2_public_lgb, ...
    result = pd.DataFrame()
    for t in TARGET_COLUMNS:
        pred_col = f"{t}_public_lgb"
        if pred_col not in df.columns:
            raise KeyError(f"컬럼 없음: {pred_col} in {run_name}")
        result[t] = df[pred_col].values
    result["y_" + TARGET_COLUMNS[0]] = df[TARGET_COLUMNS[0]].values  # 진짜 레이블용
    # 레이블도 같이 반환
    for t in TARGET_COLUMNS:
        result[f"label_{t}"] = df[t].values
    return result


def find_optimal_weights(
    oof_a: dict[str, np.ndarray],
    oof_b: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    *,
    n_grid: int = GRID_POINTS,
) -> dict[str, float]:
    """
    각 타겟별로 w* = argmin LogLoss(y, w*pa + (1-w)*pb) 탐색.
    반환: {target: w_opt}  (w=1이면 run-a 단독, w=0이면 run-b 단독)
    """
    weights: dict[str, float] = {}
    print(f"\n{'타겟':5s} {'w_opt':6s} {'loss_blend':10s} {'loss_a':10s} {'loss_b':10s} {'Δ_vs_a':10s}")
    print("-" * 55)

    for t in TARGET_COLUMNS:
        pa = clip_probabilities(oof_a[t])
        pb = clip_probabilities(oof_b[t])
        y  = labels[t]

        loss_a = log_loss(y, pa)
        loss_b = log_loss(y, pb)
        best_w, best_ll = 1.0, loss_a

        for w in np.linspace(0.0, 1.0, n_grid):
            blended = clip_probabilities(w * pa + (1.0 - w) * pb)
            ll = log_loss(y, blended)
            if ll < best_ll:
                best_ll, best_w = ll, float(w)

        weights[t] = best_w
        delta = loss_a - best_ll
        print(f"{t:5s} {best_w:6.3f} {best_ll:10.6f} {loss_a:10.6f} {loss_b:10.6f} {delta:+10.6f}")

    return weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="타겟별 최적 블렌딩 가중치 탐색")
    p.add_argument("--run-a", required=True,
                   help="예: public_lgb_targetwise_histmix_guarded_v1")
    p.add_argument("--run-b", required=True,
                   help="예: public_lgb_public_core")
    p.add_argument("--tag", required=True,
                   help="출력 파일 태그")
    p.add_argument("--clip-min", type=float, default=0.02)
    p.add_argument("--clip-max", type=float, default=0.98)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()

    print(f"Run A: {args.run_a}")
    print(f"Run B: {args.run_b}")

    # OOF 로드
    oof_a_df = _load_oof_predictions(args.run_a)
    oof_b_df = _load_oof_predictions(args.run_b)

    oof_a    = {t: oof_a_df[t].values for t in TARGET_COLUMNS}
    oof_b    = {t: oof_b_df[t].values for t in TARGET_COLUMNS}
    labels   = {t: oof_a_df[f"label_{t}"].values for t in TARGET_COLUMNS}

    # 최적 가중치 탐색
    weights = find_optimal_weights(oof_a, oof_b, labels)

    # OOF blend 최종 점수
    from sklearn.metrics import log_loss as sk_ll
    blend_scores: dict[str, float] = {}
    for t in TARGET_COLUMNS:
        pa = clip_probabilities(oof_a[t])
        pb = clip_probabilities(oof_b[t])
        blended = clip_probabilities(weights[t] * pa + (1.0 - weights[t]) * pb)
        blend_scores[t] = float(sk_ll(labels[t], blended))
    blend_scores["mean"] = float(np.mean([blend_scores[t] for t in TARGET_COLUMNS]))

    print(f"\nOOF blend mean: {blend_scores['mean']:.6f}")

    # Test 예측 블렌딩
    test_a_path = MODELS_DIR / f"test_predictions_{args.run_a}.csv"
    test_b_path = MODELS_DIR / f"test_predictions_{args.run_b}.csv"
    if not test_a_path.exists():
        raise FileNotFoundError(f"test 예측 없음: {test_a_path}")
    if not test_b_path.exists():
        raise FileNotFoundError(f"test 예측 없음: {test_b_path}")

    test_a = pd.read_csv(test_a_path)
    test_b = pd.read_csv(test_b_path)

    template = load_submission_template()
    submission = template[KEY_COLUMNS].copy()
    for t in TARGET_COLUMNS:
        raw = weights[t] * test_a[t].values + (1.0 - weights[t]) * test_b[t].values
        submission[t] = np.clip(raw, args.clip_min, args.clip_max)

    out_path = SUBMISSIONS_DIR / f"submission_{args.tag}_blend_opt.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n제출 파일 저장: {out_path}")

    # 가중치 및 점수 저장
    payload = {
        "run_a":        args.run_a,
        "run_b":        args.run_b,
        "tag":          args.tag,
        "weights":      weights,
        "oof_scores":   blend_scores,
    }
    write_json(FEATURES_DIR / f"{args.tag}_blend_weights.json", payload)

    # 리포트
    lines = [
        f"# Blend Optimization Report: {args.tag}", "",
        f"- Run A: `{args.run_a}`",
        f"- Run B: `{args.run_b}`",
        f"- OOF blend mean: {blend_scores['mean']:.6f}", "",
        "## 타겟별 가중치 (w=1: A 단독, w=0: B 단독)", "",
        "| 타겟 | w_opt | OOF blend |",
        "|------|-------|-----------|",
    ]
    for t in TARGET_COLUMNS:
        lines.append(f"| {t} | {weights[t]:.3f} | {blend_scores[t]:.6f} |")
    write_markdown(REPORT_SUBMISSIONS_DIR / f"{args.tag}_blend_opt.md", "\n".join(lines))

    print("\n완료.")


if __name__ == "__main__":
    main()
```

---

## 실행 명령 목록

```bash
cd "<repo_root>"

# 1. guarded_v1 vs public_core (현재 softblend 정밀화)
python3 scripts/optimize_target_blend.py \
    --run-a public_lgb_targetwise_histmix_guarded_v1 \
    --run-b public_lgb_public_core \
    --tag blend_guarded_core

# 2. guarded_v1 vs hist411 (S4 추가 탐색)
python3 scripts/optimize_target_blend.py \
    --run-a public_lgb_targetwise_histmix_guarded_v1 \
    --run-b public_lgb_public_hist411 \
    --tag blend_guarded_h411

# 3. T3 완료 후: tuned 모델 vs guarded_v1
python3 scripts/optimize_target_blend.py \
    --run-a public_lgb_targetwise_histmix_guarded_v1_tuned \
    --run-b public_lgb_targetwise_histmix_guarded_v1 \
    --tag blend_tuned_guarded
```

---

## 검증 체크리스트

- [ ] T1 이후 생성된 OOF를 사용하는가? (`split_scheme = "subject_holdout"`)
- [ ] 각 타겟별 w_opt가 0~1 사이인가?
- [ ] blend OOF mean이 run-a 단독보다 낮거나 같은가?
- [ ] 제출 파일 행 수가 250인가?
- [ ] 예측값이 [0.02, 0.98] 범위인가?

---

## 주의사항

**T1 이전에 T4 실행 금지.**  
StratifiedKFold OOF(0.572)로 최적화하면 누수된 점수를 기준으로 가중치가 설정되어  
실제 test에서 성능이 보장되지 않는다.  
반드시 subject_holdout OOF가 생성된 후 실행.

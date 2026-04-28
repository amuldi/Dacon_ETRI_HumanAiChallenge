#!/usr/bin/env python3
"""타겟별 하이퍼파라미터로 histmix_guarded_v1_tuned 학습. (T3)"""
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

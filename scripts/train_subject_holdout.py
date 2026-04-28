#!/usr/bin/env python3
"""subject_holdout CV로 histmix_guarded_v1 재학습. (T1)"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import REPORT_SUBMISSIONS_DIR, SUBMISSIONS_DIR
from etri_human_challenge.public_lgb import (
    _train_public_lgb_with_target_views,
    resolve_target_feature_views,
)
from etri_human_challenge.utils import write_markdown


SEEDS = [42, 1234, 9999, 7, 314, 2025, 777, 555]
PRESET_NAME = "histmix_guarded_v1"
RUN_NAME = "public_lgb_targetwise_histmix_guarded_v1_subject_holdout"
TAG = "public_lgb_v5"
CLIP_MIN = 0.02
CLIP_MAX = 0.98


target_feature_views = resolve_target_feature_views(
    default_feature_view="public_core",
    preset_name=PRESET_NAME,
)

result = _train_public_lgb_with_target_views(
    run_name=RUN_NAME,
    target_feature_views=target_feature_views,
    n_folds=5,
    seeds=SEEDS,
    cv_scheme="subject_holdout",
    persist=True,
)

print("=== subject_holdout OOF ===")
print(f"mean: {result['scores']['mean']:.6f}")
for t, v in result["scores"].items():
    if t not in ("mean", "std"):
        print(f"  {t}: {v:.6f}")

# 제출 파일도 동시 생성
test_pred_path = Path(result["artifacts"]["test_prediction_path"])
test_preds = pd.read_csv(test_pred_path)
submission = load_submission_template()[KEY_COLUMNS].copy()
for target in TARGET_COLUMNS:
    submission[target] = np.clip(test_preds[target].to_numpy(dtype=float), CLIP_MIN, CLIP_MAX)

output_path = SUBMISSIONS_DIR / f"submission_{TAG}_{RUN_NAME}.csv"
submission.to_csv(output_path, index=False)

report_lines = [
    f"# Public LGB Targetwise Submission Report ({PRESET_NAME}, subject_holdout)",
    "",
    f"- Source run: `{RUN_NAME}`",
    f"- Source OOF mean log-loss: {result['scores']['mean']:.6f}",
    "- Validation scheme: `subject_holdout`",
    f"- Folds: {result['n_folds']}",
    f"- Seeds: {result['seeds']}",
    f"- Clip range: [{CLIP_MIN:.2f}, {CLIP_MAX:.2f}]",
    f"- Output: `{output_path}`",
    "",
    "## Target Views",
    "",
]
for target in TARGET_COLUMNS:
    report_lines.append(f"- `{target}`: `{target_feature_views[target]}`")
report_lines.extend(["", "## Target Scores", ""])
for target in TARGET_COLUMNS:
    report_lines.append(f"- `{target}`: {result['scores'][target]:.6f}")
write_markdown(
    REPORT_SUBMISSIONS_DIR / f"submission_{TAG}_{RUN_NAME}.md",
    "\n".join(report_lines),
)

print(f"submission: {output_path}")

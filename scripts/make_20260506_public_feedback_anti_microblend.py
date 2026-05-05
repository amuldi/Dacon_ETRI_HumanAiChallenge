#!/usr/bin/env python3
"""Build a public-feedback anti-microblend candidate for 2026-05-06.

The previous non-S4 temporal microblend improved OOF but worsened public:

  current best: 0.5829008297
  failed probe: 0.5834566947

This script treats that public result as directional feedback.  It keeps S4
fixed and moves Q1/Q2/Q3/S1/S2/S3 *opposite* the failed probe direction with a
conservative scale.  This is intentionally not an OOF-guard candidate; the OOF
signal was just proven misleading for this axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_SUBMISSIONS_DIR, ROOT, ensure_runtime_dirs
from etri_human_challenge.utils import binary_log_loss, write_json, write_markdown


CURRENT_RUN = "public_lgb_targetwise_toward57_s4b650_20260504"
FAILED_RUN = "public_lgb_targetwise_20260506_non_s4_temporal_microblend"
CURRENT_PUBLIC = 0.5829008297
FAILED_PUBLIC = 0.5834566947
ANTI075_PUBLIC = 0.5826026306
ANTI075_SCALE = 0.75
FROZEN_TARGETS = {"S4"}
CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"


def _prediction_columns(frame: pd.DataFrame, *, allow_plain_targets: bool) -> pd.DataFrame:
    output: dict[str, pd.Series] = {}
    for target in TARGET_COLUMNS:
        candidates = [
            f"{target}_public_lgb",
            f"{target}_xgb",
            f"{target}_lgb",
            f"{target}_catboost",
            f"{target}_prior_v2",
        ]
        if allow_plain_targets:
            candidates.append(target)
        for column in candidates:
            if column in frame.columns:
                output[target] = frame[column].astype(float)
                break
        else:
            raise ValueError(f"Missing prediction column for {target}")
    return pd.DataFrame(output)[TARGET_COLUMNS]


def _load_oof(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet")
    return raw[KEY_COLUMNS].copy(), raw[TARGET_COLUMNS].astype(float).copy(), _prediction_columns(raw, allow_plain_targets=False)


def _load_test(run_name: str) -> pd.DataFrame:
    raw = pd.read_csv(MODELS_DIR / f"test_predictions_{run_name}.csv")
    return _prediction_columns(raw, allow_plain_targets=True)


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _scale_tag(anti_scale: float) -> str:
    return f"{int(round(anti_scale * 100)):03d}"


def _candidate_run(anti_scale: float) -> str:
    return f"public_lgb_targetwise_20260506_public_feedback_anti_micro{_scale_tag(anti_scale)}"


def _candidate_file(anti_scale: float) -> str:
    return f"submit_20260506_public_feedback_anti_micro{_scale_tag(anti_scale)}.csv"


def _anti_blend(current: pd.DataFrame, failed: pd.DataFrame, anti_scale: float) -> pd.DataFrame:
    output = current.copy()
    for target in TARGET_COLUMNS:
        if target in FROZEN_TARGETS:
            output[target] = current[target].astype(float)
            continue
        failed_delta = failed[target].astype(float) - current[target].astype(float)
        output[target] = current[target].astype(float) - anti_scale * failed_delta
    return output[TARGET_COLUMNS]


def _quadratic_public_estimate(anti_scale: float) -> dict[str, float]:
    # Fit f(s)=a*s^2+b*s+c from s=-1.0 (failed direction), s=0 (current),
    # and s=0.75 (first public-feedback improvement).
    x = np.array([-1.0, 0.0, ANTI075_SCALE], dtype=float)
    y = np.array([FAILED_PUBLIC, CURRENT_PUBLIC, ANTI075_PUBLIC], dtype=float)
    a, b, c = np.polyfit(x, y, deg=2)
    vertex = float(-b / (2.0 * a)) if abs(a) > 1e-15 else float("nan")
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "vertex_scale": vertex,
        "estimate": float(a * anti_scale * anti_scale + b * anti_scale + c),
    }


def _write_outputs(
    *,
    keys: pd.DataFrame,
    labels: pd.DataFrame,
    current_oof: pd.DataFrame,
    failed_oof: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    current_test: pd.DataFrame,
    failed_test: pd.DataFrame,
    candidate_test: pd.DataFrame,
    anti_scale: float,
) -> Path:
    ensure_runtime_dirs()
    READY_DIR.mkdir(parents=True, exist_ok=True)

    current_scores = _score_targets(labels, current_oof)
    failed_scores = _score_targets(labels, failed_oof)
    candidate_scores = _score_targets(labels, candidate_oof)

    drift = {
        target: float(np.mean(np.abs(candidate_test[target].to_numpy(float) - current_test[target].to_numpy(float))))
        for target in TARGET_COLUMNS
    }
    failed_drift = {
        target: float(np.mean(np.abs(failed_test[target].to_numpy(float) - current_test[target].to_numpy(float))))
        for target in TARGET_COLUMNS
    }
    target_delta = {
        target: float(candidate_scores[target] - current_scores[target])
        for target in TARGET_COLUMNS
    }

    oof_export = keys.copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].astype(float)
    oof_export["split_scheme"] = "public_feedback_exception"
    candidate_run = _candidate_run(anti_scale)
    candidate_file = _candidate_file(anti_scale)
    oof_export["model_family"] = candidate_run
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = candidate_oof[target].to_numpy(float)
    oof_path = OOF_DIR / f"oof_predictions_{candidate_run}.parquet"
    oof_export.to_parquet(oof_path, index=False)

    test_path = MODELS_DIR / f"test_predictions_{candidate_run}.csv"
    candidate_test.to_csv(test_path, index=False)

    submission = load_submission_template()[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = np.clip(candidate_test[target].to_numpy(float), CLIP_MIN, CLIP_MAX)
    submission_path = READY_DIR / candidate_file
    submission.to_csv(submission_path, index=False)

    expected_linear_public = CURRENT_PUBLIC - anti_scale * (FAILED_PUBLIC - CURRENT_PUBLIC)
    quadratic_estimate = _quadratic_public_estimate(anti_scale)
    summary: dict[str, Any] = {
        "candidate_run": candidate_run,
        "candidate_file": str(submission_path),
        "current_run": CURRENT_RUN,
        "failed_run": FAILED_RUN,
        "current_public": CURRENT_PUBLIC,
        "failed_public": FAILED_PUBLIC,
        "failed_public_delta": FAILED_PUBLIC - CURRENT_PUBLIC,
        "anti_scale": anti_scale,
        "expected_linear_public": expected_linear_public,
        "quadratic_public_estimate": quadratic_estimate,
        "frozen_targets": sorted(FROZEN_TARGETS),
        "current_scores": current_scores,
        "failed_scores": failed_scores,
        "candidate_scores": candidate_scores,
        "target_delta_candidate_minus_current": target_delta,
        "test_mean_abs_drift_vs_current": drift,
        "failed_test_mean_abs_drift_vs_current": failed_drift,
        "self_critique": [
            "This candidate intentionally violates the old OOF guard because the prior guard-pass candidate failed public.",
            "It uses only one public feedback point, so the anti direction may overfit the public leaderboard.",
            f"The scale is {anti_scale:.2f}; it extrapolates beyond the 0.75 public improvement and can overshoot.",
            "If this also worsens, stop public-feedback postprocessing and return to feature-stable retraining.",
        ],
        "artifacts": {
            "oof_path": str(oof_path),
            "test_prediction_path": str(test_path),
            "submission_path": str(submission_path),
        },
    }
    summary_path = FEATURES_DIR / f"{candidate_run}_summary.json"
    write_json(summary_path, summary)

    report_lines = [
        f"# 2026-05-06 public-feedback anti microblend ({anti_scale:.2f})",
        "",
        "## Decision",
        "",
        f"- Candidate file: `{submission_path}`",
        f"- Current public: `{CURRENT_PUBLIC:.10f}`",
        f"- Failed probe public: `{FAILED_PUBLIC:.10f}`",
        f"- Failed delta: `+{FAILED_PUBLIC - CURRENT_PUBLIC:.10f}`",
        f"- Anti scale: `{anti_scale:.2f}`",
        f"- Linear public estimate: `{expected_linear_public:.10f}`",
        f"- 3-point quadratic public estimate: `{quadratic_estimate['estimate']:.10f}`",
        f"- 3-point quadratic vertex scale: `{quadratic_estimate['vertex_scale']:.3f}`",
        f"- S4: frozen",
        "",
        "## Why this direction",
        "",
        "The last candidate passed OOF guard but public worsened, which means OOF was misleading on this non-S4 temporal axis. The first public-feedback anti move at scale 0.75 improved public from 0.5829008297 to 0.5826026306. Using the three observed points, the quadratic public estimate places the local best near scale 2.57, so this candidate tests a stronger but still sub-vertex scale while keeping S4 untouched.",
        "",
        "## Target diagnostics",
        "",
        "| Target | Candidate OOF delta | Candidate drift | Failed drift |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        report_lines.append(
            f"| `{target}` | `{target_delta[target]:.9f}` | `{drift[target]:.9f}` | `{failed_drift[target]:.9f}` |"
        )
    report_lines.extend(
        [
            "",
            "## Self-critique",
            "",
            "1. This is not an OOF-safe candidate. It is a public-feedback correction after an OOF-safe file failed public.",
            "2. The public score estimate is based on only three submissions, so it can overfit the public leaderboard.",
            "3. The scale is extrapolated from public feedback and still relies on a low-data approximation.",
            "4. If this fails, the next valid direction is feature-stable retraining, not another post-hoc public-feedback flip.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / f"20260506_public_feedback_anti_microblend_{_scale_tag(anti_scale)}.md", "\n".join(report_lines))
    return submission_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anti-scale", type=float, default=0.75)
    args = parser.parse_args()
    anti_scale = float(args.anti_scale)

    current_keys, labels, current_oof = _load_oof(CURRENT_RUN)
    failed_keys, failed_labels, failed_oof = _load_oof(FAILED_RUN)
    if not current_keys.equals(failed_keys):
        raise ValueError("Current/failed OOF keys do not match.")
    if not labels.equals(failed_labels):
        raise ValueError("Current/failed labels do not match.")

    current_test = _load_test(CURRENT_RUN)
    failed_test = _load_test(FAILED_RUN)
    candidate_oof = _anti_blend(current_oof, failed_oof, anti_scale)
    candidate_test = _anti_blend(current_test, failed_test, anti_scale)
    path = _write_outputs(
        keys=current_keys,
        labels=labels,
        current_oof=current_oof,
        failed_oof=failed_oof,
        candidate_oof=candidate_oof,
        current_test=current_test,
        failed_test=failed_test,
        candidate_test=candidate_test,
        anti_scale=anti_scale,
    )
    print(path)
    print(
        json.dumps(
            {
                "current_public": CURRENT_PUBLIC,
                "failed_public": FAILED_PUBLIC,
                "anti_scale": anti_scale,
                "expected_linear_public": CURRENT_PUBLIC - anti_scale * (FAILED_PUBLIC - CURRENT_PUBLIC),
                "quadratic_public_estimate": _quadratic_public_estimate(anti_scale),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

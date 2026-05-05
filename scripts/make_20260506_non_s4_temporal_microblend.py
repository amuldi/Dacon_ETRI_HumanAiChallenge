#!/usr/bin/env python3
"""Build the 2026-05-06 non-S4 temporal microblend candidate.

Strategy:
  - keep the current S4 best completely frozen;
  - only borrow non-S4 temporal-prior movement from a previous temporal run;
  - cap each target's mean absolute test drift at 0.005;
  - reject any target whose OOF score does not improve.

This is intentionally a small candidate.  The goal is not to replay the failed
full temporal push, but to test whether its non-S4 signal helps when S4 risk is
removed and every target is drift-capped.
"""

from __future__ import annotations

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
SOURCE_RUN = "public_lgb_targetwise_temporal_push_s4_v1"
CANDIDATE_RUN = "public_lgb_targetwise_20260506_non_s4_temporal_microblend"
CANDIDATE_FILE = "submit_20260506_non_s4_temporal_microblend.csv"

READY_DIR = ROOT / "submissions" / "ready"
CLIP_MIN = 0.02
CLIP_MAX = 0.98
MAX_TARGET_TEST_DRIFT = 0.005
WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 1001), 3)
FROZEN_TARGETS = {"S4"}
MIN_NON_S4_IMPROVED = 2


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
    keys = raw[KEY_COLUMNS].copy()
    labels = raw[TARGET_COLUMNS].astype(float).copy()
    preds = _prediction_columns(raw, allow_plain_targets=False)
    return keys, labels, preds


def _load_test(run_name: str) -> pd.DataFrame:
    raw = pd.read_csv(MODELS_DIR / f"test_predictions_{run_name}.csv")
    return _prediction_columns(raw, allow_plain_targets=True)


def _score_one(labels: pd.DataFrame, target: str, prediction: np.ndarray | pd.Series) -> float:
    return binary_log_loss(labels[target].to_numpy(float), np.asarray(prediction, dtype=float))


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {target: _score_one(labels, target, predictions[target]) for target in TARGET_COLUMNS}
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _best_weight_for_target(
    *,
    target: str,
    labels: pd.DataFrame,
    current_oof: pd.DataFrame,
    source_oof: pd.DataFrame,
    current_test: pd.DataFrame,
    source_test: pd.DataFrame,
) -> dict[str, Any]:
    current_score = _score_one(labels, target, current_oof[target])
    current_test_values = current_test[target].to_numpy(float)
    source_test_values = source_test[target].to_numpy(float)
    full_test_drift = float(np.mean(np.abs(source_test_values - current_test_values)))

    if target in FROZEN_TARGETS or full_test_drift == 0.0:
        return {
            "target": target,
            "weight": 0.0,
            "score": current_score,
            "delta": 0.0,
            "test_drift": 0.0,
            "reason": "frozen" if target in FROZEN_TARGETS else "no test effect",
        }

    best = {
        "target": target,
        "weight": 0.0,
        "score": current_score,
        "delta": 0.0,
        "test_drift": 0.0,
        "reason": "no improving drift-safe weight",
    }
    current_oof_values = current_oof[target].to_numpy(float)
    source_oof_values = source_oof[target].to_numpy(float)
    max_weight_by_drift = min(1.0, MAX_TARGET_TEST_DRIFT / full_test_drift)

    for weight in WEIGHT_GRID:
        if weight > max_weight_by_drift + 1e-12:
            break
        blended = (1.0 - weight) * current_oof_values + weight * source_oof_values
        score = _score_one(labels, target, blended)
        delta = float(score - current_score)
        if delta < best["delta"] - 1e-12:
            best = {
                "target": target,
                "weight": float(weight),
                "score": float(score),
                "delta": delta,
                "test_drift": float(weight * full_test_drift),
                "reason": "selected",
            }

    # Guard against OOF-only noise: if the best move is effectively zero, do not move.
    if best["delta"] >= -1e-12:
        best.update({"weight": 0.0, "score": current_score, "delta": 0.0, "test_drift": 0.0})
    return best


def _blend_predictions(current: pd.DataFrame, source: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    output = current.copy()
    for target in TARGET_COLUMNS:
        weight = weights.get(target, 0.0)
        output[target] = (1.0 - weight) * current[target].astype(float) + weight * source[target].astype(float)
    return output[TARGET_COLUMNS]


def _write_candidate_artifacts(
    *,
    keys: pd.DataFrame,
    labels: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    candidate_test: pd.DataFrame,
    target_decisions: list[dict[str, Any]],
    current_scores: dict[str, float],
    candidate_scores: dict[str, float],
    current_test: pd.DataFrame,
) -> Path:
    ensure_runtime_dirs()
    READY_DIR.mkdir(parents=True, exist_ok=True)

    oof_export = keys.copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].astype(float)
    oof_export["split_scheme"] = "public_stratified"
    oof_export["model_family"] = CANDIDATE_RUN
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = candidate_oof[target].to_numpy(float)
    oof_path = OOF_DIR / f"oof_predictions_{CANDIDATE_RUN}.parquet"
    oof_export.to_parquet(oof_path, index=False)

    test_path = MODELS_DIR / f"test_predictions_{CANDIDATE_RUN}.csv"
    candidate_test[TARGET_COLUMNS].to_csv(test_path, index=False)

    submission = load_submission_template()[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = np.clip(candidate_test[target].to_numpy(float), CLIP_MIN, CLIP_MAX)
    submission_path = READY_DIR / CANDIDATE_FILE
    submission.to_csv(submission_path, index=False)

    weights = {row["target"]: float(row["weight"]) for row in target_decisions}
    drift = {
        target: float(np.mean(np.abs(candidate_test[target].to_numpy(float) - current_test[target].to_numpy(float))))
        for target in TARGET_COLUMNS
    }
    improved_non_s4 = [
        target for target in TARGET_COLUMNS
        if target != "S4" and candidate_scores[target] < current_scores[target]
    ]
    target_delta = {
        target: float(candidate_scores[target] - current_scores[target])
        for target in TARGET_COLUMNS
    }
    approved_by_internal_guard = (
        len(improved_non_s4) >= MIN_NON_S4_IMPROVED
        and max(max(delta, 0.0) for delta in target_delta.values()) <= 5e-5
        and max(drift.values()) <= MAX_TARGET_TEST_DRIFT + 1e-12
        and any(abs(value) > 1e-12 for value in weights.values())
    )

    summary = {
        "candidate_run": CANDIDATE_RUN,
        "candidate_file": str(submission_path),
        "current_run": CURRENT_RUN,
        "source_run": SOURCE_RUN,
        "clip_range": [CLIP_MIN, CLIP_MAX],
        "target_decisions": target_decisions,
        "weights": weights,
        "improved_non_s4_targets": improved_non_s4,
        "target_delta_candidate_minus_current": target_delta,
        "test_mean_abs_drift": drift,
        "current_scores": current_scores,
        "candidate_scores": candidate_scores,
        "approved_by_internal_guard": bool(approved_by_internal_guard),
        "self_critique": [
            "The source full temporal-push submission failed public, so S4 is frozen and every non-S4 target is drift-capped.",
            "This is still a small public-risk experiment: OOF improvement is not a public guarantee.",
            "If the official guard fails, the ready CSV must be removed and current best remains the upload file.",
        ],
        "artifacts": {
            "oof_path": str(oof_path),
            "test_prediction_path": str(test_path),
            "submission_path": str(submission_path),
        },
    }
    summary_path = FEATURES_DIR / f"{CANDIDATE_RUN}_summary.json"
    write_json(summary_path, summary)
    weights_path = FEATURES_DIR / f"{CANDIDATE_RUN}_weights.json"
    write_json(weights_path, {"weights": weights})

    report_lines = [
        "# 2026-05-06 non-S4 temporal microblend",
        "",
        "## Decision",
        "",
        f"- Candidate file: `{submission_path}`",
        f"- Current anchor: `{CURRENT_RUN}`",
        f"- Source signal: `{SOURCE_RUN}`",
        f"- Internal guard: `{'PASS' if approved_by_internal_guard else 'FAIL'}`",
        f"- Candidate OOF mean: `{candidate_scores['mean']:.9f}`",
        f"- Current OOF mean: `{current_scores['mean']:.9f}`",
        f"- Mean delta: `{candidate_scores['mean'] - current_scores['mean']:.9f}`",
        "",
        "## Why this is different from failed S4 pushes",
        "",
        "- S4 is fixed at the current public-best prediction.",
        "- Q1/Q2/Q3/S1/S2/S3 may move only when target OOF improves.",
        "- Every moved target is capped to mean absolute test drift `<= 0.005`.",
        "- The candidate name avoids blocked axes: no S4 extension, no S4 up/down, no QS head/micro/scal, no XGB guard003/005.",
        "",
        "## Target weights",
        "",
        "| Target | Weight | OOF delta | Test drift | Reason |",
        "|---|---:|---:|---:|---|",
    ]
    for row in target_decisions:
        report_lines.append(
            f"| `{row['target']}` | `{row['weight']:.3f}` | `{row['delta']:.9f}` | `{row['test_drift']:.9f}` | {row['reason']} |"
        )
    report_lines.extend(
        [
            "",
            "## Self-critique",
            "",
            "1. The full temporal-push direction already failed public, so this candidate must not reuse its S4 movement.",
            "2. The expected lift is small. It is a safer 5.7-direction probe, not a guaranteed jump to 0.57.",
            "3. If public worsens, the lesson is that non-S4 temporal OOF gains are also split-sensitive; next work should move to feature-stable retraining instead of post-hoc blending.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "20260506_non_s4_temporal_microblend.md", "\n".join(report_lines))
    return submission_path


def main() -> None:
    current_keys, labels, current_oof = _load_oof(CURRENT_RUN)
    source_keys, source_labels, source_oof = _load_oof(SOURCE_RUN)
    if not current_keys.equals(source_keys):
        raise ValueError("Current/source OOF keys do not match.")
    if not labels.equals(source_labels):
        raise ValueError("Current/source labels do not match.")

    current_test = _load_test(CURRENT_RUN)
    source_test = _load_test(SOURCE_RUN)
    current_scores = _score_targets(labels, current_oof)

    target_decisions = [
        _best_weight_for_target(
            target=target,
            labels=labels,
            current_oof=current_oof,
            source_oof=source_oof,
            current_test=current_test,
            source_test=source_test,
        )
        for target in TARGET_COLUMNS
    ]
    weights = {row["target"]: float(row["weight"]) for row in target_decisions}
    candidate_oof = _blend_predictions(current_oof, source_oof, weights)
    candidate_test = _blend_predictions(current_test, source_test, weights)
    candidate_scores = _score_targets(labels, candidate_oof)
    path = _write_candidate_artifacts(
        keys=current_keys,
        labels=labels,
        candidate_oof=candidate_oof,
        candidate_test=candidate_test,
        target_decisions=target_decisions,
        current_scores=current_scores,
        candidate_scores=candidate_scores,
        current_test=current_test,
    )
    print(path)
    print(json.dumps({"weights": weights, "current_mean": current_scores["mean"], "candidate_mean": candidate_scores["mean"]}, indent=2))


if __name__ == "__main__":
    main()

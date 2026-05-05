#!/usr/bin/env python3
"""Build CV-filtered tight calibration candidates from the current best LGB run."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template, load_train_labels
from etri_human_challenge.paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_SUBMISSIONS_DIR, ROOT, ensure_runtime_dirs
from etri_human_challenge.utils import binary_log_loss, write_json, write_markdown


EXPERIMENT_NAME = "calib_tight_v1"
ANCHOR_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
BASE_CAL_RUN = "public_lgb_targetwise_calibrated_v1"
RUN_NAME = "public_lgb_targetwise_calib_tight_v1"
SAFE_RUN_NAME = "public_lgb_targetwise_calib_tight_v1_safe"

CURRENT_PUBLIC_BEST_FILE = "lgb_stable_calibrated.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5946792872
BASE_OOF_MEAN = 0.5649741834032371
BASE_SAFE_OOF_MEAN = 0.5663894518132937

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

# group_time calibration screen에서 확장 보정이 유효했던 target만 변경합니다.
TIGHT_PARAM_OVERRIDES: dict[str, dict[str, float]] = {
    "Q3": {"temp": 1.30, "bias": -0.21},
    "S1": {"temp": 1.30, "bias": -0.34},
    "S2": {"temp": 1.35, "bias": -0.27},
}


def _now_kst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _logit(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _transform(values: np.ndarray | pd.Series, temp: float, bias: float) -> np.ndarray:
    return np.clip(_sigmoid(temp * _logit(values) + bias), CLIP_MIN, CLIP_MAX)


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _validate_inputs(anchor_raw: pd.DataFrame, anchor_test: pd.DataFrame) -> None:
    train = load_train_labels()
    sample = load_submission_template()
    if len(train) != 450:
        raise ValueError(f"Unexpected train row count: {len(train)}")
    if len(sample) != 250:
        raise ValueError(f"Unexpected test row count: {len(sample)}")
    missing_train = [target for target in TARGET_COLUMNS if target not in train.columns]
    if missing_train:
        raise ValueError(f"Missing train target columns: {missing_train}")
    missing_oof = [target for target in TARGET_COLUMNS if target not in anchor_raw.columns]
    if missing_oof:
        raise ValueError(f"Missing OOF target columns: {missing_oof}")
    missing_test = [target for target in TARGET_COLUMNS if target not in anchor_test.columns]
    if missing_test:
        raise ValueError(f"Missing test prediction columns: {missing_test}")


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    anchor_raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{ANCHOR_RUN}.parquet")
    labels = anchor_raw[TARGET_COLUMNS].copy()
    anchor_oof = pd.DataFrame({target: anchor_raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    anchor_test = pd.read_csv(MODELS_DIR / f"test_predictions_{ANCHOR_RUN}.csv")[TARGET_COLUMNS].astype(float)
    summary = json.loads((FEATURES_DIR / f"{BASE_CAL_RUN}_summary.json").read_text())
    _validate_inputs(anchor_raw, anchor_test)
    return labels, anchor_oof, anchor_test, anchor_raw[KEY_COLUMNS].copy(), summary["params"]


def _apply_candidate(
    frame: pd.DataFrame,
    base_params: dict[str, dict[str, float]],
    *,
    blend: float,
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for target in TARGET_COLUMNS:
        raw = frame[target].to_numpy(float)
        base = _transform(raw, base_params[target]["temp"], base_params[target]["bias"])
        if target in TIGHT_PARAM_OVERRIDES:
            params = TIGHT_PARAM_OVERRIDES[target]
            tight = _transform(raw, params["temp"], params["bias"])
            output[target] = np.clip((1.0 - blend) * base + blend * tight, CLIP_MIN, CLIP_MAX)
        else:
            output[target] = base
    return output[TARGET_COLUMNS]


def _make_submission(path: Path, predictions: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = np.clip(predictions[target].to_numpy(float), CLIP_MIN, CLIP_MAX)
    if submission.shape != (250, 10):
        raise ValueError(f"Invalid submission shape: {submission.shape}")
    if submission.columns.tolist() != template.columns.tolist():
        raise ValueError("Submission columns do not match sample_submission.")
    values = submission[TARGET_COLUMNS].to_numpy(float)
    if values.min() < CLIP_MIN or values.max() > CLIP_MAX:
        raise ValueError(f"Predictions outside [{CLIP_MIN}, {CLIP_MAX}].")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def _write_prediction_artifacts(
    run_name: str,
    keys: pd.DataFrame,
    labels: pd.DataFrame,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    scores: dict[str, float],
    blend: float,
) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "public_stratified_posthoc_tight_calibration"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(
        FEATURES_DIR / f"{run_name}_summary.json",
        {
            "run_name": run_name,
            "base_run": BASE_CAL_RUN,
            "scores": scores,
            "tight_param_overrides": TIGHT_PARAM_OVERRIDES,
            "blend": blend,
        },
    )


def _append_experiment(name: str, scores: dict[str, float], submission: Path, blend: float, notes: str) -> None:
    path = LOG_DIR / "experiments.csv"
    columns = [
        "timestamp", "experiment_name", "validation_scheme", "seeds", "total_oof_logloss",
        "target_logloss_Q1", "target_logloss_Q2", "target_logloss_Q3", "target_logloss_S1",
        "target_logloss_S2", "target_logloss_S3", "target_logloss_S4", "submission_file",
        "feature_view_by_target", "notes",
    ]
    rows = _read_csv_rows(path)
    rows = [row for row in rows if row.get("experiment_name") != name]
    rows.append(
        {
            "timestamp": _now_kst(),
            "experiment_name": name,
            "validation_scheme": "public_stratified_posthoc_tight_calibration",
            "seeds": "posthoc",
            "total_oof_logloss": scores["mean"],
            "target_logloss_Q1": scores["Q1"],
            "target_logloss_Q2": scores["Q2"],
            "target_logloss_Q3": scores["Q3"],
            "target_logloss_S1": scores["S1"],
            "target_logloss_S2": scores["S2"],
            "target_logloss_S3": scores["S3"],
            "target_logloss_S4": scores["S4"],
            "submission_file": str(submission.relative_to(ROOT)),
            "feature_view_by_target": json.dumps({"overrides": TIGHT_PARAM_OVERRIDES, "blend": blend}, sort_keys=True),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(tight_scores: dict[str, float], safe_scores: dict[str, float]) -> None:
    rows = [
        {
            "rank": 0,
            "candidate": CURRENT_PUBLIC_BEST_FILE,
            "oof_mean": BASE_OOF_MEAN,
            "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}",
            "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}",
            "notes": "Current public best; stable_tuned plus per-target logit calibration.",
        },
        {
            "rank": 1,
            "candidate": "lgb_calib_tight.csv",
            "oof_mean": tight_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/lgb_calib_tight.csv",
            "notes": "Next candidate; tighter calibration only on Q3/S1/S2.",
        },
        {
            "rank": 2,
            "candidate": "lgb_calib_tight_safe.csv",
            "oof_mean": safe_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/lgb_calib_tight_safe.csv",
            "notes": "Backup; 60% tight calibration blended with current best calibration.",
        },
        {
            "rank": 3,
            "candidate": "lgb_stable_calibrated_safe.csv",
            "oof_mean": BASE_SAFE_OOF_MEAN,
            "public_score": "",
            "submission_file": "submissions/ready/lgb_stable_calibrated_safe.csv",
            "notes": "Older conservative backup; keep only if both tight candidates fail.",
        },
        {
            "rank": 4,
            "candidate": "lgb_stable_tuned.csv",
            "oof_mean": 0.5687692970981916,
            "public_score": "0.5956332255",
            "submission_file": "submissions/archive/2026-04-29_cleanup/lgb_stable_tuned.csv",
            "notes": "Archived previous best.",
        },
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(tight_scores: dict[str, float], safe_scores: dict[str, float]) -> None:
    lines = [
        "# Submissions", "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.", "",
        "## Current Best", "",
        f"- Public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`", "",
        "## Recommended Order", "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_PUBLIC_BEST_FILE}` | `{BASE_OOF_MEAN:.6f}` | `{CURRENT_PUBLIC_BEST_SCORE:.10f}` | 현재 best |",
        f"| 1 | `ready/lgb_calib_tight.csv` | `{tight_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/lgb_calib_tight_safe.csv` | `{safe_scores['mean']:.6f}` |  | 1순위 실패 시 백업 |",
        f"| 3 | `ready/lgb_stable_calibrated_safe.csv` | `{BASE_SAFE_OOF_MEAN:.6f}` |  | 기존 보수 백업 |",
        "| 4 | `archive/2026-04-29_cleanup/lgb_stable_tuned.csv` | `0.568769` | `0.5956332255` | 이전 best |", "",
        "## Notes", "",
        "- `lgb_calib_tight`: current best calibration에서 `Q3/S1/S2`만 한 단계 더 강하게 보정.",
        "- `lgb_calib_tight_safe`: `lgb_calib_tight` 변화분을 60%만 반영한 보수 버전.",
        "- 기존 public best 파일은 덮어쓰지 않았습니다.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(
    base_scores: dict[str, float],
    tight_scores: dict[str, float],
    safe_scores: dict[str, float],
    tight_test: pd.DataFrame,
    safe_test: pd.DataFrame,
) -> None:
    lines = [
        "# Submission Report: calib_tight_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Current best OOF: `{BASE_OOF_MEAN:.6f}`",
        f"- Tight OOF: `{tight_scores['mean']:.6f}`",
        f"- Tight safe OOF: `{safe_scores['mean']:.6f}`", "",
        "## Changed Targets", "",
        "| Target | Temp | Bias | Base OOF | Tight OOF | Delta | Tight test mean | Safe test mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        if target in TIGHT_PARAM_OVERRIDES:
            params = TIGHT_PARAM_OVERRIDES[target]
            lines.append(
                f"| {target} | {params['temp']:.2f} | {params['bias']:.2f} | "
                f"`{base_scores[target]:.6f}` | `{tight_scores[target]:.6f}` | "
                f"`{tight_scores[target] - base_scores[target]:+.6f}` | "
                f"`{tight_test[target].mean():.4f}` | `{safe_test[target].mean():.4f}` |"
            )
    lines.extend(
        [
            "",
            "## Target Scores",
            "",
            "| Target | Base | Tight | Tight Safe |",
            "|---|---:|---:|---:|",
        ]
    )
    for target in TARGET_COLUMNS:
        lines.append(f"| {target} | `{base_scores[target]:.6f}` | `{tight_scores[target]:.6f}` | `{safe_scores[target]:.6f}` |")
    write_markdown(REPORT_SUBMISSIONS_DIR / "calib_tight_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    labels, anchor_oof, anchor_test, keys, base_params = _load_inputs()

    base_oof = _apply_candidate(anchor_oof, base_params, blend=0.0)
    tight_oof = _apply_candidate(anchor_oof, base_params, blend=1.0)
    tight_test = _apply_candidate(anchor_test, base_params, blend=1.0)
    safe_oof = _apply_candidate(anchor_oof, base_params, blend=0.6)
    safe_test = _apply_candidate(anchor_test, base_params, blend=0.6)

    base_scores = _score_targets(labels, base_oof)
    tight_scores = _score_targets(labels, tight_oof)
    safe_scores = _score_targets(labels, safe_oof)

    tight_path = READY_DIR / "lgb_calib_tight.csv"
    safe_path = READY_DIR / "lgb_calib_tight_safe.csv"
    _make_submission(tight_path, tight_test)
    _make_submission(safe_path, safe_test)

    _write_prediction_artifacts(RUN_NAME, keys, labels, tight_oof, tight_test, tight_scores, 1.0)
    _write_prediction_artifacts(SAFE_RUN_NAME, keys, labels, safe_oof, safe_test, safe_scores, 0.6)
    _append_experiment(
        EXPERIMENT_NAME,
        tight_scores,
        tight_path,
        1.0,
        "Tighter per-target calibration on Q3/S1/S2 only; other targets frozen at current best.",
    )
    _append_experiment(
        f"{EXPERIMENT_NAME}_safe",
        safe_scores,
        safe_path,
        0.6,
        "60% blend between current best calibration and tight Q3/S1/S2 calibration.",
    )
    _write_candidate_scores(tight_scores, safe_scores)
    _write_readme(tight_scores, safe_scores)
    _write_report(base_scores, tight_scores, safe_scores, tight_test, safe_test)

    print("=== calib_tight_v1 ===")
    print(f"tight: {tight_path} OOF={tight_scores['mean']:.6f}")
    print(f"safe: {safe_path} OOF={safe_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        delta = tight_scores[target] - base_scores[target]
        print(f"{target}: base={base_scores[target]:.6f} tight={tight_scores[target]:.6f} delta={delta:+.6f}")


if __name__ == "__main__":
    main()

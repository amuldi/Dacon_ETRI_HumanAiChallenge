#!/usr/bin/env python3
"""Build micro calibration candidates from the current public-best tight calibration."""

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


EXPERIMENT_NAME = "calib_micro_v1"
ANCHOR_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
BASE_CAL_RUN = "public_lgb_targetwise_calibrated_v1"
RUN_NAME = "public_lgb_targetwise_calib_micro_v1"
SAFE_RUN_NAME = "public_lgb_targetwise_calib_micro_v1_safe"

CURRENT_PUBLIC_BEST_FILE = "lgb_calib_tight.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5944158654
CURRENT_BEST_OOF_MEAN = 0.5645484761869698

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

TIGHT_PARAMS: dict[str, dict[str, float]] = {
    "Q3": {"temp": 1.30, "bias": -0.21},
    "S1": {"temp": 1.30, "bias": -0.34},
    "S2": {"temp": 1.35, "bias": -0.27},
}

# 1순위: Q2는 건드리지 않고 public 리스크가 낮은 target만 미세 조정합니다.
MICRO_PARAMS: dict[str, dict[str, float]] = {
    "Q3": {"temp": 1.34, "bias": -0.22},
    "S1": {"temp": 1.32, "bias": -0.35},
    "S2": {"temp": 1.38, "bias": -0.29},
    "S3": {"temp": 1.26, "bias": -0.24},
    "S4": {"temp": 1.25, "bias": -0.08},
}

# 백업: public에서 이미 검증된 Q3/S1/S2 축만 유지합니다.
SAFE_MICRO_PARAMS: dict[str, dict[str, float]] = {
    "Q3": {"temp": 1.34, "bias": -0.22},
    "S1": {"temp": 1.32, "bias": -0.35},
    "S2": {"temp": 1.38, "bias": -0.29},
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


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


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


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    anchor_raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{ANCHOR_RUN}.parquet")
    labels = anchor_raw[TARGET_COLUMNS].copy()
    anchor_oof = pd.DataFrame({target: anchor_raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    anchor_test = pd.read_csv(MODELS_DIR / f"test_predictions_{ANCHOR_RUN}.csv")[TARGET_COLUMNS].astype(float)
    base_params = json.loads((FEATURES_DIR / f"{BASE_CAL_RUN}_summary.json").read_text())["params"]
    _validate_inputs(anchor_raw, anchor_test)
    return labels, anchor_oof, anchor_test, anchor_raw[KEY_COLUMNS].copy(), base_params


def _current_params(base_params: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    params = {target: {"temp": float(base_params[target]["temp"]), "bias": float(base_params[target]["bias"])} for target in TARGET_COLUMNS}
    params.update(TIGHT_PARAMS)
    return params


def _apply_params(frame: pd.DataFrame, params: dict[str, dict[str, float]]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for target in TARGET_COLUMNS:
        output[target] = _transform(frame[target], params[target]["temp"], params[target]["bias"])
    return output[TARGET_COLUMNS]


def _make_candidate_params(base_params: dict[str, dict[str, float]], overrides: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    params = _current_params(base_params)
    params.update(overrides)
    return params


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
    params: dict[str, dict[str, float]],
    notes: str,
) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "public_stratified_posthoc_micro_calibration"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "params": params, "notes": notes})


def _append_experiment(name: str, scores: dict[str, float], submission: Path, params: dict[str, dict[str, float]], notes: str) -> None:
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
            "validation_scheme": "public_stratified_posthoc_micro_calibration",
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
            "feature_view_by_target": json.dumps(params, sort_keys=True),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(micro_scores: dict[str, float], safe_scores: dict[str, float]) -> None:
    rows = [
        {
            "rank": 0,
            "candidate": CURRENT_PUBLIC_BEST_FILE,
            "oof_mean": CURRENT_BEST_OOF_MEAN,
            "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}",
            "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}",
            "notes": "Current public best; tight calibration on Q3/S1/S2.",
        },
        {
            "rank": 1,
            "candidate": "lgb_calib_micro.csv",
            "oof_mean": micro_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/lgb_calib_micro.csv",
            "notes": "Next candidate; micro calibration on Q3/S1/S2/S3/S4, Q2 frozen.",
        },
        {
            "rank": 2,
            "candidate": "lgb_calib_micro_safe.csv",
            "oof_mean": safe_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/lgb_calib_micro_safe.csv",
            "notes": "Backup; micro calibration on previously validated Q3/S1/S2 only.",
        },
        {
            "rank": 3,
            "candidate": "lgb_stable_calibrated.csv",
            "oof_mean": 0.5649741834032371,
            "public_score": "0.5946792872",
            "submission_file": "submissions/ready/lgb_stable_calibrated.csv",
            "notes": "Previous public best.",
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


def _write_readme(micro_scores: dict[str, float], safe_scores: dict[str, float]) -> None:
    lines = [
        "# Submissions", "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.", "",
        "## Current Best", "",
        f"- Public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`", "",
        "## Recommended Order", "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_PUBLIC_BEST_FILE}` | `{CURRENT_BEST_OOF_MEAN:.6f}` | `{CURRENT_PUBLIC_BEST_SCORE:.10f}` | 현재 best |",
        f"| 1 | `ready/lgb_calib_micro.csv` | `{micro_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/lgb_calib_micro_safe.csv` | `{safe_scores['mean']:.6f}` |  | 1순위 실패 시 백업 |",
        "| 3 | `ready/lgb_stable_calibrated.csv` | `0.564974` | `0.5946792872` | 이전 best |",
        "| 4 | `archive/2026-04-29_cleanup/lgb_stable_tuned.csv` | `0.568769` | `0.5956332255` | 이전 best |", "",
        "## Notes", "",
        "- `lgb_calib_micro`: `Q3/S1/S2/S3/S4`만 micro calibration, `Q2`는 public 리스크 때문에 고정.",
        "- `lgb_calib_micro_safe`: public에서 이미 성공한 `Q3/S1/S2` 축만 micro calibration.",
        "- 기존 current best 파일은 덮어쓰지 않았습니다.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(
    current_scores: dict[str, float],
    micro_scores: dict[str, float],
    safe_scores: dict[str, float],
    micro_test: pd.DataFrame,
    safe_test: pd.DataFrame,
) -> None:
    lines = [
        "# Submission Report: calib_micro_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Current best OOF: `{CURRENT_BEST_OOF_MEAN:.6f}`",
        f"- Micro OOF: `{micro_scores['mean']:.6f}`",
        f"- Micro safe OOF: `{safe_scores['mean']:.6f}`", "",
        "## Changed Targets", "",
        "| Target | Current OOF | Micro OOF | Safe OOF | Micro Delta | Micro test mean | Safe test mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        if target in MICRO_PARAMS or target in SAFE_MICRO_PARAMS:
            lines.append(
                f"| {target} | `{current_scores[target]:.6f}` | `{micro_scores[target]:.6f}` | "
                f"`{safe_scores[target]:.6f}` | `{micro_scores[target] - current_scores[target]:+.6f}` | "
                f"`{micro_test[target].mean():.4f}` | `{safe_test[target].mean():.4f}` |"
            )
    lines.extend(["", "## Parameters", "", "| Target | Micro temp | Micro bias | Safe temp | Safe bias |", "|---|---:|---:|---:|---:|"])
    for target in TARGET_COLUMNS:
        if target in MICRO_PARAMS or target in SAFE_MICRO_PARAMS:
            micro = MICRO_PARAMS.get(target, {})
            safe = SAFE_MICRO_PARAMS.get(target, {})
            lines.append(
                f"| {target} | `{micro.get('temp', '')}` | `{micro.get('bias', '')}` | "
                f"`{safe.get('temp', '')}` | `{safe.get('bias', '')}` |"
            )
    write_markdown(REPORT_SUBMISSIONS_DIR / "calib_micro_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    labels, anchor_oof, anchor_test, keys, base_params = _load_inputs()

    current_params = _current_params(base_params)
    micro_params = _make_candidate_params(base_params, MICRO_PARAMS)
    safe_params = _make_candidate_params(base_params, SAFE_MICRO_PARAMS)

    current_oof = _apply_params(anchor_oof, current_params)
    micro_oof = _apply_params(anchor_oof, micro_params)
    micro_test = _apply_params(anchor_test, micro_params)
    safe_oof = _apply_params(anchor_oof, safe_params)
    safe_test = _apply_params(anchor_test, safe_params)

    current_scores = _score_targets(labels, current_oof)
    micro_scores = _score_targets(labels, micro_oof)
    safe_scores = _score_targets(labels, safe_oof)

    micro_path = READY_DIR / "lgb_calib_micro.csv"
    safe_path = READY_DIR / "lgb_calib_micro_safe.csv"
    _make_submission(micro_path, micro_test)
    _make_submission(safe_path, safe_test)

    _write_prediction_artifacts(RUN_NAME, keys, labels, micro_oof, micro_test, micro_scores, micro_params, "Micro calibration on Q3/S1/S2/S3/S4; Q2 frozen.")
    _write_prediction_artifacts(SAFE_RUN_NAME, keys, labels, safe_oof, safe_test, safe_scores, safe_params, "Micro calibration on Q3/S1/S2 only.")
    _append_experiment(EXPERIMENT_NAME, micro_scores, micro_path, micro_params, "Micro calibration on Q3/S1/S2/S3/S4 with Q2 frozen.")
    _append_experiment(f"{EXPERIMENT_NAME}_safe", safe_scores, safe_path, safe_params, "Safe micro calibration on Q3/S1/S2 only.")
    _write_candidate_scores(micro_scores, safe_scores)
    _write_readme(micro_scores, safe_scores)
    _write_report(current_scores, micro_scores, safe_scores, micro_test, safe_test)

    print("=== calib_micro_v1 ===")
    print(f"micro: {micro_path} OOF={micro_scores['mean']:.6f}")
    print(f"safe: {safe_path} OOF={safe_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        delta = micro_scores[target] - current_scores[target]
        print(f"{target}: current={current_scores[target]:.6f} micro={micro_scores[target]:.6f} delta={delta:+.6f}")


if __name__ == "__main__":
    main()

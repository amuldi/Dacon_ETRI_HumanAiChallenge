#!/usr/bin/env python3
"""Create per-target logit calibration candidates from the stable_tuned anchor."""

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
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_SUBMISSIONS_DIR, ROOT, ensure_runtime_dirs
from etri_human_challenge.utils import binary_log_loss, write_json, write_markdown


EXPERIMENT_NAME = "calibrated_v1"
ANCHOR_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
RUN_NAME = "public_lgb_targetwise_calibrated_v1"
CONSERVATIVE_RUN_NAME = "public_lgb_targetwise_calibrated_v1_conservative"

CURRENT_PUBLIC_BEST_FILE = "next_stable_tuned.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5956332255
PREVIOUS_PUBLIC_BEST_SCORE = 0.5960566585
ANCHOR_OOF_MEAN = 0.5687692970981916

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"


def _now_kst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _transform(values: np.ndarray, temp: float, bias: float) -> np.ndarray:
    return np.clip(_sigmoid(temp * _logit(values) + bias), CLIP_MIN, CLIP_MAX)


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _load_anchor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{ANCHOR_RUN}.parquet")
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(MODELS_DIR / f"test_predictions_{ANCHOR_RUN}.csv")[TARGET_COLUMNS].astype(float)
    return labels, oof, test, raw[KEY_COLUMNS].copy()


def _find_params(labels: pd.DataFrame, oof: pd.DataFrame) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        y = labels[target].to_numpy(float)
        p = oof[target].to_numpy(float)
        best = {
            "score": binary_log_loss(y, p),
            "temp": 1.0,
            "bias": 0.0,
        }
        for temp in np.linspace(0.70, 1.20, 26):
            for bias in np.linspace(-0.20, 0.20, 41):
                pred = _transform(p, float(temp), float(bias))
                score = binary_log_loss(y, pred)
                if score < best["score"]:
                    best = {"score": float(score), "temp": float(temp), "bias": float(bias)}
        params[target] = best
    return params


def _apply_params(frame: pd.DataFrame, params: dict[str, dict[str, float]], *, blend: float) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for target in TARGET_COLUMNS:
        raw = frame[target].to_numpy(float)
        calibrated = _transform(raw, params[target]["temp"], params[target]["bias"])
        output[target] = np.clip((1.0 - blend) * raw + blend * calibrated, CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _make_submission(path: Path, predictions: pd.DataFrame) -> None:
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


def _write_prediction_artifacts(run_name: str, keys: pd.DataFrame, labels: pd.DataFrame, oof: pd.DataFrame, test: pd.DataFrame, scores: dict[str, float], params: dict[str, Any], blend: float) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "public_stratified_posthoc_calibration"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "params": params, "blend": blend})


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


def _update_public_scores() -> None:
    path = LOG_DIR / "public_scores.csv"
    columns = ["timestamp", "submission_file", "experiment_name", "public_score", "delta_vs_best", "notes"]
    rows = _read_csv_rows(path)
    rows = [row for row in rows if row.get("experiment_name") != "guarded_v2_stable_tuned"]
    rows.append(
        {
            "timestamp": "2026-04-29 00:57:34",
            "submission_file": CURRENT_PUBLIC_BEST_FILE,
            "experiment_name": "guarded_v2_stable_tuned",
            "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}",
            "delta_vs_best": f"{CURRENT_PUBLIC_BEST_SCORE - PREVIOUS_PUBLIC_BEST_SCORE:.10f}",
            "notes": "New current best; stable feature subset + target-specific LGB params.",
        }
    )
    _write_csv_rows(path, rows, columns)


def _append_experiment(name: str, scores: dict[str, float], submission: Path, params: dict[str, Any], notes: str) -> None:
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
            "validation_scheme": "public_stratified_posthoc_calibration",
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


def _write_candidate_scores(primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    rows = [
        {"rank": 0, "candidate": CURRENT_PUBLIC_BEST_FILE, "oof_mean": ANCHOR_OOF_MEAN, "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}", "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}", "notes": "Current public best."},
        {"rank": 1, "candidate": "next_calibrated.csv", "oof_mean": primary_scores["mean"], "public_score": "", "submission_file": "submissions/ready/next_calibrated.csv", "notes": "Per-target logit calibration from stable_tuned anchor."},
        {"rank": 2, "candidate": "next_calibrated_conservative.csv", "oof_mean": conservative_scores["mean"], "public_score": "", "submission_file": "submissions/ready/next_calibrated_conservative.csv", "notes": "50% blended calibration; lower-risk backup."},
        {"rank": 3, "candidate": "next_target_history.csv", "oof_mean": 0.5687692970981916, "public_score": "", "submission_file": "submissions/ready/next_target_history.csv", "notes": "No meaningful OOF gain; do not submit before calibration."},
        {"rank": 4, "candidate": "00_best_guarded.csv", "oof_mean": 0.5721963000561784, "public_score": "0.5960566585", "submission_file": "submissions/ready/00_best_guarded.csv", "notes": "Previous public best anchor."},
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    lines = [
        "# Submissions", "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.", "",
        "## Current Best", "",
        f"- Public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`", "",
        "## Recommended Order", "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_PUBLIC_BEST_FILE}` | `{ANCHOR_OOF_MEAN:.6f}` | `{CURRENT_PUBLIC_BEST_SCORE:.10f}` | 현재 best |",
        f"| 1 | `ready/next_calibrated.csv` | `{primary_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/next_calibrated_conservative.csv` | `{conservative_scores['mean']:.6f}` |  | 1순위 실패 시 제출 |",
        "| 3 | `ready/next_target_history.csv` | `0.568769` |  | OOF 이득 없음, 보류 |",
        "| 4 | `ready/00_best_guarded.csv` | `0.572196` | `0.5960566585` | 이전 best |", "",
        "## Notes", "",
        "- `calibrated_v1`: per-target logit temperature/bias calibration.",
        "- `target_history_prior_v1`: OOF 이득이 거의 없어 제출 우선순위에서 내림.",
    ]
    (ROOT / "submissions" / "README.md").write_text("\n".join(lines) + "\n")


def _write_report(params: dict[str, Any], primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    lines = [
        "# Submission Report: calibrated_v1", "",
        f"- Anchor public: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Anchor OOF: `{ANCHOR_OOF_MEAN:.6f}`",
        f"- Primary OOF: `{primary_scores['mean']:.6f}`",
        f"- Conservative OOF: `{conservative_scores['mean']:.6f}`", "",
        "| Target | Temp | Bias | Calibrated OOF |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        item = params[target]
        lines.append(f"| {target} | {item['temp']:.2f} | {item['bias']:.2f} | {item['score']:.6f} |")
    write_markdown(REPORT_SUBMISSIONS_DIR / "calibrated_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    labels, anchor_oof, anchor_test, keys = _load_anchor()
    params = _find_params(labels, anchor_oof)
    primary_oof = _apply_params(anchor_oof, params, blend=1.0)
    primary_test = _apply_params(anchor_test, params, blend=1.0)
    conservative_oof = _apply_params(anchor_oof, params, blend=0.5)
    conservative_test = _apply_params(anchor_test, params, blend=0.5)
    primary_scores = _score_targets(labels, primary_oof)
    conservative_scores = _score_targets(labels, conservative_oof)

    primary_path = READY_DIR / "next_calibrated.csv"
    conservative_path = READY_DIR / "next_calibrated_conservative.csv"
    _make_submission(primary_path, primary_test)
    _make_submission(conservative_path, conservative_test)
    _write_prediction_artifacts(RUN_NAME, keys, labels, primary_oof, primary_test, primary_scores, params, 1.0)
    _write_prediction_artifacts(CONSERVATIVE_RUN_NAME, keys, labels, conservative_oof, conservative_test, conservative_scores, params, 0.5)
    _update_public_scores()
    _append_experiment(EXPERIMENT_NAME, primary_scores, primary_path, params, "Per-target logit temperature/bias calibration from stable_tuned OOF.")
    _append_experiment(f"{EXPERIMENT_NAME}_conservative", conservative_scores, conservative_path, params, "50% blend between stable_tuned and calibrated probabilities.")
    _write_candidate_scores(primary_scores, conservative_scores)
    _write_readme(primary_scores, conservative_scores)
    _write_report(params, primary_scores, conservative_scores)

    print("=== calibrated_v1 ===")
    print(f"primary: {primary_path} OOF={primary_scores['mean']:.6f}")
    print(f"conservative: {conservative_path} OOF={conservative_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(
            f"{target}: temp={params[target]['temp']:.2f} bias={params[target]['bias']:.2f} "
            f"score={params[target]['score']:.6f}"
        )


if __name__ == "__main__":
    main()

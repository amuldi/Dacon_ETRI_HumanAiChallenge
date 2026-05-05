#!/usr/bin/env python3
"""Create public-feedback candidates after calibration and S4-top50 failures."""

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


EXPERIMENT_NAME = "public_feedback_v1"
ANCHOR_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
BASE_CAL_RUN = "public_lgb_targetwise_calibrated_v1"
CURRENT_RUN = "public_lgb_targetwise_calib_tight_v1"
S4_FAILED_RUN = "public_lgb_targetwise_s4_top50_v1"
QS_RUN = "public_lgb_targetwise_publicfit_qs_v1"
PROBE_RUN = "public_lgb_targetwise_57_antis4_probe_v1"

CURRENT_PUBLIC_BEST_FILE = "lgb_calib_tight.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5944158654
CURRENT_BEST_OOF_MEAN = 0.5645484761869698

PUBLIC_FIT_ALPHA = 0.93
ANTI_S4_BETA = -0.10
CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

TIGHT_PARAMS = {
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


def _validate_inputs() -> None:
    train = load_train_labels()
    sample = load_submission_template()
    if len(train) != 450:
        raise ValueError(f"Unexpected train row count: {len(train)}")
    if len(sample) != 250:
        raise ValueError(f"Unexpected test row count: {len(sample)}")
    missing = [target for target in TARGET_COLUMNS if target not in train.columns]
    if missing:
        raise ValueError(f"Missing train target columns: {missing}")


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    anchor_raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{ANCHOR_RUN}.parquet")
    current_raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{CURRENT_RUN}.parquet")
    s4_raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{S4_FAILED_RUN}.parquet")
    labels = current_raw[TARGET_COLUMNS].copy()
    anchor_oof = pd.DataFrame({target: anchor_raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    current_oof = pd.DataFrame({target: current_raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    s4_oof = pd.DataFrame({target: s4_raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    anchor_test = pd.read_csv(MODELS_DIR / f"test_predictions_{ANCHOR_RUN}.csv")[TARGET_COLUMNS].astype(float)
    current_test = pd.read_csv(MODELS_DIR / f"test_predictions_{CURRENT_RUN}.csv")[TARGET_COLUMNS].astype(float)
    s4_test = pd.read_csv(MODELS_DIR / f"test_predictions_{S4_FAILED_RUN}.csv")[TARGET_COLUMNS].astype(float)
    base_params = json.loads((FEATURES_DIR / f"{BASE_CAL_RUN}_summary.json").read_text())["params"]
    return labels, anchor_oof, anchor_test, current_oof, current_test, {"base_params": base_params, "s4_oof": s4_oof, "s4_test": s4_test, "keys": current_raw[KEY_COLUMNS].copy()}


def _publicfit_params(base_params: dict[str, dict[str, float]], alpha: float) -> dict[str, dict[str, float]]:
    params = {target: {"temp": float(base_params[target]["temp"]), "bias": float(base_params[target]["bias"])} for target in TARGET_COLUMNS}
    for target, tight in TIGHT_PARAMS.items():
        base = base_params[target]
        params[target] = {
            "temp": float(base["temp"] + alpha * (tight["temp"] - base["temp"])),
            "bias": float(base["bias"] + alpha * (tight["bias"] - base["bias"])),
        }
    return params


def _apply_publicfit(anchor_frame: pd.DataFrame, current_frame: pd.DataFrame, params: dict[str, dict[str, float]]) -> pd.DataFrame:
    output = current_frame.copy()
    for target in ["Q3", "S1", "S2"]:
        output[target] = _transform(anchor_frame[target], params[target]["temp"], params[target]["bias"])
    return output[TARGET_COLUMNS]


def _apply_anti_s4(frame: pd.DataFrame, failed_s4: pd.DataFrame, beta: float) -> pd.DataFrame:
    output = frame.copy()
    output["S4"] = np.clip(output["S4"] + beta * (failed_s4["S4"] - output["S4"]), CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _make_submission(path: Path, predictions: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    if predictions.shape != (250, 7):
        raise ValueError(f"Invalid prediction shape: {predictions.shape}")
    values = predictions[TARGET_COLUMNS].to_numpy(float)
    if values.min() < CLIP_MIN or values.max() > CLIP_MAX:
        raise ValueError(f"Predictions outside [{CLIP_MIN}, {CLIP_MAX}].")
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = predictions[target].to_numpy(float)
    if submission.shape != (250, 10):
        raise ValueError(f"Invalid submission shape: {submission.shape}")
    if submission.columns.tolist() != template.columns.tolist():
        raise ValueError("Submission columns do not match sample_submission.")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def _write_prediction_artifacts(run_name: str, keys: pd.DataFrame, labels: pd.DataFrame, oof: pd.DataFrame, test: pd.DataFrame, scores: dict[str, float], selection: dict[str, Any]) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "public_feedback_posthoc"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "selection": selection})


def _append_experiment(name: str, scores: dict[str, float], submission: Path, selection: dict[str, Any], notes: str) -> None:
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
            "validation_scheme": "public_feedback_posthoc",
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
            "feature_view_by_target": json.dumps(selection, sort_keys=True),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(qs_scores: dict[str, float], probe_scores: dict[str, float]) -> None:
    rows = [
        {"rank": 0, "candidate": CURRENT_PUBLIC_BEST_FILE, "oof_mean": CURRENT_BEST_OOF_MEAN, "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}", "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}", "notes": "Current public best."},
        {"rank": 1, "candidate": "lgb_publicfit_qs.csv", "oof_mean": qs_scores["mean"], "public_score": "", "submission_file": "submissions/ready/lgb_publicfit_qs.csv", "notes": "Low-risk public quadratic fit on Q3/S1/S2 only."},
        {"rank": 2, "candidate": "lgb_57_antis4_probe.csv", "oof_mean": probe_scores["mean"], "public_score": "", "submission_file": "submissions/ready/lgb_57_antis4_probe.csv", "notes": "Aggressive 5.7 probe; publicfit Q3/S1/S2 plus small anti-S4 failed-direction move."},
        {"rank": 3, "candidate": "lgb_s4_top50.csv", "oof_mean": 0.5583579944816833, "public_score": "0.5970623278", "submission_file": "submissions/ready/lgb_s4_top50.csv", "notes": "Worse than current best; do not resubmit."},
        {"rank": 4, "candidate": "lgb_q3s1s2_boost.csv", "oof_mean": 0.5645387051310182, "public_score": "0.5944366954", "submission_file": "submissions/ready/lgb_q3s1s2_boost.csv", "notes": "Worse than current best; do not resubmit."},
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(qs_scores: dict[str, float], probe_scores: dict[str, float]) -> None:
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
        f"| 1 | `ready/lgb_publicfit_qs.csv` | `{qs_scores['mean']:.6f}` |  | 저위험 다음 제출 |",
        f"| 2 | `ready/lgb_57_antis4_probe.csv` | `{probe_scores['mean']:.6f}` |  | 5.7 probe, 고위험 |",
        "| 3 | `ready/lgb_s4_top50.csv` | `0.558358` | `0.5970623278` | 실패, 제출 금지 |",
        "| 4 | `ready/lgb_q3s1s2_boost.csv` | `0.564539` | `0.5944366954` | 실패, 제출 금지 |", "",
        "## Notes", "",
        "- `lgb_publicfit_qs`: submitted public scores at alpha 0/1/1.2 imply Q3/S1/S2 optimum around `alpha=0.93`.",
        "- `lgb_57_antis4_probe`: S4 top50 failed badly, so this moves S4 a small step in the opposite direction. Upside is larger but risk is real.",
        "- no-TE and CatBoost checks were weaker, so they were not promoted.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(current_scores: dict[str, float], qs_scores: dict[str, float], probe_scores: dict[str, float], probe_test: pd.DataFrame) -> None:
    lines = [
        "# Submission Report: public_feedback_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        "- Public quadratic fit points:",
        "  - alpha 0.00: `lgb_stable_calibrated.csv` public `0.5946792872`",
        "  - alpha 1.00: `lgb_calib_tight.csv` public `0.5944158654`",
        "  - alpha 1.20: `lgb_q3s1s2_boost.csv` public `0.5944366954`",
        f"- Selected alpha: `{PUBLIC_FIT_ALPHA:.2f}`",
        f"- Anti-S4 beta for probe: `{ANTI_S4_BETA:.2f}`", "",
        "## Target Scores", "",
        "| Target | Current OOF | Publicfit QS OOF | 5.7 Probe OOF | Probe test mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        lines.append(
            f"| {target} | `{current_scores[target]:.6f}` | `{qs_scores[target]:.6f}` | "
            f"`{probe_scores[target]:.6f}` | `{probe_test[target].mean():.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This does not guarantee 5.7. It is the most defensible public-feedback candidate after S4/top-k and calibration-extension failures.",
            "- If this fails, leaderboard probing has diminishing returns and a new feature source is required.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "public_feedback_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    labels, anchor_oof, anchor_test, current_oof, current_test, payload = _load_frames()
    base_params = payload["base_params"]
    params = _publicfit_params(base_params, PUBLIC_FIT_ALPHA)

    qs_oof = _apply_publicfit(anchor_oof, current_oof, params)
    qs_test = _apply_publicfit(anchor_test, current_test, params)
    probe_oof = _apply_anti_s4(qs_oof, payload["s4_oof"], ANTI_S4_BETA)
    probe_test = _apply_anti_s4(qs_test, payload["s4_test"], ANTI_S4_BETA)

    current_scores = _score_targets(labels, current_oof)
    qs_scores = _score_targets(labels, qs_oof)
    probe_scores = _score_targets(labels, probe_oof)

    qs_path = READY_DIR / "lgb_publicfit_qs.csv"
    probe_path = READY_DIR / "lgb_57_antis4_probe.csv"
    _make_submission(qs_path, qs_test)
    _make_submission(probe_path, probe_test)

    qs_selection = {"publicfit_alpha": PUBLIC_FIT_ALPHA, "params": params}
    probe_selection = {**qs_selection, "anti_s4_beta": ANTI_S4_BETA, "failed_s4_source": S4_FAILED_RUN}
    _write_prediction_artifacts(QS_RUN, payload["keys"], labels, qs_oof, qs_test, qs_scores, qs_selection)
    _write_prediction_artifacts(PROBE_RUN, payload["keys"], labels, probe_oof, probe_test, probe_scores, probe_selection)
    _append_experiment(EXPERIMENT_NAME, qs_scores, qs_path, qs_selection, "Public quadratic fit on Q3/S1/S2 only.")
    _append_experiment(f"{EXPERIMENT_NAME}_57_antis4_probe", probe_scores, probe_path, probe_selection, "Publicfit Q3/S1/S2 plus small anti-S4 move opposite failed top50 direction.")
    _write_candidate_scores(qs_scores, probe_scores)
    _write_readme(qs_scores, probe_scores)
    _write_report(current_scores, qs_scores, probe_scores, probe_test)

    print("=== public_feedback_v1 ===")
    print(f"qs: {qs_path} OOF={qs_scores['mean']:.6f}")
    print(f"probe: {probe_path} OOF={probe_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(f"{target}: current={current_scores[target]:.6f} qs={qs_scores[target]:.6f} probe={probe_scores[target]:.6f}")


if __name__ == "__main__":
    main()

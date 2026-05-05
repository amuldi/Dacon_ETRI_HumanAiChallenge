#!/usr/bin/env python3
"""Create S4-only temporal recovery candidates after push failed."""

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


EXPERIMENT_NAME = "temporal_s4_recovery_v1"
CURRENT_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
S4_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"
HALF_RUN = "public_lgb_targetwise_temporal_s4half_v1"

CURRENT_PUBLIC_BEST_FILE = "lgb_temporal_targetwise.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5863944910
S4_HALF_FILE = "lgb_temporal_s4half.csv"

S4_BETA = 0.50
CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"


def _now_kst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


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


def _load_run(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet")
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(MODELS_DIR / f"test_predictions_{run_name}.csv")[TARGET_COLUMNS].astype(float)
    return raw[KEY_COLUMNS].copy(), labels, oof, test


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _s4_blend(current: pd.DataFrame, s4_frame: pd.DataFrame, beta: float) -> pd.DataFrame:
    output = current.copy()
    output["S4"] = np.clip(current["S4"] + beta * (s4_frame["S4"] - current["S4"]), CLIP_MIN, CLIP_MAX)
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


def _write_prediction_artifacts(
    run_name: str,
    keys: pd.DataFrame,
    labels: pd.DataFrame,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    scores: dict[str, float],
    selection: dict[str, Any],
) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "temporal_s4_recovery"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "selection": selection})


def _append_experiment(name: str, scores: dict[str, float], submission: Path, selection: dict[str, Any]) -> None:
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
            "validation_scheme": "temporal_s4_recovery",
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
            "notes": "S4-only half step from current best toward temporal S4 probe after over-push failed.",
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(current_scores: dict[str, float], half_scores: dict[str, float], full_scores: dict[str, float]) -> None:
    rows = [
        {"rank": 0, "candidate": CURRENT_PUBLIC_BEST_FILE, "oof_mean": current_scores["mean"], "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}", "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}", "notes": "Current public best."},
        {"rank": 1, "candidate": S4_HALF_FILE, "oof_mean": half_scores["mean"], "public_score": "", "submission_file": f"submissions/ready/{S4_HALF_FILE}", "notes": "Next submit; S4-only half temporal step, no Q/S over-push."},
        {"rank": 2, "candidate": "lgb_temporal_targetwise_s4.csv", "oof_mean": full_scores["mean"], "public_score": "", "submission_file": "submissions/ready/lgb_temporal_targetwise_s4.csv", "notes": "Full S4 temporal probe; higher risk."},
        {"rank": 3, "candidate": "lgb_temporal_push.csv", "oof_mean": 0.5595372562363035, "public_score": "0.5880629390", "submission_file": "submissions/archive/2026-05-02_push_failed/lgb_temporal_push.csv", "notes": "Failed public; archived, do not resubmit."},
        {"rank": 4, "candidate": "lgb_temporal_prior.csv", "oof_mean": 0.5628776447116296, "public_score": "0.5886545849", "submission_file": "submissions/ready/lgb_temporal_prior.csv", "notes": "Previous public best."},
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(current_scores: dict[str, float], half_scores: dict[str, float], full_scores: dict[str, float]) -> None:
    lines = [
        "# Submissions", "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.", "",
        "## Current Best", "",
        f"- Public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`", "",
        "## Recommended Order", "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_PUBLIC_BEST_FILE}` | `{current_scores['mean']:.6f}` | `{CURRENT_PUBLIC_BEST_SCORE:.10f}` | 현재 best |",
        f"| 1 | `ready/{S4_HALF_FILE}` | `{half_scores['mean']:.6f}` |  | 다음 제출, S4 half |",
        f"| 2 | `ready/lgb_temporal_targetwise_s4.csv` | `{full_scores['mean']:.6f}` |  | 고위험 S4 full |",
        "| 3 | `archive/2026-05-02_push_failed/lgb_temporal_push.csv` | `0.559537` | `0.5880629390` | 실패, 제출 금지 |", "",
        "## Notes", "",
        "- `lgb_temporal_push`는 과하게 밀어 public이 악화되어 archive로 이동했습니다.",
        "- 다음 후보는 성공한 targetwise를 유지하고 S4만 작게 움직입니다.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(current_scores: dict[str, float], half_scores: dict[str, float], full_scores: dict[str, float], half_test: pd.DataFrame, full_test: pd.DataFrame) -> None:
    lines = [
        "# Submission Report: temporal_s4_recovery_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        "- Failed file: `lgb_temporal_push.csv` public `0.5880629390`",
        f"- Next file: `submissions/ready/{S4_HALF_FILE}`", "",
        "| Target | Current | S4 half | S4 full | Half test mean | Full test mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        lines.append(
            f"| {target} | `{current_scores[target]:.6f}` | `{half_scores[target]:.6f}` | "
            f"`{full_scores[target]:.6f}` | `{half_test[target].mean():.4f}` | `{full_test[target].mean():.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Stop Q/S over-push. It failed public.",
            "- Try the S4-only half step next; it changes only S4 and keeps the proven Q/S targetwise predictions.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "temporal_s4_recovery_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    keys, labels, current_oof, current_test = _load_run(CURRENT_RUN)
    _, _, full_oof, full_test = _load_run(S4_RUN)
    half_oof = _s4_blend(current_oof, full_oof, S4_BETA)
    half_test = _s4_blend(current_test, full_test, S4_BETA)

    current_scores = _score_targets(labels, current_oof)
    half_scores = _score_targets(labels, half_oof)
    full_scores = _score_targets(labels, full_oof)

    half_path = READY_DIR / S4_HALF_FILE
    _make_submission(half_path, half_test)

    selection = {"current_run": CURRENT_RUN, "s4_run": S4_RUN, "s4_beta": S4_BETA}
    _write_prediction_artifacts(HALF_RUN, keys, labels, half_oof, half_test, half_scores, selection)
    _append_experiment(EXPERIMENT_NAME, half_scores, half_path, selection)
    _write_candidate_scores(current_scores, half_scores, full_scores)
    _write_readme(current_scores, half_scores, full_scores)
    _write_report(current_scores, half_scores, full_scores, half_test, full_test)

    print("=== temporal_s4_recovery_v1 ===")
    print(f"current: {CURRENT_PUBLIC_BEST_FILE} OOF={current_scores['mean']:.6f} public={CURRENT_PUBLIC_BEST_SCORE:.10f}")
    print(f"s4_half: {half_path} OOF={half_scores['mean']:.6f}")
    print(f"s4_full: submissions/ready/lgb_temporal_targetwise_s4.csv OOF={full_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(f"{target}: current={current_scores[target]:.6f} half={half_scores[target]:.6f} full={full_scores[target]:.6f}")


if __name__ == "__main__":
    main()

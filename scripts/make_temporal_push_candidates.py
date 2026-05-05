#!/usr/bin/env python3
"""Create public-success-direction temporal push candidates."""

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


EXPERIMENT_NAME = "temporal_push_v1"
CURRENT_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
PRIOR_RUN = "public_lgb_targetwise_temporal_prior_v1"
S4_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"
PUSH_RUN = "public_lgb_targetwise_temporal_push_v1"
PUSH_S4_RUN = "public_lgb_targetwise_temporal_push_s4_v1"

CURRENT_PUBLIC_BEST_FILE = "lgb_temporal_targetwise.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5863944910

PUSH_FILE = "lgb_temporal_push.csv"
PUSH_S4_FILE = "lgb_temporal_push_s4.csv"

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

# OOF optimum along prior -> targetwise direction, target by target.
PUSH_ALPHA = {
    "Q1": 2.00,
    "Q2": 2.00,
    "Q3": 1.72,
    "S1": 1.39,
    "S2": 1.09,
    "S3": 1.04,
    "S4": 0.00,
}
S4_BETA = 0.75


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


def _push(prior: pd.DataFrame, targetwise: pd.DataFrame, alpha_by_target: dict[str, float]) -> pd.DataFrame:
    output = pd.DataFrame(index=prior.index)
    for target in TARGET_COLUMNS:
        alpha = float(alpha_by_target[target])
        output[target] = np.clip(prior[target] + alpha * (targetwise[target] - prior[target]), CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _apply_s4(push: pd.DataFrame, targetwise: pd.DataFrame, s4_probe: pd.DataFrame, beta: float) -> pd.DataFrame:
    output = push.copy()
    output["S4"] = np.clip(targetwise["S4"] + beta * (s4_probe["S4"] - targetwise["S4"]), CLIP_MIN, CLIP_MAX)
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
    oof_export["split_scheme"] = "temporal_public_success_push"
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
            "validation_scheme": "temporal_public_success_push",
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


def _write_candidate_scores(current_scores: dict[str, float], push_scores: dict[str, float], push_s4_scores: dict[str, float]) -> None:
    rows = [
        {"rank": 0, "candidate": CURRENT_PUBLIC_BEST_FILE, "oof_mean": current_scores["mean"], "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}", "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}", "notes": "Current public best."},
        {"rank": 1, "candidate": PUSH_FILE, "oof_mean": push_scores["mean"], "public_score": "", "submission_file": f"submissions/ready/{PUSH_FILE}", "notes": "Next submit; target-specific push along proven temporal direction, S4 frozen."},
        {"rank": 2, "candidate": PUSH_S4_FILE, "oof_mean": push_s4_scores["mean"], "public_score": "", "submission_file": f"submissions/ready/{PUSH_S4_FILE}", "notes": "Higher-risk push with small S4 temporal move."},
        {"rank": 3, "candidate": "lgb_temporal_targetwise_s4.csv", "oof_mean": 0.559687095376793, "public_score": "", "submission_file": "submissions/ready/lgb_temporal_targetwise_s4.csv", "notes": "Existing S4 probe; submit only after S4-frozen push decision."},
        {"rank": 4, "candidate": "lgb_temporal_prior.csv", "oof_mean": 0.5628776447116296, "public_score": "0.5886545849", "submission_file": "submissions/ready/lgb_temporal_prior.csv", "notes": "Previous public best."},
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(current_scores: dict[str, float], push_scores: dict[str, float], push_s4_scores: dict[str, float]) -> None:
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
        f"| 1 | `ready/{PUSH_FILE}` | `{push_scores['mean']:.6f}` |  | 다음 제출, S4 동결 |",
        f"| 2 | `ready/{PUSH_S4_FILE}` | `{push_s4_scores['mean']:.6f}` |  | 고위험 S4 push |",
        "| 3 | `ready/lgb_temporal_targetwise_s4.csv` | `0.559687` |  | 기존 S4 probe |",
        "| 4 | `ready/lgb_temporal_prior.csv` | `0.562878` | `0.5886545849` | 이전 best |", "",
        "## Notes", "",
        "- `lgb_temporal_push`: public에서 성공한 `temporal_prior -> temporal_targetwise` 방향을 target별로 조금 더 민 후보입니다.",
        "- S4는 public 실패 이력이 있어 1순위에서는 동결합니다.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(
    prior_scores: dict[str, float],
    current_scores: dict[str, float],
    push_scores: dict[str, float],
    push_s4_scores: dict[str, float],
    push_test: pd.DataFrame,
    push_s4_test: pd.DataFrame,
) -> None:
    lines = [
        "# Submission Report: temporal_push_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Current best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`",
        f"- Push file: `submissions/ready/{PUSH_FILE}`",
        f"- Push S4 file: `submissions/ready/{PUSH_S4_FILE}`", "",
        "## OOF Target Scores", "",
        "| Target | Prior v1 | Current best | Push | Push+S4 | Alpha | Push mean | Push+S4 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        lines.append(
            f"| {target} | `{prior_scores[target]:.6f}` | `{current_scores[target]:.6f}` | "
            f"`{push_scores[target]:.6f}` | `{push_s4_scores[target]:.6f}` | "
            f"`{PUSH_ALPHA[target]:.2f}` | `{push_test[target].mean():.4f}` | `{push_s4_test[target].mean():.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Push OOF mean: `{push_scores['mean']:.6f}` vs current best `{current_scores['mean']:.6f}`.",
            "- This is a narrow continuation of the public-success direction, not a new model family.",
            "- Submit `lgb_temporal_push.csv` first; S4 version only after confirming this direction still improves public.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "temporal_push_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    keys, labels, current_oof, current_test = _load_run(CURRENT_RUN)
    _, _, prior_oof, prior_test = _load_run(PRIOR_RUN)
    _, _, s4_oof, s4_test = _load_run(S4_RUN)

    push_oof = _push(prior_oof, current_oof, PUSH_ALPHA)
    push_test = _push(prior_test, current_test, PUSH_ALPHA)
    push_s4_oof = _apply_s4(push_oof, current_oof, s4_oof, S4_BETA)
    push_s4_test = _apply_s4(push_test, current_test, s4_test, S4_BETA)

    prior_scores = _score_targets(labels, prior_oof)
    current_scores = _score_targets(labels, current_oof)
    push_scores = _score_targets(labels, push_oof)
    push_s4_scores = _score_targets(labels, push_s4_oof)

    push_path = READY_DIR / PUSH_FILE
    push_s4_path = READY_DIR / PUSH_S4_FILE
    _make_submission(push_path, push_test)
    _make_submission(push_s4_path, push_s4_test)

    push_selection = {"prior_run": PRIOR_RUN, "current_run": CURRENT_RUN, "alpha_by_target": PUSH_ALPHA, "s4_frozen": True}
    push_s4_selection = {**push_selection, "s4_source_run": S4_RUN, "s4_beta": S4_BETA, "s4_frozen": False}
    _write_prediction_artifacts(PUSH_RUN, keys, labels, push_oof, push_test, push_scores, push_selection)
    _write_prediction_artifacts(PUSH_S4_RUN, keys, labels, push_s4_oof, push_s4_test, push_s4_scores, push_s4_selection)
    _append_experiment(EXPERIMENT_NAME, push_scores, push_path, push_selection, "Target-specific push along public-success temporal direction with S4 frozen.")
    _append_experiment(f"{EXPERIMENT_NAME}_s4", push_s4_scores, push_s4_path, push_s4_selection, "Target-specific push plus small S4 temporal movement.")
    _write_candidate_scores(current_scores, push_scores, push_s4_scores)
    _write_readme(current_scores, push_scores, push_s4_scores)
    _write_report(prior_scores, current_scores, push_scores, push_s4_scores, push_test, push_s4_test)

    print("=== temporal_push_v1 ===")
    print(f"current: {CURRENT_PUBLIC_BEST_FILE} OOF={current_scores['mean']:.6f} public={CURRENT_PUBLIC_BEST_SCORE:.10f}")
    print(f"push: {push_path} OOF={push_scores['mean']:.6f}")
    print(f"push_s4: {push_s4_path} OOF={push_s4_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(
            f"{target}: prior={prior_scores[target]:.6f} current={current_scores[target]:.6f} "
            f"push={push_scores[target]:.6f} push_s4={push_s4_scores[target]:.6f}"
        )


if __name__ == "__main__":
    main()

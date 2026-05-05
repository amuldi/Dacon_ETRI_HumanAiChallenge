#!/usr/bin/env python3
"""Create bidirectional temporal-prior blend candidates.

The public rows are interleaved with train rows for the same subjects.  This
post-process uses nearby known train labels on both sides of each date as a
small transductive prior, then blends it into the current public best.
"""

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


EXPERIMENT_NAME = "temporal_prior_v1"
BASE_RUN = "public_lgb_targetwise_57_antis4_probe_v1"
PRIMARY_RUN = "public_lgb_targetwise_temporal_prior_v1"
SAFE_RUN = "public_lgb_targetwise_temporal_prior_v1_safe"

CURRENT_PUBLIC_BEST_FILE = "lgb_57_antis4_probe.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5943866101

PRIMARY_FILE = "lgb_temporal_prior.csv"
SAFE_FILE = "lgb_temporal_prior_safe.csv"

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

# OOF grid best: local leave-one-out temporal prior, small target-wise weights.
PRIOR_CONFIG = {
    "kind": "power",
    "max_days": 7,
    "power": 2.0,
    "subject_weight": 0.05,
    "global_weight": 0.05,
}
PRIMARY_WEIGHTS = {
    "Q1": 0.27,
    "Q2": 0.25,
    "Q3": 0.12,
    "S1": 0.02,
    "S2": 0.07,
    "S3": 0.14,
    "S4": 0.00,
}
SAFE_SCALE = 0.60


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


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{BASE_RUN}.parquet")
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(MODELS_DIR / f"test_predictions_{BASE_RUN}.csv")[TARGET_COLUMNS].astype(float)
    return raw[KEY_COLUMNS].copy(), labels, oof, test


def _weighted_prior(values: np.ndarray, distances: np.ndarray, config: dict[str, Any]) -> float | None:
    mask = (distances > 0) & (distances <= float(config["max_days"]))
    if not np.any(mask):
        return None
    local_dist = distances[mask].astype(float)
    if config["kind"] != "power":
        raise ValueError(f"Unsupported prior kind: {config['kind']}")
    weights = 1.0 / np.power(local_dist + 1.0, float(config["power"]))
    total = float(weights.sum())
    if total <= 1e-12:
        return None
    return float(np.sum(weights * values[mask]) / total)


def _make_train_prior(keys: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    source = pd.concat([keys[KEY_COLUMNS].copy(), labels[TARGET_COLUMNS].copy()], axis=1)
    global_mean = labels[TARGET_COLUMNS].mean()
    subject_mean = source.groupby("subject_id")[TARGET_COLUMNS].mean()
    prior = pd.DataFrame(index=source.index, columns=TARGET_COLUMNS, dtype=float)

    for subject_id, group in source.groupby("subject_id", sort=False):
        idx = group.index.to_numpy()
        days = pd.to_datetime(group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        distances = np.abs(days[:, None] - days[None, :]).astype(float)
        for row_pos, row_idx in enumerate(idx):
            for target in TARGET_COLUMNS:
                local = _weighted_prior(group[target].to_numpy(float), distances[row_pos], config)
                if local is None:
                    local = float(subject_mean.loc[subject_id, target])
                subject_part = float(subject_mean.loc[subject_id, target])
                global_part = float(global_mean[target])
                main_weight = 1.0 - float(config["subject_weight"]) - float(config["global_weight"])
                prior.loc[row_idx, target] = main_weight * local + float(config["subject_weight"]) * subject_part + float(config["global_weight"]) * global_part

    return prior[TARGET_COLUMNS].clip(CLIP_MIN, CLIP_MAX)


def _make_test_prior(train_labels: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    sample = load_submission_template()[KEY_COLUMNS].copy()
    source = train_labels[KEY_COLUMNS + TARGET_COLUMNS].copy()
    global_mean = source[TARGET_COLUMNS].mean()
    subject_mean = source.groupby("subject_id")[TARGET_COLUMNS].mean()
    prior = pd.DataFrame(index=sample.index, columns=TARGET_COLUMNS, dtype=float)

    for subject_id, test_group in sample.groupby("subject_id", sort=False):
        train_group = source[source["subject_id"] == subject_id].sort_values("lifelog_date")
        train_days = pd.to_datetime(train_group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        test_days = pd.to_datetime(test_group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        distances = np.abs(test_days[:, None] - train_days[None, :]).astype(float)
        for row_pos, row_idx in enumerate(test_group.index.to_numpy()):
            for target in TARGET_COLUMNS:
                local = _weighted_prior(train_group[target].to_numpy(float), distances[row_pos], config)
                if local is None:
                    local = float(subject_mean.loc[subject_id, target])
                subject_part = float(subject_mean.loc[subject_id, target])
                global_part = float(global_mean[target])
                main_weight = 1.0 - float(config["subject_weight"]) - float(config["global_weight"])
                prior.loc[row_idx, target] = main_weight * local + float(config["subject_weight"]) * subject_part + float(config["global_weight"]) * global_part

    return prior[TARGET_COLUMNS].clip(CLIP_MIN, CLIP_MAX)


def _blend(base: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    output = pd.DataFrame(index=base.index)
    for target in TARGET_COLUMNS:
        weight = float(weights[target])
        output[target] = np.clip((1.0 - weight) * base[target] + weight * prior[target], CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _scaled_weights(scale: float) -> dict[str, float]:
    return {target: float(PRIMARY_WEIGHTS[target] * scale) for target in TARGET_COLUMNS}


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
    weights: dict[str, float],
) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "loo_bidirectional_temporal_prior"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(
        FEATURES_DIR / f"{run_name}_summary.json",
        {
            "run_name": run_name,
            "base_run": BASE_RUN,
            "scores": scores,
            "prior_config": PRIOR_CONFIG,
            "blend_weights": weights,
        },
    )


def _append_experiment(name: str, scores: dict[str, float], submission: Path, weights: dict[str, float], notes: str) -> None:
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
            "validation_scheme": "loo_bidirectional_temporal_prior",
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
            "feature_view_by_target": json.dumps({"base_run": BASE_RUN, "prior_config": PRIOR_CONFIG, "blend_weights": weights}, sort_keys=True),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(base_scores: dict[str, float], primary_scores: dict[str, float], safe_scores: dict[str, float]) -> None:
    rows = [
        {"rank": 0, "candidate": PRIMARY_FILE, "oof_mean": primary_scores["mean"], "public_score": "", "submission_file": f"submissions/ready/{PRIMARY_FILE}", "notes": "Next submit; bidirectional same-subject temporal label prior blended into current best."},
        {"rank": 1, "candidate": CURRENT_PUBLIC_BEST_FILE, "oof_mean": base_scores["mean"], "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}", "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}", "notes": "Current public best."},
        {"rank": 2, "candidate": SAFE_FILE, "oof_mean": safe_scores["mean"], "public_score": "", "submission_file": f"submissions/ready/{SAFE_FILE}", "notes": f"Backup; same prior with {SAFE_SCALE:.0%} blend weights."},
        {"rank": 3, "candidate": "lgb_calib_tight.csv", "oof_mean": 0.5645484761869698, "public_score": "0.5944158654", "submission_file": "submissions/ready/lgb_calib_tight.csv", "notes": "Previous public best."},
        {"rank": 4, "candidate": "lgb_s4_top50.csv", "oof_mean": 0.5583579944816833, "public_score": "0.5970623278", "submission_file": "submissions/ready/lgb_s4_top50.csv", "notes": "Failed public; do not resubmit."},
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(base_scores: dict[str, float], primary_scores: dict[str, float], safe_scores: dict[str, float]) -> None:
    lines = [
        "# Submissions", "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.", "",
        "## Current Best", "",
        f"- Public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`", "",
        "## Recommended Order", "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 1 | `ready/{PRIMARY_FILE}` | `{primary_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/{CURRENT_PUBLIC_BEST_FILE}` | `{base_scores['mean']:.6f}` | `{CURRENT_PUBLIC_BEST_SCORE:.10f}` | 현재 best |",
        f"| 3 | `ready/{SAFE_FILE}` | `{safe_scores['mean']:.6f}` |  | 1번이 실패하면 백업 |",
        "| 4 | `ready/lgb_calib_tight.csv` | `0.564548` | `0.5944158654` | 이전 best |",
        "| 5 | `ready/lgb_s4_top50.csv` | `0.558358` | `0.5970623278` | 실패, 제출 금지 |", "",
        "## Notes", "",
        "- `lgb_temporal_prior`: same subject의 가까운 train 라벨을 과거/미래 양방향으로 모아 current best에 작게 섞은 후보입니다.",
        "- OOF 검증은 leave-one-out temporal prior라서 strict fold CV보다 낙관적일 수 있습니다. 다만 test 예측 시에는 모든 train 라벨이 실제로 사용 가능합니다.",
        "- S4 top50은 public에서 크게 실패했으므로 계속 제출 금지입니다.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(
    base_scores: dict[str, float],
    prior_scores: dict[str, float],
    primary_scores: dict[str, float],
    safe_scores: dict[str, float],
    primary_test: pd.DataFrame,
    safe_test: pd.DataFrame,
) -> None:
    safe_weights = _scaled_weights(SAFE_SCALE)
    lines = [
        "# Submission Report: temporal_prior_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Base run: `{BASE_RUN}`",
        f"- Prior config: `{PRIOR_CONFIG}`",
        f"- Primary file: `submissions/ready/{PRIMARY_FILE}`",
        f"- Safe file: `submissions/ready/{SAFE_FILE}`", "",
        "## OOF Target Scores", "",
        "| Target | Base | Prior only | Primary blend | Safe blend | Primary weight | Safe weight | Primary test mean | Safe test mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        lines.append(
            f"| {target} | `{base_scores[target]:.6f}` | `{prior_scores[target]:.6f}` | "
            f"`{primary_scores[target]:.6f}` | `{safe_scores[target]:.6f}` | "
            f"`{PRIMARY_WEIGHTS[target]:.2f}` | `{safe_weights[target]:.2f}` | "
            f"`{primary_test[target].mean():.4f}` | `{safe_test[target].mean():.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Primary OOF mean: `{primary_scores['mean']:.6f}` vs base `{base_scores['mean']:.6f}`.",
            "- This is the first non-calibration candidate with a meaningful OOF move after the latest public feedback.",
            "- It can still fail public if leaderboard split is not temporally random within subject; submit primary first, then safe only if needed.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "temporal_prior_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()

    keys, labels, base_oof, base_test = _load_base()
    train_prior = _make_train_prior(keys, labels, PRIOR_CONFIG)
    test_prior = _make_test_prior(load_train_labels(), PRIOR_CONFIG)

    safe_weights = _scaled_weights(SAFE_SCALE)
    primary_oof = _blend(base_oof, train_prior, PRIMARY_WEIGHTS)
    safe_oof = _blend(base_oof, train_prior, safe_weights)
    primary_test = _blend(base_test, test_prior, PRIMARY_WEIGHTS)
    safe_test = _blend(base_test, test_prior, safe_weights)

    base_scores = _score_targets(labels, base_oof)
    prior_scores = _score_targets(labels, train_prior)
    primary_scores = _score_targets(labels, primary_oof)
    safe_scores = _score_targets(labels, safe_oof)

    primary_path = READY_DIR / PRIMARY_FILE
    safe_path = READY_DIR / SAFE_FILE
    _make_submission(primary_path, primary_test)
    _make_submission(safe_path, safe_test)

    _write_prediction_artifacts(PRIMARY_RUN, keys, labels, primary_oof, primary_test, primary_scores, PRIMARY_WEIGHTS)
    _write_prediction_artifacts(SAFE_RUN, keys, labels, safe_oof, safe_test, safe_scores, safe_weights)
    _append_experiment(EXPERIMENT_NAME, primary_scores, primary_path, PRIMARY_WEIGHTS, "Bidirectional same-subject temporal label prior blended into current public best.")
    _append_experiment(f"{EXPERIMENT_NAME}_safe", safe_scores, safe_path, safe_weights, f"Safe {SAFE_SCALE:.0%} blend of bidirectional temporal label prior.")
    _write_candidate_scores(base_scores, primary_scores, safe_scores)
    _write_readme(base_scores, primary_scores, safe_scores)
    _write_report(base_scores, prior_scores, primary_scores, safe_scores, primary_test, safe_test)

    print("=== temporal_prior_v1 ===")
    print(f"base:    {CURRENT_PUBLIC_BEST_FILE} OOF={base_scores['mean']:.6f} public={CURRENT_PUBLIC_BEST_SCORE:.10f}")
    print(f"primary: {primary_path} OOF={primary_scores['mean']:.6f}")
    print(f"safe:    {safe_path} OOF={safe_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(
            f"{target}: base={base_scores[target]:.6f} prior={prior_scores[target]:.6f} "
            f"primary={primary_scores[target]:.6f} safe={safe_scores[target]:.6f}"
        )


if __name__ == "__main__":
    main()

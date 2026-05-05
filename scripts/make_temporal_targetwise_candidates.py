#!/usr/bin/env python3
"""Create target-wise temporal-prior candidates after public temporal success."""

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


EXPERIMENT_NAME = "temporal_targetwise_v1"
BASE_RUN = "public_lgb_targetwise_57_antis4_probe_v1"
CURRENT_BEST_RUN = "public_lgb_targetwise_temporal_prior_v1"
PRIMARY_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
S4_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"

CURRENT_PUBLIC_BEST_FILE = "lgb_temporal_prior.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5886545849
CURRENT_PUBLIC_BEST_OOF = 0.5628776447116296

PRIMARY_FILE = "lgb_temporal_targetwise.csv"
S4_FILE = "lgb_temporal_targetwise_s4.csv"

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

DEFAULT_PRIOR = {"kind": "power", "max_days": 7, "power": 2.0, "subject_weight": 0.05, "global_weight": 0.05}
TARGET_PRIOR_CONFIGS: dict[str, dict[str, Any]] = {
    "Q1": {"kind": "power", "max_days": 7, "power": 3.0, "subject_weight": 0.05, "global_weight": 0.05},
    "Q2": {"kind": "power", "max_days": 5, "power": 2.0, "subject_weight": 0.05, "global_weight": 0.05},
    "Q3": {"kind": "power", "max_days": 3, "power": 3.0, "subject_weight": 0.05, "global_weight": 0.05},
    "S1": {"kind": "power", "max_days": 3, "power": 3.0, "subject_weight": 0.05, "global_weight": 0.05},
    "S2": {"kind": "exp", "max_days": 21, "tau": 45.0, "subject_weight": 0.05, "global_weight": 0.05},
    "S3": {"kind": "exp", "max_days": 90, "tau": 45.0, "subject_weight": 0.05, "global_weight": 0.05},
    "S4": {"kind": "exp", "max_days": 21, "tau": 45.0, "subject_weight": 0.05, "global_weight": 0.05},
}
TARGET_WEIGHTS_WITH_S4 = {
    "Q1": 0.27,
    "Q2": 0.26,
    "Q3": 0.18,
    "S1": 0.07,
    "S2": 0.43,
    "S3": 0.57,
    "S4": 0.19,
}
TARGET_WEIGHTS_PRIMARY = {**TARGET_WEIGHTS_WITH_S4, "S4": 0.0}


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


def _load_run(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet")
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(MODELS_DIR / f"test_predictions_{run_name}.csv")[TARGET_COLUMNS].astype(float)
    return raw[KEY_COLUMNS].copy(), labels, oof, test


def _weights_for_distances(distances: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if config["kind"] == "power":
        return 1.0 / np.power(distances + 1.0, float(config["power"]))
    if config["kind"] == "exp":
        return np.exp(-distances / float(config["tau"]))
    raise ValueError(f"Unsupported prior kind: {config['kind']}")


def _weighted_prior(values: np.ndarray, distances: np.ndarray, config: dict[str, Any], *, exclude_self: bool) -> float | None:
    mask = distances <= float(config["max_days"])
    if exclude_self:
        mask &= distances > 0
    if not np.any(mask):
        return None
    local_dist = distances[mask].astype(float)
    weights = _weights_for_distances(local_dist, config)
    total = float(weights.sum())
    if total <= 1e-12:
        return None
    return float(np.sum(weights * values[mask]) / total)


def _target_prior_train(keys: pd.DataFrame, labels: pd.DataFrame, target: str, config: dict[str, Any]) -> np.ndarray:
    source = pd.concat([keys[KEY_COLUMNS].copy(), labels[TARGET_COLUMNS].copy()], axis=1)
    global_mean = float(labels[target].mean())
    subject_mean = source.groupby("subject_id")[target].mean()
    output = np.empty(len(source), dtype=float)

    for subject_id, group in source.groupby("subject_id", sort=False):
        idx = group.index.to_numpy()
        values = group[target].to_numpy(float)
        days = pd.to_datetime(group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        distances = np.abs(days[:, None] - days[None, :]).astype(float)
        for row_pos, row_idx in enumerate(idx):
            local = _weighted_prior(values, distances[row_pos], config, exclude_self=True)
            if local is None:
                local = float(subject_mean.loc[subject_id])
            subject_part = float(subject_mean.loc[subject_id])
            main_weight = 1.0 - float(config["subject_weight"]) - float(config["global_weight"])
            output[row_idx] = main_weight * local + float(config["subject_weight"]) * subject_part + float(config["global_weight"]) * global_mean
    return np.clip(output, CLIP_MIN, CLIP_MAX)


def _target_prior_test(train_labels: pd.DataFrame, target: str, config: dict[str, Any]) -> np.ndarray:
    sample = load_submission_template()[KEY_COLUMNS].copy()
    source = train_labels[KEY_COLUMNS + TARGET_COLUMNS].copy()
    global_mean = float(source[target].mean())
    subject_mean = source.groupby("subject_id")[target].mean()
    output = np.empty(len(sample), dtype=float)

    for subject_id, test_group in sample.groupby("subject_id", sort=False):
        train_group = source[source["subject_id"] == subject_id].sort_values("lifelog_date")
        train_values = train_group[target].to_numpy(float)
        train_days = pd.to_datetime(train_group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        test_days = pd.to_datetime(test_group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        distances = np.abs(test_days[:, None] - train_days[None, :]).astype(float)
        for row_pos, row_idx in enumerate(test_group.index.to_numpy()):
            local = _weighted_prior(train_values, distances[row_pos], config, exclude_self=False)
            if local is None:
                local = float(subject_mean.loc[subject_id])
            subject_part = float(subject_mean.loc[subject_id])
            main_weight = 1.0 - float(config["subject_weight"]) - float(config["global_weight"])
            output[row_idx] = main_weight * local + float(config["subject_weight"]) * subject_part + float(config["global_weight"]) * global_mean
    return np.clip(output, CLIP_MIN, CLIP_MAX)


def _make_targetwise_prior(keys: pd.DataFrame, labels: pd.DataFrame, train_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    train_prior = pd.DataFrame(index=keys.index)
    test_prior = pd.DataFrame(index=load_submission_template().index)
    prior_scores: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        config = TARGET_PRIOR_CONFIGS.get(target, DEFAULT_PRIOR)
        train_prior[target] = _target_prior_train(keys, labels, target, config)
        test_prior[target] = _target_prior_test(train_labels, target, config)
        prior_scores[target] = {
            "config": config,
            "prior_oof": binary_log_loss(labels[target].to_numpy(float), train_prior[target].to_numpy(float)),
        }
    return train_prior[TARGET_COLUMNS], test_prior[TARGET_COLUMNS], prior_scores


def _blend(base: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    output = pd.DataFrame(index=base.index)
    for target in TARGET_COLUMNS:
        weight = float(weights[target])
        output[target] = np.clip((1.0 - weight) * base[target] + weight * prior[target], CLIP_MIN, CLIP_MAX)
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
    weights: dict[str, float],
    prior_scores: dict[str, dict[str, Any]],
) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "loo_targetwise_bidirectional_temporal_prior"
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
            "prior_scores": prior_scores,
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
            "validation_scheme": "loo_targetwise_bidirectional_temporal_prior",
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
            "feature_view_by_target": json.dumps({"base_run": BASE_RUN, "target_prior_configs": TARGET_PRIOR_CONFIGS, "blend_weights": weights}, sort_keys=True),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(current_scores: dict[str, float], primary_scores: dict[str, float], s4_scores: dict[str, float]) -> None:
    rows = [
        {"rank": 0, "candidate": CURRENT_PUBLIC_BEST_FILE, "oof_mean": current_scores["mean"], "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}", "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}", "notes": "Current public best; keep as anchor."},
        {"rank": 1, "candidate": PRIMARY_FILE, "oof_mean": primary_scores["mean"], "public_score": "", "submission_file": f"submissions/ready/{PRIMARY_FILE}", "notes": "Next submit; target-wise temporal prior with S4 frozen."},
        {"rank": 2, "candidate": S4_FILE, "oof_mean": s4_scores["mean"], "public_score": "", "submission_file": f"submissions/ready/{S4_FILE}", "notes": "Higher-risk S4 temporal-prior probe."},
        {"rank": 3, "candidate": "lgb_temporal_prior_safe.csv", "oof_mean": 0.5632553889012931, "public_score": "", "submission_file": "submissions/ready/lgb_temporal_prior_safe.csv", "notes": "Backup from previous temporal_prior_v1."},
        {"rank": 4, "candidate": "lgb_57_antis4_probe.csv", "oof_mean": 0.5653974623125179, "public_score": "0.5943866101", "submission_file": "submissions/archive/2026-05-01_after_temporal_prior/lgb_57_antis4_probe.csv", "notes": "Previous public best; archived."},
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(current_scores: dict[str, float], primary_scores: dict[str, float], s4_scores: dict[str, float]) -> None:
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
        f"| 1 | `ready/{PRIMARY_FILE}` | `{primary_scores['mean']:.6f}` |  | 다음 제출, S4 동결 |",
        f"| 2 | `ready/{S4_FILE}` | `{s4_scores['mean']:.6f}` |  | 고위험 S4 probe |",
        "| 3 | `ready/lgb_temporal_prior_safe.csv` | `0.563255` |  | 백업 |", "",
        "## Ready Folder", "",
        "```text",
        "submissions/ready/",
        f"  {CURRENT_PUBLIC_BEST_FILE}        # current best, public {CURRENT_PUBLIC_BEST_SCORE:.10f}",
        f"  {PRIMARY_FILE}  # next target-wise temporal prior",
        f"  {S4_FILE}  # S4 probe",
        "  lgb_temporal_prior_safe.csv  # backup",
        "```", "",
        "Old/failed CSV files are in `submissions/archive/`.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(
    current_scores: dict[str, float],
    base_scores: dict[str, float],
    primary_scores: dict[str, float],
    s4_scores: dict[str, float],
    prior_scores: dict[str, dict[str, Any]],
    primary_test: pd.DataFrame,
    s4_test: pd.DataFrame,
) -> None:
    lines = [
        "# Submission Report: temporal_targetwise_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Current best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`",
        f"- Primary file: `submissions/ready/{PRIMARY_FILE}`",
        f"- S4 probe file: `submissions/ready/{S4_FILE}`", "",
        "## OOF Target Scores", "",
        "| Target | Base anti-S4 | Current best | Targetwise | Targetwise+S4 | Prior only | Primary w | S4 w | Primary mean | S4 mean | Config |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target in TARGET_COLUMNS:
        lines.append(
            f"| {target} | `{base_scores[target]:.6f}` | `{current_scores[target]:.6f}` | "
            f"`{primary_scores[target]:.6f}` | `{s4_scores[target]:.6f}` | "
            f"`{prior_scores[target]['prior_oof']:.6f}` | `{TARGET_WEIGHTS_PRIMARY[target]:.2f}` | "
            f"`{TARGET_WEIGHTS_WITH_S4[target]:.2f}` | `{primary_test[target].mean():.4f}` | "
            f"`{s4_test[target].mean():.4f}` | `{prior_scores[target]['config']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Primary OOF mean: `{primary_scores['mean']:.6f}` vs current best `{current_scores['mean']:.6f}`.",
            f"- S4 probe OOF mean: `{s4_scores['mean']:.6f}`, but S4 is historically unstable on public.",
            "- Submit the S4-frozen targetwise file first.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "temporal_targetwise_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    keys, labels, base_oof, base_test = _load_run(BASE_RUN)
    _, _, current_oof, _ = _load_run(CURRENT_BEST_RUN)

    train_prior, test_prior, prior_scores = _make_targetwise_prior(keys, labels, load_train_labels())
    primary_oof = _blend(base_oof, train_prior, TARGET_WEIGHTS_PRIMARY)
    primary_test = _blend(base_test, test_prior, TARGET_WEIGHTS_PRIMARY)
    s4_oof = _blend(base_oof, train_prior, TARGET_WEIGHTS_WITH_S4)
    s4_test = _blend(base_test, test_prior, TARGET_WEIGHTS_WITH_S4)

    base_scores = _score_targets(labels, base_oof)
    current_scores = _score_targets(labels, current_oof)
    primary_scores = _score_targets(labels, primary_oof)
    s4_scores = _score_targets(labels, s4_oof)

    primary_path = READY_DIR / PRIMARY_FILE
    s4_path = READY_DIR / S4_FILE
    _make_submission(primary_path, primary_test)
    _make_submission(s4_path, s4_test)

    _write_prediction_artifacts(PRIMARY_RUN, keys, labels, primary_oof, primary_test, primary_scores, TARGET_WEIGHTS_PRIMARY, prior_scores)
    _write_prediction_artifacts(S4_RUN, keys, labels, s4_oof, s4_test, s4_scores, TARGET_WEIGHTS_WITH_S4, prior_scores)
    _append_experiment(EXPERIMENT_NAME, primary_scores, primary_path, TARGET_WEIGHTS_PRIMARY, "Target-wise temporal prior candidate with S4 frozen.")
    _append_experiment(f"{EXPERIMENT_NAME}_s4_probe", s4_scores, s4_path, TARGET_WEIGHTS_WITH_S4, "Target-wise temporal prior candidate including small S4 movement.")
    _write_candidate_scores(current_scores, primary_scores, s4_scores)
    _write_readme(current_scores, primary_scores, s4_scores)
    _write_report(current_scores, base_scores, primary_scores, s4_scores, prior_scores, primary_test, s4_test)

    print("=== temporal_targetwise_v1 ===")
    print(f"current: {CURRENT_PUBLIC_BEST_FILE} OOF={current_scores['mean']:.6f} public={CURRENT_PUBLIC_BEST_SCORE:.10f}")
    print(f"primary: {primary_path} OOF={primary_scores['mean']:.6f}")
    print(f"s4_probe: {s4_path} OOF={s4_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(
            f"{target}: current={current_scores[target]:.6f} "
            f"primary={primary_scores[target]:.6f} s4={s4_scores[target]:.6f} "
            f"prior={prior_scores[target]['prior_oof']:.6f}"
        )


if __name__ == "__main__":
    main()

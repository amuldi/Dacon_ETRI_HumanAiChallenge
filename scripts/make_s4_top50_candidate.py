#!/usr/bin/env python3
"""Train an S4-only top50 LightGBM candidate and assemble submissions."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template, load_train_labels
from etri_human_challenge.paths import (
    EXPERIMENTS_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    OOF_DIR,
    REPORT_SUBMISSIONS_DIR,
    ROOT,
    ensure_runtime_dirs,
)
from etri_human_challenge.public_lgb import (
    PUBLIC_LGB_DEFAULT_SEEDS,
    PUBLIC_LGB_PARAMS,
    get_public_lgb_feature_columns,
    load_public_lgb_feature_table,
)
from etri_human_challenge.utils import binary_log_loss, write_json, write_markdown


EXPERIMENT_NAME = "s4_top50_v1"
CURRENT_BEST_RUN = "public_lgb_targetwise_calib_tight_v1"
RAW_ANCHOR_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
PRIMARY_RUN = "public_lgb_targetwise_s4_top50_v1"
BLEND_RUN = "public_lgb_targetwise_s4_top50_v1_blend"

CURRENT_PUBLIC_BEST_FILE = "lgb_calib_tight.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5944158654
CURRENT_BEST_OOF_MEAN = 0.5645484761869698

S4_TOP_K = 50
S4_BLEND_WEIGHT = 0.70
S4_CAL_TEMP = 1.16
S4_CAL_BIAS = -0.075

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
EXP_DIR = EXPERIMENTS_DIR / EXPERIMENT_NAME


def _now_kst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _logit(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _calibrate_s4(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(_sigmoid(S4_CAL_TEMP * _logit(values) + S4_CAL_BIAS), CLIP_MIN, CLIP_MAX)


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


def _validate_submission(path: Path, predictions: pd.DataFrame) -> None:
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    if predictions.shape != (250, 7):
        raise ValueError(f"Invalid prediction shape: {predictions.shape}")
    values = predictions[TARGET_COLUMNS].to_numpy(float)
    if values.min() < CLIP_MIN or values.max() > CLIP_MAX:
        raise ValueError(f"Predictions outside [{CLIP_MIN}, {CLIP_MAX}] before submission write.")
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = predictions[target].to_numpy(float)
    if submission.shape != (250, 10):
        raise ValueError(f"Invalid submission shape: {submission.shape}")
    if submission.columns.tolist() != template.columns.tolist():
        raise ValueError("Submission columns do not match sample_submission.")


def _make_submission(path: Path, predictions: pd.DataFrame) -> None:
    _validate_submission(path, predictions)
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = predictions[target].to_numpy(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def _load_current_best() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{CURRENT_BEST_RUN}.parquet")
    labels = raw[TARGET_COLUMNS].copy()
    current_oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    current_test = pd.read_csv(MODELS_DIR / f"test_predictions_{CURRENT_BEST_RUN}.csv")[TARGET_COLUMNS].astype(float)
    return labels, current_oof, current_test, raw[KEY_COLUMNS].copy()


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


def _s4_feature_cols(frame: pd.DataFrame) -> list[str]:
    imp_path = EXPERIMENTS_DIR / "guarded_v2_stable_tuned" / "S4_public_hist411_all_importance.parquet"
    if not imp_path.exists():
        raise FileNotFoundError(f"Missing S4 importance file: {imp_path}")
    importance = pd.read_parquet(imp_path)
    base_cols = set(get_public_lgb_feature_columns(frame, "public_hist411"))
    ranked = [col for col in importance.mean(axis=1).sort_values(ascending=False).index.tolist() if col in base_cols]
    return ranked[:S4_TOP_K]


def _train_s4_top50() -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    frame = load_public_lgb_feature_table(rebuild=False)
    train = frame[frame["split"] == "train"].reset_index(drop=True).copy()
    test = frame[frame["split"] == "test"].reset_index(drop=True).copy()
    cols = _s4_feature_cols(frame)

    X = train[cols].replace([np.inf, -np.inf], np.nan)
    X_test = test[cols].replace([np.inf, -np.inf], np.nan)
    y = train["S4"].astype(int).to_numpy()

    oof_total = np.zeros(len(train), dtype=float)
    test_total = np.zeros(len(test), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    params = {**PUBLIC_LGB_PARAMS, "n_estimators": 2500}

    for seed in PUBLIC_LGB_DEFAULT_SEEDS:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
        seed_oof = np.zeros(len(train), dtype=float)
        seed_test = np.zeros(len(test), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
            model = lgb.LGBMClassifier(**{**params, "random_state": int(seed), "verbose": -1, "n_jobs": -1})
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                eval_set=[(X.iloc[valid_idx], y[valid_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
            )
            valid_pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            seed_oof[valid_idx] = valid_pred
            seed_test += model.predict_proba(X_test)[:, 1] / 5
            fold_rows.append(
                {
                    "target": "S4",
                    "fold": int(fold),
                    "seed": int(seed),
                    "logloss": binary_log_loss(y[valid_idx], valid_pred),
                }
            )
        oof_total += seed_oof
        test_total += seed_test

    return oof_total / len(PUBLIC_LGB_DEFAULT_SEEDS), test_total / len(PUBLIC_LGB_DEFAULT_SEEDS), cols, fold_rows


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
    oof_export["split_scheme"] = "public_stratified_s4_top50"
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
            "validation_scheme": "public_stratified_s4_top50",
            "seeds": json.dumps(PUBLIC_LGB_DEFAULT_SEEDS),
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


def _write_candidate_scores(primary_scores: dict[str, float], blend_scores: dict[str, float]) -> None:
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
            "candidate": "lgb_s4_top50.csv",
            "oof_mean": primary_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/lgb_s4_top50.csv",
            "notes": "Next candidate; replace S4 with top50 LGB + S4-only calibration.",
        },
        {
            "rank": 2,
            "candidate": "lgb_s4_top50_blend.csv",
            "oof_mean": blend_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/lgb_s4_top50_blend.csv",
            "notes": "Backup; 70% S4 top50 and 30% current-best S4.",
        },
        {
            "rank": 3,
            "candidate": "lgb_q3s1s2_boost.csv",
            "oof_mean": 0.5645387051310182,
            "public_score": "0.5944366954",
            "submission_file": "submissions/ready/lgb_q3s1s2_boost.csv",
            "notes": "Worse than current best; do not resubmit.",
        },
        {
            "rank": 4,
            "candidate": "lgb_calib_micro.csv",
            "oof_mean": 0.5644909064044111,
            "public_score": "0.5947643318",
            "submission_file": "submissions/ready/lgb_calib_micro.csv",
            "notes": "Worse than current best; do not resubmit.",
        },
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(primary_scores: dict[str, float], blend_scores: dict[str, float]) -> None:
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
        f"| 1 | `ready/lgb_s4_top50.csv` | `{primary_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/lgb_s4_top50_blend.csv` | `{blend_scores['mean']:.6f}` |  | 백업 |",
        "| 3 | `ready/lgb_q3s1s2_boost.csv` | `0.564539` | `0.5944366954` | 실패, 제출 금지 |",
        "| 4 | `ready/lgb_calib_micro.csv` | `0.564491` | `0.5947643318` | 실패, 제출 금지 |", "",
        "## Notes", "",
        "- `lgb_s4_top50`: S4만 `public_hist411` importance top50 LGB로 교체하고 S4 전용 logit calibration 적용.",
        "- `lgb_s4_top50_blend`: S4만 `70% top50 + 30% current best`로 섞은 보수 버전.",
        "- posthoc calibration 확장 방향은 중단했고, 이번 후보는 새 S4 feature/model 축입니다.",
    ]
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(
    current_scores: dict[str, float],
    primary_scores: dict[str, float],
    blend_scores: dict[str, float],
    s4_raw_score: float,
    s4_raw_mean: float,
    s4_cal_mean: float,
    feature_cols: list[str],
) -> None:
    lines = [
        "# Submission Report: s4_top50_v1", "",
        f"- Current public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Current best OOF: `{CURRENT_BEST_OOF_MEAN:.6f}`",
        f"- Primary OOF: `{primary_scores['mean']:.6f}`",
        f"- Blend OOF: `{blend_scores['mean']:.6f}`",
        f"- S4 current OOF: `{current_scores['S4']:.6f}`",
        f"- S4 top50 raw OOF: `{s4_raw_score:.6f}`",
        f"- S4 top50 calibrated OOF: `{primary_scores['S4']:.6f}`",
        f"- S4 raw test mean: `{s4_raw_mean:.4f}`",
        f"- S4 calibrated test mean: `{s4_cal_mean:.4f}`", "",
        "## Target Scores", "",
        "| Target | Current | Primary | Blend |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        lines.append(f"| {target} | `{current_scores[target]:.6f}` | `{primary_scores[target]:.6f}` | `{blend_scores[target]:.6f}` |")
    lines.extend(["", "## S4 Top Features", ""])
    for col in feature_cols[:20]:
        lines.append(f"- `{col}`")
    write_markdown(REPORT_SUBMISSIONS_DIR / "s4_top50_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    labels, current_oof, current_test, keys = _load_current_best()
    s4_oof_raw, s4_test_raw, feature_cols, fold_rows = _train_s4_top50()
    s4_oof_cal = _calibrate_s4(s4_oof_raw)
    s4_test_cal = _calibrate_s4(s4_test_raw)

    primary_oof = current_oof.copy()
    primary_test = current_test.copy()
    primary_oof["S4"] = s4_oof_cal
    primary_test["S4"] = s4_test_cal

    blend_oof = current_oof.copy()
    blend_test = current_test.copy()
    blend_oof["S4"] = np.clip(S4_BLEND_WEIGHT * s4_oof_cal + (1.0 - S4_BLEND_WEIGHT) * current_oof["S4"], CLIP_MIN, CLIP_MAX)
    blend_test["S4"] = np.clip(S4_BLEND_WEIGHT * s4_test_cal + (1.0 - S4_BLEND_WEIGHT) * current_test["S4"], CLIP_MIN, CLIP_MAX)

    current_scores = _score_targets(labels, current_oof)
    primary_scores = _score_targets(labels, primary_oof)
    blend_scores = _score_targets(labels, blend_oof)
    s4_raw_score = binary_log_loss(labels["S4"].to_numpy(float), s4_oof_raw)

    primary_path = READY_DIR / "lgb_s4_top50.csv"
    blend_path = READY_DIR / "lgb_s4_top50_blend.csv"
    _make_submission(primary_path, primary_test)
    _make_submission(blend_path, blend_test)

    selection = {
        "S4": {
            "source": "public_hist411_importance_top50",
            "n_features": len(feature_cols),
            "params": "PUBLIC_LGB_PARAMS",
            "calibration": {"temp": S4_CAL_TEMP, "bias": S4_CAL_BIAS},
            "fold_rows": fold_rows,
        }
    }
    _write_prediction_artifacts(PRIMARY_RUN, keys, labels, primary_oof, primary_test, primary_scores, selection)
    _write_prediction_artifacts(BLEND_RUN, keys, labels, blend_oof, blend_test, blend_scores, {**selection, "blend_weight": S4_BLEND_WEIGHT})
    _append_experiment(EXPERIMENT_NAME, primary_scores, primary_path, selection, "S4-only top50 LGB replacement with S4 logit calibration.")
    _append_experiment(f"{EXPERIMENT_NAME}_blend", blend_scores, blend_path, {**selection, "blend_weight": S4_BLEND_WEIGHT}, "S4-only 70% top50 + 30% current-best blend.")
    _write_candidate_scores(primary_scores, blend_scores)
    _write_readme(primary_scores, blend_scores)
    _write_report(
        current_scores,
        primary_scores,
        blend_scores,
        s4_raw_score,
        float(np.mean(s4_test_raw)),
        float(np.mean(s4_test_cal)),
        feature_cols,
    )

    print("=== s4_top50_v1 ===")
    print(f"primary: {primary_path} OOF={primary_scores['mean']:.6f}")
    print(f"blend: {blend_path} OOF={blend_scores['mean']:.6f}")
    print(f"S4 raw={s4_raw_score:.6f} cal={primary_scores['S4']:.6f} current={current_scores['S4']:.6f}")
    print(f"S4 test mean raw={np.mean(s4_test_raw):.4f} cal={np.mean(s4_test_cal):.4f}")


if __name__ == "__main__":
    main()

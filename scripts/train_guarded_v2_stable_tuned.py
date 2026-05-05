#!/usr/bin/env python3
"""Train guarded_v2_stable_tuned submission candidates."""

from __future__ import annotations

import argparse
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
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.lgb_target_params import TARGET_LGB_PARAMS
from etri_human_challenge.paths import (
    EXPERIMENTS_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    OOF_DIR,
    REPORT_FEATURES_DIR,
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


EXPERIMENT_NAME = "guarded_v2_stable_tuned"
PRIMARY_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
CONSERVATIVE_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned_conservative"
BASELINE_RUN = "public_lgb_targetwise_histmix_guarded_v1"

CURRENT_BEST_PUBLIC = 0.5960566585
NEXT_GUARDED_PUBLIC = 0.5960906414
CURRENT_BEST_OOF = 0.5721963000561784

CLIP_MIN = 0.02
CLIP_MAX = 0.98
N_FOLDS = 5
CV_SCHEME = "public_stratified"
CV_THRESHOLD = 1.5
PRIMARY_MEAN_TOL = 5e-5
OTHER_TARGET_TOL = 1e-4
PROTECTED_TARGETS = {"Q2", "Q3", "S4"}
CONSERVATIVE_MIN_GAIN = 0.002
DRIFT_MEAN_LIMIT = 0.012
DRIFT_MAX_LIMIT = 0.055

TARGET_VIEW = {
    "Q1": "public_core",
    "Q2": "public_hist411",
    "Q3": "public_hist365",
    "S1": "public_core",
    "S2": "public_core",
    "S3": "public_core",
    "S4": "public_hist411",
}

TOP_K_BY_VIEW: dict[str, list[int | str]] = {
    "public_core": [300, 400, 500, "all"],
    "public_hist365": [250, 300, 365, "all"],
    "public_hist411": [250, 300, 350, 411, "all"],
}

READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
EXP_DIR = EXPERIMENTS_DIR / EXPERIMENT_NAME


def _now_kst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _candidate_label(k: int | str) -> str:
    return "all" if k == "all" else f"top{int(k)}"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_baseline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    oof_path = OOF_DIR / f"oof_predictions_{BASELINE_RUN}.parquet"
    test_path = MODELS_DIR / f"test_predictions_{BASELINE_RUN}.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing baseline OOF: {oof_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing baseline test predictions: {test_path}")
    oof_raw = pd.read_parquet(oof_path)
    labels = oof_raw[TARGET_COLUMNS].copy()
    oof_pred = pd.DataFrame(
        {target: oof_raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS}
    )
    test_pred = pd.read_csv(test_path)[TARGET_COLUMNS].astype(float)
    scores = _score_targets(labels, oof_pred)
    return labels, oof_pred, test_pred, scores


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _score_one(labels: pd.DataFrame, target: str, pred: np.ndarray) -> float:
    return binary_log_loss(labels[target].to_numpy(float), pred)


def _select_stable_features(imp_df: pd.DataFrame, X: pd.DataFrame, top_k: int) -> list[str]:
    clean = X.replace([np.inf, -np.inf], np.nan)
    missing_rate = clean.isna().mean()
    med = clean.median(numeric_only=True)
    variance = clean.fillna(med).var(numeric_only=True)

    mean_gain = imp_df.mean(axis=1).reindex(X.columns).fillna(0.0)
    std_gain = imp_df.std(axis=1).reindex(X.columns).fillna(0.0)
    cv_gain = std_gain / (mean_gain.abs() + 1e-9)
    median_gain = float(mean_gain.median())

    valid = (missing_rate < 0.95) & (variance > 1e-12)
    stable = (cv_gain <= CV_THRESHOLD) | (mean_gain >= median_gain)
    ranked = mean_gain[valid & stable].sort_values(ascending=False)
    if ranked.empty:
        ranked = mean_gain[valid].sort_values(ascending=False)
    if ranked.empty:
        ranked = mean_gain.sort_values(ascending=False)
    return ranked.index.tolist()[: min(int(top_k), len(ranked))]


def _params_for_target(target: str, seed: int) -> dict[str, Any]:
    params = TARGET_LGB_PARAMS.get(target, PUBLIC_LGB_PARAMS)
    return {**params, "random_state": int(seed)}


def _cache_paths(target: str, label: str) -> tuple[Path, Path, Path]:
    stem = f"{target}_{TARGET_VIEW[target]}_{label}"
    return EXP_DIR / f"{stem}.npz", EXP_DIR / f"{stem}.json", EXP_DIR / f"{stem}_importance.parquet"


def _train_target_candidate(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    label: str,
    seeds: list[int],
    retrain: bool,
    collect_importance: bool,
) -> dict[str, Any]:
    npz_path, meta_path, imp_path = _cache_paths(target, label)
    if npz_path.exists() and meta_path.exists() and not retrain:
        meta = json.loads(meta_path.read_text())
        if meta.get("feature_cols") == feature_cols:
            data = np.load(npz_path)
            imp_df = pd.read_parquet(imp_path) if collect_importance and imp_path.exists() else None
            return {
                **meta,
                "oof": data["oof"],
                "test": data["test"],
                "importance": imp_df,
            }

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = train[target].astype(int).to_numpy()

    oof_total = np.zeros(len(train), dtype=float)
    test_total = np.zeros(len(test), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    importance_parts: list[pd.Series] = []

    for seed in seeds:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed))
        seed_oof = np.zeros(len(train), dtype=float)
        seed_test = np.zeros(len(test), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_train, y)):
            model = lgb.LGBMClassifier(**_params_for_target(target, int(seed)))
            model.fit(
                X_train.iloc[train_idx],
                y[train_idx],
                eval_set=[(X_train.iloc[valid_idx], y[valid_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
            )
            valid_pred = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
            seed_oof[valid_idx] = valid_pred
            seed_test += model.predict_proba(X_test)[:, 1] / N_FOLDS
            fold_rows.append(
                {
                    "target": target,
                    "candidate": label,
                    "fold": int(fold),
                    "seed": int(seed),
                    "logloss": binary_log_loss(y[valid_idx], valid_pred),
                }
            )
            if collect_importance:
                importance_parts.append(
                    pd.Series(
                        model.booster_.feature_importance(importance_type="gain"),
                        index=feature_cols,
                        name=f"{seed}_{fold}",
                        dtype=float,
                    )
                )
        oof_total += seed_oof
        test_total += seed_test

    oof = oof_total / len(seeds)
    test_pred = test_total / len(seeds)
    score = _score_one(labels, target, oof)
    imp_df = pd.concat(importance_parts, axis=1).fillna(0.0) if importance_parts else None

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, oof=oof, test=test_pred)
    meta = {
        "target": target,
        "view": TARGET_VIEW[target],
        "candidate": label,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "score": score,
        "fold_rows": fold_rows,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if imp_df is not None:
        imp_df.to_parquet(imp_path)
    return {**meta, "oof": oof, "test": test_pred, "importance": imp_df}


def _drift_stats(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float | bool]:
    diff = np.abs(candidate - baseline)
    mean_abs = float(np.mean(diff))
    max_abs = float(np.max(diff))
    return {
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "ok": bool(mean_abs <= DRIFT_MEAN_LIMIT and max_abs <= DRIFT_MAX_LIMIT),
    }


def _make_submission(path: Path, test_pred: pd.DataFrame, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = np.clip(test_pred[target].to_numpy(float), CLIP_MIN, CLIP_MAX)
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
    *,
    run_name: str,
    train_keys: pd.DataFrame,
    labels: pd.DataFrame,
    oof_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    scores: dict[str, float],
    selection: dict[str, Any],
) -> None:
    oof_export = train_keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = CV_SCHEME
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof_pred[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test_pred[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(
        FEATURES_DIR / f"{run_name}_summary.json",
        {
            "run_name": run_name,
            "experiment_name": EXPERIMENT_NAME,
            "validation_scheme": CV_SCHEME,
            "seeds": PUBLIC_LGB_DEFAULT_SEEDS,
            "scores": scores,
            "selection": selection,
        },
    )


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
    rows = [row for row in rows if row.get("experiment_name") != "guarded_s2s3"]
    rows.append(
        {
            "timestamp": "2026-04-29 00:17:30",
            "submission_file": "next_guarded_s2s3.csv",
            "experiment_name": "guarded_s2s3",
            "public_score": f"{NEXT_GUARDED_PUBLIC:.10f}",
            "delta_vs_best": f"{NEXT_GUARDED_PUBLIC - CURRENT_BEST_PUBLIC:.10f}",
            "notes": "Worse than current best; stop S2/S3 subject-holdout blend direction.",
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_outputs(
    *,
    primary_scores: dict[str, float],
    conservative_scores: dict[str, float],
    primary_path: Path,
    conservative_path: Path,
    primary_notes: str,
    conservative_notes: str,
) -> None:
    rows = [
        {
            "rank": 0,
            "candidate": "00_best_guarded.csv",
            "oof_mean": CURRENT_BEST_OOF,
            "public_score": f"{CURRENT_BEST_PUBLIC:.10f}",
            "submission_file": "submissions/ready/00_best_guarded.csv",
            "notes": "Reference current best; keep as anchor.",
        },
        {
            "rank": 1,
            "candidate": primary_path.name,
            "oof_mean": primary_scores["mean"],
            "public_score": "",
            "submission_file": str(primary_path.relative_to(ROOT)),
            "notes": primary_notes,
        },
        {
            "rank": 2,
            "candidate": conservative_path.name,
            "oof_mean": conservative_scores["mean"],
            "public_score": "",
            "submission_file": str(conservative_path.relative_to(ROOT)),
            "notes": conservative_notes,
        },
        {
            "rank": 3,
            "candidate": "next_guarded_s2s3.csv",
            "oof_mean": 0.5721908206619111,
            "public_score": f"{NEXT_GUARDED_PUBLIC:.10f}",
            "submission_file": "submissions/ready/next_guarded_s2s3.csv",
            "notes": "Submitted and slightly worse; stop S2/S3 holdout blend direction.",
        },
        {
            "rank": 4,
            "candidate": "01_soft_w090.csv",
            "oof_mean": 0.5721474375721535,
            "public_score": "0.5962890684",
            "submission_file": "submissions/ready/01_soft_w090.csv",
            "notes": "Submitted and worse; stop Q2/Q3 softblend direction.",
        },
        {
            "rank": 5,
            "candidate": "04_soft_w085.csv",
            "oof_mean": 0.5721394374047376,
            "public_score": "",
            "submission_file": "submissions/ready/04_soft_w085.csv",
            "notes": "Hold; inherits failed softblend direction.",
        },
        {
            "rank": 6,
            "candidate": "05_soft_w095.csv",
            "oof_mean": 0.5721663642243604,
            "public_score": "",
            "submission_file": "submissions/ready/05_soft_w095.csv",
            "notes": "Hold; inherits failed softblend direction.",
        },
    ]
    _write_csv_rows(
        LOG_DIR / "candidate_scores.csv",
        rows,
        ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"],
    )


def _write_readme(primary_path: Path, conservative_path: Path, primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    lines = [
        "# Submissions",
        "",
        "DACON에 올릴 파일은 기본적으로 `submissions/ready/`만 보면 됩니다.",
        "",
        "## Current Best",
        "",
        f"- Public best: `{CURRENT_BEST_PUBLIC:.10f}`",
        "- Best file: `submissions/ready/00_best_guarded.csv`",
        "",
        "## Recent Public Results",
        "",
        "- `next_guarded_s2s3.csv`: `0.5960906414`, worse than best by `+0.0000339829`.",
        "- `01_soft_w090.csv`: `0.5962890684`, worse than best by `+0.0002324099`.",
        "",
        "## Recommended Order",
        "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/00_best_guarded.csv` | `{CURRENT_BEST_OOF:.6f}` | `{CURRENT_BEST_PUBLIC:.10f}` | 기준 파일 |",
        f"| 1 | `ready/{primary_path.name}` | `{primary_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/{conservative_path.name}` | `{conservative_scores['mean']:.6f}` |  | 1순위 실패 시 제출 |",
        "| 3 | `ready/next_guarded_s2s3.csv` | `0.572191` | `0.5960906414` | 실패, 중단 |",
        "| 4 | `ready/01_soft_w090.csv` | `0.572147` | `0.5962890684` | 실패, 중단 |",
        "",
        "## Notes",
        "",
        "- `stable_tuned`: current best target map은 유지하고 stable feature subset + target-specific LGB params만 적용.",
        "- 공유 LSTM 노트북은 모델이 아니라 lag/rolling, stable feature selection 아이디어만 인용.",
        "- 기존 best 파일은 덮어쓰지 않음.",
    ]
    (ROOT / "submissions" / "README.md").write_text("\n".join(lines) + "\n")


def _write_stability_log(selection: dict[str, Any], *, filename: str) -> None:
    rows: list[dict[str, Any]] = []
    baseline_path = LOG_DIR / "stability" / "stability_histmix_guarded_v1_reproduce.csv"
    baseline_rows = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()

    for target in TARGET_COLUMNS:
        item = selection[target]
        if item["source"] == BASELINE_RUN and not baseline_rows.empty:
            target_rows = baseline_rows[baseline_rows["target"] == target][["target", "fold", "seed", "logloss"]].copy()
        else:
            _, meta_path, _ = _cache_paths(target, item["candidate"])
            meta = json.loads(meta_path.read_text())
            target_rows = pd.DataFrame(meta["fold_rows"])[["target", "fold", "seed", "logloss"]].copy()
        mean_value = float(item["score"])
        std_value = float(target_rows["logloss"].std(ddof=0))
        target_rows["mean"] = mean_value
        target_rows["std"] = std_value
        rows.extend(target_rows.to_dict("records"))

    out_path = LOG_DIR / "stability" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["target", "fold", "seed", "logloss", "mean", "std"]).to_csv(out_path, index=False)


def _append_experiment_log(
    *,
    experiment_name: str,
    scores: dict[str, float],
    submission_file: Path,
    selection: dict[str, Any],
    notes: str,
) -> None:
    path = LOG_DIR / "experiments.csv"
    columns = [
        "timestamp",
        "experiment_name",
        "validation_scheme",
        "seeds",
        "total_oof_logloss",
        "target_logloss_Q1",
        "target_logloss_Q2",
        "target_logloss_Q3",
        "target_logloss_S1",
        "target_logloss_S2",
        "target_logloss_S3",
        "target_logloss_S4",
        "submission_file",
        "feature_view_by_target",
        "notes",
    ]
    rows = _read_csv_rows(path)
    rows = [row for row in rows if row.get("experiment_name") != experiment_name]
    rows.append(
        {
            "timestamp": _now_kst(),
            "experiment_name": experiment_name,
            "validation_scheme": CV_SCHEME,
            "seeds": _safe_json(PUBLIC_LGB_DEFAULT_SEEDS),
            "total_oof_logloss": scores["mean"],
            "target_logloss_Q1": scores["Q1"],
            "target_logloss_Q2": scores["Q2"],
            "target_logloss_Q3": scores["Q3"],
            "target_logloss_S1": scores["S1"],
            "target_logloss_S2": scores["S2"],
            "target_logloss_S3": scores["S3"],
            "target_logloss_S4": scores["S4"],
            "submission_file": str(submission_file.relative_to(ROOT)),
            "feature_view_by_target": _safe_json(selection),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_reports(
    *,
    target_summaries: list[dict[str, Any]],
    primary_selection: dict[str, Any],
    conservative_selection: dict[str, Any],
    primary_scores: dict[str, float],
    conservative_scores: dict[str, float],
) -> None:
    feature_lines = [
        "# Guarded V2 Stable Tuned Feature Stability",
        "",
        "| Target | View | Baseline | Selected | Score | Drift mean | Drift max |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in target_summaries:
        selected = primary_selection[row["target"]]
        feature_lines.append(
            f"| {row['target']} | `{row['view']}` | {row['baseline_score']:.6f} | "
            f"`{selected['candidate']}` ({selected['n_features']}) | {selected['score']:.6f} | "
            f"{selected['drift']['mean_abs']:.6f} | {selected['drift']['max_abs']:.6f} |"
        )
    write_markdown(REPORT_FEATURES_DIR / f"{EXPERIMENT_NAME}.md", "\n".join(feature_lines))

    submission_lines = [
        f"# Submission Report: {EXPERIMENT_NAME}",
        "",
        f"- Current best public: `{CURRENT_BEST_PUBLIC:.10f}`",
        f"- Current best OOF: `{CURRENT_BEST_OOF:.6f}`",
        f"- Primary OOF: `{primary_scores['mean']:.6f}`",
        f"- Conservative OOF: `{conservative_scores['mean']:.6f}`",
        "",
        "## Primary Selection",
        "",
        "| Target | Candidate | Features | OOF | Source |",
        "|---|---|---:|---:|---|",
    ]
    for target in TARGET_COLUMNS:
        item = primary_selection[target]
        submission_lines.append(
            f"| {target} | `{item['candidate']}` | {item['n_features']} | {item['score']:.6f} | {item['source']} |"
        )
    submission_lines.extend(["", "## Conservative Selection", "", "| Target | Candidate | Features | OOF | Source |", "|---|---|---:|---:|---|"])
    for target in TARGET_COLUMNS:
        item = conservative_selection[target]
        submission_lines.append(
            f"| {target} | `{item['candidate']}` | {item['n_features']} | {item['score']:.6f} | {item['source']} |"
        )
    write_markdown(REPORT_SUBMISSIONS_DIR / f"{EXPERIMENT_NAME}.md", "\n".join(submission_lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite ready submissions if they already exist.")
    parser.add_argument("--retrain", action="store_true", help="Ignore cached target/K candidates.")
    args = parser.parse_args()

    ensure_runtime_dirs()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_public_lgb_feature_table(rebuild=False)
    train = frame[frame["split"] == "train"].reset_index(drop=True).copy()
    test = frame[frame["split"] == "test"].reset_index(drop=True).copy()
    if len(train) != 450 or len(test) != 250:
        raise ValueError(f"Unexpected row counts: train={len(train)} test={len(test)}")
    missing_targets = [target for target in TARGET_COLUMNS if target not in train.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns: {missing_targets}")

    labels, baseline_oof, baseline_test, baseline_scores = _load_baseline()
    train_keys = pd.read_parquet(OOF_DIR / f"oof_predictions_{BASELINE_RUN}.parquet")[KEY_COLUMNS]
    seeds = PUBLIC_LGB_DEFAULT_SEEDS.copy()

    print(f"baseline mean: {baseline_scores['mean']:.6f}")
    target_results: dict[str, list[dict[str, Any]]] = {}
    target_summaries: list[dict[str, Any]] = []

    for target in TARGET_COLUMNS:
        view = TARGET_VIEW[target]
        all_cols = get_public_lgb_feature_columns(frame, view)
        print(f"\n[{target}] view={view} features={len(all_cols)}")
        all_result = _train_target_candidate(
            train=train,
            test=test,
            labels=labels,
            target=target,
            feature_cols=all_cols,
            label="all",
            seeds=seeds,
            retrain=args.retrain,
            collect_importance=True,
        )
        imp_df = all_result["importance"]
        if imp_df is None:
            _, _, imp_path = _cache_paths(target, "all")
            imp_df = pd.read_parquet(imp_path)

        results = [all_result]
        X_view = train[all_cols].copy()
        seen_subsets = {tuple(all_cols)}
        for k in TOP_K_BY_VIEW[view]:
            if k == "all":
                continue
            subset = _select_stable_features(imp_df, X_view, int(k))
            key = tuple(subset)
            if key in seen_subsets:
                continue
            seen_subsets.add(key)
            result = _train_target_candidate(
                train=train,
                test=test,
                labels=labels,
                target=target,
                feature_cols=subset,
                label=_candidate_label(k),
                seeds=seeds,
                retrain=args.retrain,
                collect_importance=False,
            )
            results.append(result)

        for result in results:
            result["drift"] = _drift_stats(result["test"], baseline_test[target].to_numpy(float))
            print(
                f"  {result['candidate']:>6s} n={result['n_features']:3d} "
                f"oof={result['score']:.6f} "
                f"drift={result['drift']['mean_abs']:.5f}/{result['drift']['max_abs']:.5f}"
            )
        target_results[target] = results
        target_summaries.append(
            {
                "target": target,
                "view": view,
                "baseline_score": baseline_scores[target],
                "candidates": [
                    {
                        "candidate": r["candidate"],
                        "n_features": r["n_features"],
                        "score": r["score"],
                        "drift": r["drift"],
                    }
                    for r in results
                ],
            }
        )

    primary_oof = baseline_oof.copy()
    primary_test = baseline_test.copy()
    conservative_oof = baseline_oof.copy()
    conservative_test = baseline_test.copy()
    primary_selection: dict[str, Any] = {}
    conservative_selection: dict[str, Any] = {}

    for target in TARGET_COLUMNS:
        baseline_item = {
            "target": target,
            "candidate": "baseline",
            "n_features": len(get_public_lgb_feature_columns(frame, TARGET_VIEW[target])),
            "score": baseline_scores[target],
            "source": BASELINE_RUN,
            "drift": {"mean_abs": 0.0, "max_abs": 0.0, "ok": True},
        }
        tolerance = 0.0 if target in PROTECTED_TARGETS else OTHER_TARGET_TOL
        feasible = [r for r in target_results[target] if r["score"] <= baseline_scores[target] + tolerance]
        primary_item = min(feasible, key=lambda r: r["score"]) if feasible else baseline_item
        if primary_item is not baseline_item:
            primary_oof[target] = primary_item["oof"]
            primary_test[target] = primary_item["test"]
            primary_selection[target] = {
                "target": target,
                "candidate": primary_item["candidate"],
                "n_features": primary_item["n_features"],
                "score": primary_item["score"],
                "source": EXPERIMENT_NAME,
                "drift": primary_item["drift"],
            }
        else:
            primary_selection[target] = baseline_item

        conservative_feasible = [
            r
            for r in target_results[target]
            if r["score"] <= baseline_scores[target] - CONSERVATIVE_MIN_GAIN and r["drift"]["ok"]
        ]
        conservative_item = min(conservative_feasible, key=lambda r: r["score"]) if conservative_feasible else baseline_item
        if conservative_item is not baseline_item:
            conservative_oof[target] = conservative_item["oof"]
            conservative_test[target] = conservative_item["test"]
            conservative_selection[target] = {
                "target": target,
                "candidate": conservative_item["candidate"],
                "n_features": conservative_item["n_features"],
                "score": conservative_item["score"],
                "source": EXPERIMENT_NAME,
                "drift": conservative_item["drift"],
            }
        else:
            conservative_selection[target] = baseline_item

    primary_scores = _score_targets(labels, primary_oof)
    conservative_scores = _score_targets(labels, conservative_oof)
    primary_drift_high = any(
        item["source"] != BASELINE_RUN and not item["drift"]["ok"] for item in primary_selection.values()
    )
    if primary_scores["mean"] > baseline_scores["mean"] + PRIMARY_MEAN_TOL or primary_drift_high:
        print("\nPrimary selection failed global safety; using conservative predictions for next_stable_tuned.csv")
        primary_oof = conservative_oof.copy()
        primary_test = conservative_test.copy()
        primary_selection = conservative_selection.copy()
        primary_scores = conservative_scores.copy()

    primary_path = READY_DIR / "next_stable_tuned.csv"
    conservative_path = READY_DIR / "next_stable_tuned_conservative.csv"
    _make_submission(primary_path, primary_test, overwrite=args.overwrite)
    _make_submission(conservative_path, conservative_test, overwrite=args.overwrite)

    _write_prediction_artifacts(
        run_name=PRIMARY_RUN,
        train_keys=train_keys,
        labels=labels,
        oof_pred=primary_oof,
        test_pred=primary_test,
        scores=primary_scores,
        selection=primary_selection,
    )
    _write_prediction_artifacts(
        run_name=CONSERVATIVE_RUN,
        train_keys=train_keys,
        labels=labels,
        oof_pred=conservative_oof,
        test_pred=conservative_test,
        scores=conservative_scores,
        selection=conservative_selection,
    )

    write_json(
        FEATURES_DIR / f"{EXPERIMENT_NAME}_feature_subsets.json",
        {
            "experiment_name": EXPERIMENT_NAME,
            "target_view": TARGET_VIEW,
            "top_k_by_view": TOP_K_BY_VIEW,
            "primary_selection": primary_selection,
            "conservative_selection": conservative_selection,
            "target_summaries": target_summaries,
            "baseline_scores": baseline_scores,
            "primary_scores": primary_scores,
            "conservative_scores": conservative_scores,
        },
    )

    _update_public_scores()
    _write_candidate_outputs(
        primary_scores=primary_scores,
        conservative_scores=conservative_scores,
        primary_path=primary_path,
        conservative_path=conservative_path,
        primary_notes="Stable feature subset + target-specific LGB params; safe target-wise assembly.",
        conservative_notes="More anchored stable_tuned version; only adopts low-drift OOF-improving targets.",
    )
    _write_readme(primary_path, conservative_path, primary_scores, conservative_scores)
    _append_experiment_log(
        experiment_name=EXPERIMENT_NAME,
        scores=primary_scores,
        submission_file=primary_path,
        selection=primary_selection,
        notes="LSTM notebook ideas used only as stable feature selection / lag-rolling inspiration; no PyTorch blend.",
    )
    _append_experiment_log(
        experiment_name=f"{EXPERIMENT_NAME}_conservative",
        scores=conservative_scores,
        submission_file=conservative_path,
        selection=conservative_selection,
        notes="Conservative low-drift subset of guarded_v2_stable_tuned.",
    )
    _write_stability_log(primary_selection, filename=f"stability_{EXPERIMENT_NAME}.csv")
    _write_stability_log(conservative_selection, filename=f"stability_{EXPERIMENT_NAME}_conservative.csv")
    _write_reports(
        target_summaries=target_summaries,
        primary_selection=primary_selection,
        conservative_selection=conservative_selection,
        primary_scores=primary_scores,
        conservative_scores=conservative_scores,
    )

    print("\n=== guarded_v2_stable_tuned ===")
    print(f"primary: {primary_path} OOF={primary_scores['mean']:.6f}")
    print(f"conservative: {conservative_path} OOF={conservative_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(
            f"  {target}: primary={primary_scores[target]:.6f} "
            f"conservative={conservative_scores[target]:.6f} "
            f"baseline={baseline_scores[target]:.6f}"
        )


if __name__ == "__main__":
    main()

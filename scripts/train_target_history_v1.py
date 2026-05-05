#!/usr/bin/env python3
"""Train fold-safe bidirectional target-history candidates."""

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


EXPERIMENT_NAME = "target_history_v1"
RUN_NAME = "public_lgb_targetwise_target_history_v1"
CONSERVATIVE_RUN_NAME = "public_lgb_targetwise_target_history_v1_conservative"
ANCHOR_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
CURRENT_PUBLIC_BEST_FILE = "next_stable_tuned.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5956332255
PREVIOUS_PUBLIC_BEST_SCORE = 0.5960566585

CLIP_MIN = 0.02
CLIP_MAX = 0.98
N_FOLDS = 5
NEIGHBOR_COUNT = 3
HISTORY_WINDOWS = [3, 7, 14]
PRIMARY_GAIN = 0.0
CONSERVATIVE_GAIN = 0.001
DRIFT_MEAN_LIMIT = 0.02
DRIFT_MAX_LIMIT = 0.08

READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
EXP_DIR = EXPERIMENTS_DIR / EXPERIMENT_NAME
STABLE_EXP_DIR = EXPERIMENTS_DIR / "guarded_v2_stable_tuned"

TARGET_VIEW = {
    "Q1": "public_core",
    "Q2": "public_hist411",
    "Q3": "public_hist365",
    "S1": "public_core",
    "S2": "public_core",
    "S3": "public_core",
    "S4": "public_hist411",
}


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


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _load_anchor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    oof_path = OOF_DIR / f"oof_predictions_{ANCHOR_RUN}.parquet"
    test_path = MODELS_DIR / f"test_predictions_{ANCHOR_RUN}.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing anchor OOF. Run stable_tuned first: {oof_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing anchor test predictions: {test_path}")
    raw = pd.read_parquet(oof_path)
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(test_path)[TARGET_COLUMNS].astype(float)
    return labels, oof, test, _score_targets(labels, oof)


def _load_anchor_selection() -> dict[str, Any]:
    path = FEATURES_DIR / "guarded_v2_stable_tuned_feature_subsets.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing stable selection: {path}")
    return json.loads(path.read_text())["primary_selection"]


def _load_base_feature_cols(frame: pd.DataFrame, target: str, selection: dict[str, Any]) -> list[str]:
    item = selection[target]
    view = TARGET_VIEW[target]
    if item["candidate"] == "baseline":
        return get_public_lgb_feature_columns(frame, view)
    meta_path = STABLE_EXP_DIR / f"{target}_{view}_{item['candidate']}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing stable feature metadata: {meta_path}")
    return json.loads(meta_path.read_text())["feature_cols"]


def _target_history_columns() -> list[str]:
    cols: list[str] = []
    for source in TARGET_COLUMNS:
        prefix = f"th_{source}"
        cols.extend([f"{prefix}_prior_mean", f"{prefix}_prior_count"])
        for rank in range(1, NEIGHBOR_COUNT + 1):
            cols.extend(
                [
                    f"{prefix}_prev{rank}_value",
                    f"{prefix}_prev{rank}_days",
                    f"{prefix}_next{rank}_value",
                    f"{prefix}_next{rank}_days",
                ]
            )
        for window in HISTORY_WINDOWS:
            cols.extend(
                [
                    f"{prefix}_past{window}_mean",
                    f"{prefix}_past{window}_count",
                    f"{prefix}_future{window}_mean",
                    f"{prefix}_future{window}_count",
                ]
            )
        cols.extend(
            [
                f"{prefix}_nearest_value",
                f"{prefix}_nearest_days",
                f"{prefix}_nearest_is_future",
                f"{prefix}_prev_next_same",
            ]
        )
    return cols


def _make_reference_by_subject(reference: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = ["_row_id", "subject_id", "lifelog_date"] + TARGET_COLUMNS
    ref = reference[cols].copy()
    ref["lifelog_date"] = pd.to_datetime(ref["lifelog_date"])
    return {
        subject: group.sort_values("lifelog_date").reset_index(drop=True)
        for subject, group in ref.groupby("subject_id", sort=False)
    }


def _build_target_history(sample: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    ref_by_subject = _make_reference_by_subject(reference)
    columns = _target_history_columns()
    rows: list[dict[str, float]] = []

    sample_frame = sample[["_row_id", "subject_id", "lifelog_date"]].copy()
    sample_frame["lifelog_date"] = pd.to_datetime(sample_frame["lifelog_date"])

    for sample_row in sample_frame.to_dict("records"):
        sample_id = sample_row["_row_id"]
        subject_id = sample_row["subject_id"]
        current_date = sample_row["lifelog_date"]
        ref = ref_by_subject.get(subject_id)
        record: dict[str, float] = {}
        if ref is None or ref.empty:
            rows.append({col: np.nan for col in columns})
            continue

        dates = ref["lifelog_date"]
        before_mask = dates < current_date
        after_mask = dates > current_date
        current_delta = (dates - current_date).dt.days.to_numpy(float)

        for source in TARGET_COLUMNS:
            prefix = f"th_{source}"
            values = ref[source].astype(float)
            valid_values = values.dropna()
            record[f"{prefix}_prior_mean"] = float(valid_values.mean()) if len(valid_values) else np.nan
            record[f"{prefix}_prior_count"] = float(len(valid_values))

            prev = ref.loc[before_mask, ["lifelog_date", source]].dropna().tail(NEIGHBOR_COUNT)
            prev = prev.iloc[::-1].reset_index(drop=True)
            nxt = ref.loc[after_mask, ["lifelog_date", source]].dropna().head(NEIGHBOR_COUNT).reset_index(drop=True)
            for rank in range(1, NEIGHBOR_COUNT + 1):
                if rank <= len(prev):
                    item = prev.iloc[rank - 1]
                    record[f"{prefix}_prev{rank}_value"] = float(item[source])
                    record[f"{prefix}_prev{rank}_days"] = float((current_date - item["lifelog_date"]).days)
                else:
                    record[f"{prefix}_prev{rank}_value"] = np.nan
                    record[f"{prefix}_prev{rank}_days"] = np.nan
                if rank <= len(nxt):
                    item = nxt.iloc[rank - 1]
                    record[f"{prefix}_next{rank}_value"] = float(item[source])
                    record[f"{prefix}_next{rank}_days"] = float((item["lifelog_date"] - current_date).days)
                else:
                    record[f"{prefix}_next{rank}_value"] = np.nan
                    record[f"{prefix}_next{rank}_days"] = np.nan

            for window in HISTORY_WINDOWS:
                past_values = ref.loc[before_mask & (current_delta >= -window), source].dropna()
                future_values = ref.loc[after_mask & (current_delta <= window), source].dropna()
                record[f"{prefix}_past{window}_mean"] = float(past_values.mean()) if len(past_values) else np.nan
                record[f"{prefix}_past{window}_count"] = float(len(past_values))
                record[f"{prefix}_future{window}_mean"] = float(future_values.mean()) if len(future_values) else np.nan
                record[f"{prefix}_future{window}_count"] = float(len(future_values))

            known = ref[["_row_id", "lifelog_date", source]].dropna().copy()
            known = known[known["_row_id"] != sample_id]
            if known.empty:
                record[f"{prefix}_nearest_value"] = np.nan
                record[f"{prefix}_nearest_days"] = np.nan
                record[f"{prefix}_nearest_is_future"] = np.nan
            else:
                dist = (known["lifelog_date"] - current_date).dt.days
                nearest_idx = dist.abs().idxmin()
                nearest_days = float(abs(dist.loc[nearest_idx]))
                record[f"{prefix}_nearest_value"] = float(known.loc[nearest_idx, source])
                record[f"{prefix}_nearest_days"] = nearest_days
                record[f"{prefix}_nearest_is_future"] = float(dist.loc[nearest_idx] > 0)

            if len(prev) and len(nxt):
                record[f"{prefix}_prev_next_same"] = float(prev.iloc[0][source] == nxt.iloc[0][source])
            else:
                record[f"{prefix}_prev_next_same"] = np.nan

        rows.append(record)

    return pd.DataFrame(rows, columns=columns, index=sample.index)


def _params_for_target(target: str, seed: int) -> dict[str, Any]:
    return {**TARGET_LGB_PARAMS.get(target, PUBLIC_LGB_PARAMS), "random_state": int(seed)}


def _cache_paths(target: str) -> tuple[Path, Path]:
    return EXP_DIR / f"{target}_target_history_v1.npz", EXP_DIR / f"{target}_target_history_v1.json"


def _train_target(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    seeds: list[int],
    retrain: bool,
) -> dict[str, Any]:
    npz_path, meta_path = _cache_paths(target)
    if npz_path.exists() and meta_path.exists() and not retrain:
        meta = json.loads(meta_path.read_text())
        if meta.get("feature_cols") == feature_cols:
            data = np.load(npz_path)
            return {**meta, "oof": data["oof"], "test": data["test"]}

    X_base_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_base_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = train[target].astype(int).to_numpy()
    test_hist = _build_target_history(test, train)

    oof_total = np.zeros(len(train), dtype=float)
    test_total = np.zeros(len(test), dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for seed in seeds:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed))
        seed_oof = np.zeros(len(train), dtype=float)
        seed_test = np.zeros(len(test), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_base_train, y)):
            tr_ref = train.iloc[train_idx].copy()
            tr_sample = train.iloc[train_idx].copy()
            va_sample = train.iloc[valid_idx].copy()
            tr_hist = _build_target_history(tr_sample, tr_ref)
            va_hist = _build_target_history(va_sample, tr_ref)

            X_tr = pd.concat([X_base_train.iloc[train_idx].reset_index(drop=True), tr_hist.reset_index(drop=True)], axis=1)
            X_va = pd.concat([X_base_train.iloc[valid_idx].reset_index(drop=True), va_hist.reset_index(drop=True)], axis=1)
            X_te = pd.concat([X_base_test.reset_index(drop=True), test_hist.reset_index(drop=True)], axis=1)

            model = lgb.LGBMClassifier(**_params_for_target(target, int(seed)))
            model.fit(
                X_tr,
                y[train_idx],
                eval_set=[(X_va, y[valid_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
            )
            valid_pred = model.predict_proba(X_va)[:, 1]
            seed_oof[valid_idx] = valid_pred
            seed_test += model.predict_proba(X_te)[:, 1] / N_FOLDS
            fold_rows.append(
                {
                    "target": target,
                    "fold": int(fold),
                    "seed": int(seed),
                    "logloss": binary_log_loss(y[valid_idx], valid_pred),
                }
            )
        oof_total += seed_oof
        test_total += seed_test

    oof = oof_total / len(seeds)
    test_pred = test_total / len(seeds)
    score = binary_log_loss(labels[target].to_numpy(float), oof)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, oof=oof, test=test_pred)
    meta = {
        "target": target,
        "n_base_features": len(feature_cols),
        "n_history_features": len(_target_history_columns()),
        "feature_cols": feature_cols,
        "score": score,
        "fold_rows": fold_rows,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return {**meta, "oof": oof, "test": test_pred}


def _drift_stats(candidate: np.ndarray, anchor: np.ndarray) -> dict[str, float | bool]:
    diff = np.abs(candidate - anchor)
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
    oof_export["split_scheme"] = "public_stratified_fold_safe_target_history"
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
            "scores": scores,
            "selection": selection,
            "history_features": _target_history_columns(),
        },
    )


def _write_stability(selection: dict[str, Any], *, filename: str) -> None:
    rows: list[dict[str, Any]] = []
    anchor_path = LOG_DIR / "stability" / "stability_guarded_v2_stable_tuned.csv"
    anchor_rows = pd.read_csv(anchor_path) if anchor_path.exists() else pd.DataFrame()
    for target in TARGET_COLUMNS:
        item = selection[target]
        if item["source"] == ANCHOR_RUN and not anchor_rows.empty:
            target_rows = anchor_rows[anchor_rows["target"] == target][["target", "fold", "seed", "logloss"]].copy()
        else:
            _, meta_path = _cache_paths(target)
            target_rows = pd.DataFrame(json.loads(meta_path.read_text())["fold_rows"])[["target", "fold", "seed", "logloss"]].copy()
        target_rows["mean"] = float(item["score"])
        target_rows["std"] = float(target_rows["logloss"].std(ddof=0))
        rows.extend(target_rows.to_dict("records"))
    out = LOG_DIR / "stability" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["target", "fold", "seed", "logloss", "mean", "std"]).to_csv(out, index=False)


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


def _write_candidate_scores(primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    rows = [
        {
            "rank": 0,
            "candidate": "next_stable_tuned.csv",
            "oof_mean": 0.5687692970981916,
            "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}",
            "submission_file": "submissions/ready/next_stable_tuned.csv",
            "notes": "Current public best.",
        },
        {
            "rank": 1,
            "candidate": "next_target_history.csv",
            "oof_mean": primary_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/next_target_history.csv",
            "notes": "Fold-safe bidirectional train-label history features on top of stable_tuned anchor.",
        },
        {
            "rank": 2,
            "candidate": "next_target_history_conservative.csv",
            "oof_mean": conservative_scores["mean"],
            "public_score": "",
            "submission_file": "submissions/ready/next_target_history_conservative.csv",
            "notes": "Conservative target-history version; only adopts high-gain low-drift targets.",
        },
        {
            "rank": 3,
            "candidate": "00_best_guarded.csv",
            "oof_mean": 0.5721963000561784,
            "public_score": "0.5960566585",
            "submission_file": "submissions/ready/00_best_guarded.csv",
            "notes": "Previous public best anchor.",
        },
        {
            "rank": 4,
            "candidate": "next_guarded_s2s3.csv",
            "oof_mean": 0.5721908206619111,
            "public_score": "0.5960906414",
            "submission_file": "submissions/ready/next_guarded_s2s3.csv",
            "notes": "Submitted and worse; stopped.",
        },
    ]
    _write_csv_rows(
        LOG_DIR / "candidate_scores.csv",
        rows,
        ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"],
    )


def _append_experiment_log(name: str, scores: dict[str, float], submission: Path, selection: dict[str, Any], notes: str) -> None:
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
    rows = [row for row in rows if row.get("experiment_name") != name]
    rows.append(
        {
            "timestamp": _now_kst(),
            "experiment_name": name,
            "validation_scheme": "public_stratified_fold_safe_target_history",
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
            "feature_view_by_target": json.dumps(selection, ensure_ascii=False, sort_keys=True),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_readme(primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    lines = [
        "# Submissions",
        "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.",
        "",
        "## Current Best",
        "",
        f"- Public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        "- Best file: `submissions/ready/next_stable_tuned.csv`",
        "",
        "## Recommended Order",
        "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/next_stable_tuned.csv` | `0.568769` | `{CURRENT_PUBLIC_BEST_SCORE:.10f}` | 현재 best |",
        f"| 1 | `ready/next_target_history.csv` | `{primary_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/next_target_history_conservative.csv` | `{conservative_scores['mean']:.6f}` |  | 1순위 실패 시 제출 |",
        "| 3 | `ready/00_best_guarded.csv` | `0.572196` | `0.5960566585` | 이전 best |",
        "",
        "## Notes",
        "",
        "- `target_history_v1`: fold-safe bidirectional target history features.",
        "- train OOF에서는 validation label을 feature reference에서 제외.",
        "- test 예측에서는 사용 가능한 전체 train label history를 사용.",
    ]
    (ROOT / "submissions" / "README.md").write_text("\n".join(lines) + "\n")


def _write_report(primary_selection: dict[str, Any], conservative_selection: dict[str, Any], primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    lines = [
        "# Submission Report: target_history_v1",
        "",
        f"- Anchor: `{ANCHOR_RUN}` public `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Primary OOF: `{primary_scores['mean']:.6f}`",
        f"- Conservative OOF: `{conservative_scores['mean']:.6f}`",
        f"- History feature count: `{len(_target_history_columns())}`",
        "",
        "## Primary Selection",
        "",
        "| Target | Source | OOF | Drift mean | Drift max |",
        "|---|---|---:|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        item = primary_selection[target]
        lines.append(
            f"| {target} | `{item['source']}` | {item['score']:.6f} | "
            f"{item['drift']['mean_abs']:.6f} | {item['drift']['max_abs']:.6f} |"
        )
    lines.extend(["", "## Conservative Selection", "", "| Target | Source | OOF | Drift mean | Drift max |", "|---|---|---:|---:|---:|"])
    for target in TARGET_COLUMNS:
        item = conservative_selection[target]
        lines.append(
            f"| {target} | `{item['source']}` | {item['score']:.6f} | "
            f"{item['drift']['mean_abs']:.6f} | {item['drift']['max_abs']:.6f} |"
        )
    write_markdown(REPORT_SUBMISSIONS_DIR / "target_history_v1.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    ensure_runtime_dirs()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_public_lgb_feature_table(rebuild=False)
    train = frame[frame["split"] == "train"].reset_index(drop=True).copy()
    test = frame[frame["split"] == "test"].reset_index(drop=True).copy()
    if len(train) != 450 or len(test) != 250:
        raise ValueError(f"Unexpected row counts: train={len(train)} test={len(test)}")
    for target in TARGET_COLUMNS:
        if target not in train.columns:
            raise KeyError(f"Missing target column: {target}")
    train["_row_id"] = np.arange(len(train), dtype=int)
    test["_row_id"] = -1 - np.arange(len(test), dtype=int)

    labels, anchor_oof, anchor_test, anchor_scores = _load_anchor()
    train_keys = pd.read_parquet(OOF_DIR / f"oof_predictions_{ANCHOR_RUN}.parquet")[KEY_COLUMNS]
    stable_selection = _load_anchor_selection()
    seeds = PUBLIC_LGB_DEFAULT_SEEDS.copy()

    target_results: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        feature_cols = _load_base_feature_cols(frame, target, stable_selection)
        print(f"\n[{target}] base_features={len(feature_cols)} history_features={len(_target_history_columns())}")
        result = _train_target(
            train=train,
            test=test,
            labels=labels,
            target=target,
            feature_cols=feature_cols,
            seeds=seeds,
            retrain=args.retrain,
        )
        result["drift"] = _drift_stats(result["test"], anchor_test[target].to_numpy(float))
        target_results[target] = result
        print(
            f"  history oof={result['score']:.6f} anchor={anchor_scores[target]:.6f} "
            f"drift={result['drift']['mean_abs']:.5f}/{result['drift']['max_abs']:.5f}"
        )

    primary_oof = anchor_oof.copy()
    primary_test = anchor_test.copy()
    conservative_oof = anchor_oof.copy()
    conservative_test = anchor_test.copy()
    primary_selection: dict[str, Any] = {}
    conservative_selection: dict[str, Any] = {}

    for target in TARGET_COLUMNS:
        anchor_item = {
            "target": target,
            "source": ANCHOR_RUN,
            "score": anchor_scores[target],
            "drift": {"mean_abs": 0.0, "max_abs": 0.0, "ok": True},
        }
        result = target_results[target]
        history_item = {
            "target": target,
            "source": EXPERIMENT_NAME,
            "score": result["score"],
            "drift": result["drift"],
        }
        if result["score"] < anchor_scores[target] - PRIMARY_GAIN:
            primary_oof[target] = result["oof"]
            primary_test[target] = result["test"]
            primary_selection[target] = history_item
        else:
            primary_selection[target] = anchor_item
        if result["score"] < anchor_scores[target] - CONSERVATIVE_GAIN and result["drift"]["ok"]:
            conservative_oof[target] = result["oof"]
            conservative_test[target] = result["test"]
            conservative_selection[target] = history_item
        else:
            conservative_selection[target] = anchor_item

    primary_scores = _score_targets(labels, primary_oof)
    conservative_scores = _score_targets(labels, conservative_oof)
    primary_path = READY_DIR / "next_target_history.csv"
    conservative_path = READY_DIR / "next_target_history_conservative.csv"
    _make_submission(primary_path, primary_test, overwrite=args.overwrite)
    _make_submission(conservative_path, conservative_test, overwrite=args.overwrite)

    _write_prediction_artifacts(
        run_name=RUN_NAME,
        train_keys=train_keys,
        labels=labels,
        oof_pred=primary_oof,
        test_pred=primary_test,
        scores=primary_scores,
        selection=primary_selection,
    )
    _write_prediction_artifacts(
        run_name=CONSERVATIVE_RUN_NAME,
        train_keys=train_keys,
        labels=labels,
        oof_pred=conservative_oof,
        test_pred=conservative_test,
        scores=conservative_scores,
        selection=conservative_selection,
    )
    write_json(
        FEATURES_DIR / "target_history_v1_selection.json",
        {
            "anchor_run": ANCHOR_RUN,
            "anchor_scores": anchor_scores,
            "primary_scores": primary_scores,
            "conservative_scores": conservative_scores,
            "primary_selection": primary_selection,
            "conservative_selection": conservative_selection,
            "target_results": {
                target: {
                    "score": result["score"],
                    "drift": result["drift"],
                    "n_base_features": result["n_base_features"],
                    "n_history_features": result["n_history_features"],
                }
                for target, result in target_results.items()
            },
        },
    )

    _update_public_scores()
    _write_candidate_scores(primary_scores, conservative_scores)
    _append_experiment_log(
        EXPERIMENT_NAME,
        primary_scores,
        primary_path,
        primary_selection,
        "Fold-safe bidirectional target history features; validation labels excluded from history reference.",
    )
    _append_experiment_log(
        f"{EXPERIMENT_NAME}_conservative",
        conservative_scores,
        conservative_path,
        conservative_selection,
        "Conservative target history candidate; only high-gain low-drift target adoptions.",
    )
    _write_stability(primary_selection, filename=f"stability_{EXPERIMENT_NAME}.csv")
    _write_stability(conservative_selection, filename=f"stability_{EXPERIMENT_NAME}_conservative.csv")
    _write_readme(primary_scores, conservative_scores)
    _write_report(primary_selection, conservative_selection, primary_scores, conservative_scores)

    print("\n=== target_history_v1 ===")
    print(f"primary: {primary_path} OOF={primary_scores['mean']:.6f}")
    print(f"conservative: {conservative_path} OOF={conservative_scores['mean']:.6f}")
    for target in TARGET_COLUMNS:
        print(
            f"  {target}: primary={primary_scores[target]:.6f} "
            f"conservative={conservative_scores[target]:.6f} anchor={anchor_scores[target]:.6f}"
        )


if __name__ == "__main__":
    main()

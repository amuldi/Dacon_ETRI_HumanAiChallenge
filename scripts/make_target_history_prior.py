#!/usr/bin/env python3
"""Build fast fold-safe target-history prior blend candidates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_SUBMISSIONS_DIR, ROOT, ensure_runtime_dirs
from etri_human_challenge.public_lgb import PUBLIC_LGB_DEFAULT_SEEDS, load_public_lgb_feature_table
from etri_human_challenge.utils import binary_log_loss, write_json, write_markdown


EXPERIMENT_NAME = "target_history_prior_v1"
ANCHOR_RUN = "public_lgb_targetwise_guarded_v2_stable_tuned"
RUN_NAME = "public_lgb_targetwise_target_history_prior_v1"
CONSERVATIVE_RUN_NAME = "public_lgb_targetwise_target_history_prior_v1_conservative"

CURRENT_PUBLIC_BEST_FILE = "next_stable_tuned.csv"
CURRENT_PUBLIC_BEST_SCORE = 0.5956332255
PREVIOUS_PUBLIC_BEST_SCORE = 0.5960566585
ANCHOR_OOF_MEAN = 0.5687692970981916

CLIP_MIN = 0.02
CLIP_MAX = 0.98
N_FOLDS = 5
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

MAX_DAYS_GRID = [14, 28]
POWER_GRID = [1.25]
SUBJECT_WEIGHT_GRID = [0.0, 0.15]
FUTURE_SCALE_GRID = [1.0]
GLOBAL_WEIGHT = 0.05
BLEND_WEIGHTS = np.linspace(0.0, 0.9, 46)
CONSERVATIVE_MAX_WEIGHT = 0.35
CONSERVATIVE_MIN_GAIN = 0.0002
DRIFT_MEAN_LIMIT = 0.020
DRIFT_MAX_LIMIT = 0.080


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


def _load_anchor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    oof_path = OOF_DIR / f"oof_predictions_{ANCHOR_RUN}.parquet"
    test_path = MODELS_DIR / f"test_predictions_{ANCHOR_RUN}.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing anchor OOF: {oof_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing anchor test predictions: {test_path}")
    raw = pd.read_parquet(oof_path)
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(test_path)[TARGET_COLUMNS].astype(float)
    return labels, oof, test, _score_targets(labels, oof), raw[KEY_COLUMNS].copy()


def _prepare_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
    test["lifelog_date"] = pd.to_datetime(test["lifelog_date"])
    return train, test


def _reference_by_subject(reference: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    cols = ["_row_id", "subject_id", "lifelog_date", target]
    return {
        subject: group[cols].sort_values("lifelog_date").reset_index(drop=True)
        for subject, group in reference.groupby("subject_id", sort=False)
    }


def _predict_prior(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    target: str,
    *,
    max_days: int,
    power: float,
    subject_weight: float,
    future_scale: float,
) -> np.ndarray:
    ref_by_subject = _reference_by_subject(reference, target)
    global_mean = float(reference[target].mean())
    output = np.zeros(len(sample), dtype=float)

    for out_idx, row in enumerate(sample[["_row_id", "subject_id", "lifelog_date"]].itertuples(index=False, name=None)):
        row_id, subject_id, current_date = row
        ref = ref_by_subject.get(subject_id)
        if ref is None or ref.empty:
            output[out_idx] = global_mean
            continue

        known = ref[ref["_row_id"] != row_id].dropna(subset=[target]).copy()
        if known.empty:
            output[out_idx] = global_mean
            continue

        delta = (known["lifelog_date"] - current_date).dt.days.to_numpy(float)
        abs_delta = np.abs(delta)
        values = known[target].to_numpy(float)
        mask = abs_delta <= max_days
        if not np.any(mask):
            mask = np.ones_like(abs_delta, dtype=bool)

        weights = 1.0 / np.power(abs_delta[mask] + 1.0, power)
        weights = weights * np.where(delta[mask] > 0, future_scale, 1.0)
        temporal = float(np.sum(weights * values[mask]) / np.sum(weights)) if np.sum(weights) > 0 else global_mean
        subject_mean = float(values.mean()) if values.size else global_mean
        temporal_weight = max(0.0, 1.0 - subject_weight - GLOBAL_WEIGHT)
        output[out_idx] = temporal_weight * temporal + subject_weight * subject_mean + GLOBAL_WEIGHT * global_mean

    return np.clip(output, CLIP_MIN, CLIP_MAX)


def _oof_prior_for_config(train: pd.DataFrame, target: str, config: dict[str, Any], seeds: list[int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = train[target].astype(int).to_numpy()
    total = np.zeros(len(train), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed))
        seed_pred = np.zeros(len(train), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train, y)):
            pred = _predict_prior(train.iloc[valid_idx], train.iloc[train_idx], target, **config)
            seed_pred[valid_idx] = pred
            fold_rows.append(
                {
                    "target": target,
                    "fold": int(fold),
                    "seed": int(seed),
                    "logloss": binary_log_loss(y[valid_idx], pred),
                }
            )
        total += seed_pred
    return total / len(seeds), fold_rows


def _find_best_prior(train: pd.DataFrame, labels: pd.DataFrame, target: str, seeds: list[int]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for max_days, power, subject_weight, future_scale in itertools.product(
        MAX_DAYS_GRID, POWER_GRID, SUBJECT_WEIGHT_GRID, FUTURE_SCALE_GRID
    ):
        config = {
            "max_days": int(max_days),
            "power": float(power),
            "subject_weight": float(subject_weight),
            "future_scale": float(future_scale),
        }
        oof, fold_rows = _oof_prior_for_config(train, target, config, seeds)
        score = binary_log_loss(labels[target].to_numpy(float), oof)
        if best is None or score < best["score"]:
            best = {"config": config, "oof": oof, "score": score, "fold_rows": fold_rows}
    assert best is not None
    return best


def _best_blend(
    labels: pd.DataFrame,
    target: str,
    anchor_oof: pd.Series,
    prior_oof: np.ndarray,
    *,
    max_weight: float,
) -> tuple[float, float, np.ndarray]:
    best_w = 0.0
    best_pred = anchor_oof.to_numpy(float)
    best_score = binary_log_loss(labels[target].to_numpy(float), best_pred)
    for weight in BLEND_WEIGHTS:
        if weight > max_weight:
            continue
        pred = np.clip((1.0 - weight) * anchor_oof.to_numpy(float) + weight * prior_oof, CLIP_MIN, CLIP_MAX)
        score = binary_log_loss(labels[target].to_numpy(float), pred)
        if score < best_score:
            best_w = float(weight)
            best_score = score
            best_pred = pred
    return best_w, best_score, best_pred


def _drift(candidate: np.ndarray, anchor: np.ndarray) -> dict[str, float | bool]:
    diff = np.abs(candidate - anchor)
    mean_abs = float(diff.mean())
    max_abs = float(diff.max())
    return {
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "ok": bool(mean_abs <= DRIFT_MEAN_LIMIT and max_abs <= DRIFT_MAX_LIMIT),
    }


def _make_submission(path: Path, predictions: pd.DataFrame, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
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


def _write_prediction_artifacts(run_name: str, keys: pd.DataFrame, labels: pd.DataFrame, oof: pd.DataFrame, test: pd.DataFrame, scores: dict[str, float], selection: dict[str, Any]) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "public_stratified_fold_safe_target_history_prior"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "selection": selection})


def _write_stability(selection: dict[str, Any], prior_rows: dict[str, list[dict[str, Any]]], *, filename: str) -> None:
    anchor_path = LOG_DIR / "stability" / "stability_guarded_v2_stable_tuned.csv"
    anchor_rows = pd.read_csv(anchor_path) if anchor_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        item = selection[target]
        if item["source"] == ANCHOR_RUN and not anchor_rows.empty:
            target_rows = anchor_rows[anchor_rows["target"] == target][["target", "fold", "seed", "logloss"]].copy()
        else:
            target_rows = pd.DataFrame(prior_rows[target])[["target", "fold", "seed", "logloss"]].copy()
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
        {"rank": 0, "candidate": CURRENT_PUBLIC_BEST_FILE, "oof_mean": ANCHOR_OOF_MEAN, "public_score": f"{CURRENT_PUBLIC_BEST_SCORE:.10f}", "submission_file": f"submissions/ready/{CURRENT_PUBLIC_BEST_FILE}", "notes": "Current public best."},
        {"rank": 1, "candidate": "next_target_history.csv", "oof_mean": primary_scores["mean"], "public_score": "", "submission_file": "submissions/ready/next_target_history.csv", "notes": "Target-history prior blended with stable_tuned anchor."},
        {"rank": 2, "candidate": "next_target_history_conservative.csv", "oof_mean": conservative_scores["mean"], "public_score": "", "submission_file": "submissions/ready/next_target_history_conservative.csv", "notes": "Conservative target-history prior blend."},
        {"rank": 3, "candidate": "00_best_guarded.csv", "oof_mean": 0.5721963000561784, "public_score": "0.5960566585", "submission_file": "submissions/ready/00_best_guarded.csv", "notes": "Previous public best anchor."},
    ]
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


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
            "validation_scheme": "public_stratified_fold_safe_target_history_prior",
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
        "# Submissions", "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.", "",
        "## Current Best", "",
        f"- Public best: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_PUBLIC_BEST_FILE}`", "",
        "## Recommended Order", "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_PUBLIC_BEST_FILE}` | `{ANCHOR_OOF_MEAN:.6f}` | `{CURRENT_PUBLIC_BEST_SCORE:.10f}` | 현재 best |",
        f"| 1 | `ready/next_target_history.csv` | `{primary_scores['mean']:.6f}` |  | 다음 제출 |",
        f"| 2 | `ready/next_target_history_conservative.csv` | `{conservative_scores['mean']:.6f}` |  | 1순위 실패 시 제출 |",
        "| 3 | `ready/00_best_guarded.csv` | `0.572196` | `0.5960566585` | 이전 best |", "",
        "## Notes", "",
        "- `target_history_prior_v1`: fold-safe train-label temporal interpolation prior.",
        "- OOF에서는 validation labels를 prior reference에서 제외.",
        "- test에서는 전체 train label history를 사용.",
    ]
    (ROOT / "submissions" / "README.md").write_text("\n".join(lines) + "\n")


def _write_report(primary_selection: dict[str, Any], conservative_selection: dict[str, Any], primary_scores: dict[str, float], conservative_scores: dict[str, float]) -> None:
    lines = [
        "# Submission Report: target_history_prior_v1", "",
        f"- Anchor public: `{CURRENT_PUBLIC_BEST_SCORE:.10f}`",
        f"- Anchor OOF: `{ANCHOR_OOF_MEAN:.6f}`",
        f"- Primary OOF: `{primary_scores['mean']:.6f}`",
        f"- Conservative OOF: `{conservative_scores['mean']:.6f}`", "",
        "## Primary Selection", "",
        "| Target | Source | Weight | OOF | Prior OOF | Config |",
        "|---|---|---:|---:|---:|---|",
    ]
    for target in TARGET_COLUMNS:
        item = primary_selection[target]
        lines.append(f"| {target} | `{item['source']}` | {item['weight']:.2f} | {item['score']:.6f} | {item['prior_score']:.6f} | `{item['config']}` |")
    lines.extend(["", "## Conservative Selection", "", "| Target | Source | Weight | OOF | Prior OOF | Config |", "|---|---|---:|---:|---:|---|"])
    for target in TARGET_COLUMNS:
        item = conservative_selection[target]
        lines.append(f"| {target} | `{item['source']}` | {item['weight']:.2f} | {item['score']:.6f} | {item['prior_score']:.6f} | `{item['config']}` |")
    write_markdown(REPORT_SUBMISSIONS_DIR / "target_history_prior_v1.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_runtime_dirs()
    READY_DIR.mkdir(parents=True, exist_ok=True)
    train, test = _prepare_frames()
    labels, anchor_oof, anchor_test, anchor_scores, train_keys = _load_anchor()
    seeds = PUBLIC_LGB_DEFAULT_SEEDS.copy()

    primary_oof = anchor_oof.copy()
    primary_test = anchor_test.copy()
    conservative_oof = anchor_oof.copy()
    conservative_test = anchor_test.copy()
    primary_selection: dict[str, Any] = {}
    conservative_selection: dict[str, Any] = {}
    prior_rows: dict[str, list[dict[str, Any]]] = {}
    target_details: dict[str, Any] = {}

    for target in TARGET_COLUMNS:
        best_prior = _find_best_prior(train, labels, target, seeds)
        prior_rows[target] = best_prior["fold_rows"]
        test_prior = _predict_prior(test, train, target, **best_prior["config"])
        primary_weight, primary_score, primary_pred = _best_blend(
            labels, target, anchor_oof[target], best_prior["oof"], max_weight=0.9
        )
        conservative_weight, conservative_score, conservative_pred = _best_blend(
            labels, target, anchor_oof[target], best_prior["oof"], max_weight=CONSERVATIVE_MAX_WEIGHT
        )
        primary_test_pred = np.clip((1.0 - primary_weight) * anchor_test[target].to_numpy(float) + primary_weight * test_prior, CLIP_MIN, CLIP_MAX)
        conservative_test_pred = np.clip((1.0 - conservative_weight) * anchor_test[target].to_numpy(float) + conservative_weight * test_prior, CLIP_MIN, CLIP_MAX)
        primary_drift = _drift(primary_test_pred, anchor_test[target].to_numpy(float))
        conservative_drift = _drift(conservative_test_pred, anchor_test[target].to_numpy(float))

        if primary_weight > 0.0 and primary_drift["ok"]:
            primary_oof[target] = primary_pred
            primary_test[target] = primary_test_pred
            primary_source = EXPERIMENT_NAME
        else:
            primary_weight = 0.0
            primary_score = anchor_scores[target]
            primary_drift = {"mean_abs": 0.0, "max_abs": 0.0, "ok": True}
            primary_source = ANCHOR_RUN

        if conservative_weight > 0.0 and conservative_score <= anchor_scores[target] - CONSERVATIVE_MIN_GAIN and conservative_drift["ok"]:
            conservative_oof[target] = conservative_pred
            conservative_test[target] = conservative_test_pred
            conservative_source = EXPERIMENT_NAME
        else:
            conservative_weight = 0.0
            conservative_score = anchor_scores[target]
            conservative_drift = {"mean_abs": 0.0, "max_abs": 0.0, "ok": True}
            conservative_source = ANCHOR_RUN

        primary_selection[target] = {
            "source": primary_source,
            "weight": float(primary_weight),
            "score": float(primary_score),
            "prior_score": float(best_prior["score"]),
            "config": best_prior["config"],
            "drift": primary_drift,
        }
        conservative_selection[target] = {
            "source": conservative_source,
            "weight": float(conservative_weight),
            "score": float(conservative_score),
            "prior_score": float(best_prior["score"]),
            "config": best_prior["config"],
            "drift": conservative_drift,
        }
        target_details[target] = {
            "anchor_score": anchor_scores[target],
            "prior_score": best_prior["score"],
            "primary_weight": primary_weight,
            "primary_score": primary_score,
            "conservative_weight": conservative_weight,
            "conservative_score": conservative_score,
            "config": best_prior["config"],
        }
        print(
            f"{target}: anchor={anchor_scores[target]:.6f} prior={best_prior['score']:.6f} "
            f"w={primary_weight:.2f} primary={primary_score:.6f} "
            f"cw={conservative_weight:.2f} conservative={conservative_score:.6f}"
        )

    primary_scores = _score_targets(labels, primary_oof)
    conservative_scores = _score_targets(labels, conservative_oof)

    primary_path = READY_DIR / "next_target_history.csv"
    conservative_path = READY_DIR / "next_target_history_conservative.csv"
    _make_submission(primary_path, primary_test, overwrite=args.overwrite)
    _make_submission(conservative_path, conservative_test, overwrite=args.overwrite)
    _write_prediction_artifacts(RUN_NAME, train_keys, labels, primary_oof, primary_test, primary_scores, primary_selection)
    _write_prediction_artifacts(CONSERVATIVE_RUN_NAME, train_keys, labels, conservative_oof, conservative_test, conservative_scores, conservative_selection)
    _write_stability(primary_selection, prior_rows, filename="stability_target_history_prior_v1.csv")
    _write_stability(conservative_selection, prior_rows, filename="stability_target_history_prior_v1_conservative.csv")
    _update_public_scores()
    _write_candidate_scores(primary_scores, conservative_scores)
    _append_experiment(EXPERIMENT_NAME, primary_scores, primary_path, primary_selection, "Fast fold-safe target-history prior blend on stable_tuned anchor.")
    _append_experiment(f"{EXPERIMENT_NAME}_conservative", conservative_scores, conservative_path, conservative_selection, "Conservative target-history prior blend.")
    _write_readme(primary_scores, conservative_scores)
    _write_report(primary_selection, conservative_selection, primary_scores, conservative_scores)
    write_json(
        FEATURES_DIR / "target_history_prior_v1_selection.json",
        {
            "anchor_scores": anchor_scores,
            "primary_scores": primary_scores,
            "conservative_scores": conservative_scores,
            "primary_selection": primary_selection,
            "conservative_selection": conservative_selection,
            "target_details": target_details,
        },
    )

    print("\n=== target_history_prior_v1 ===")
    print(f"primary: {primary_path} OOF={primary_scores['mean']:.6f}")
    print(f"conservative: {conservative_path} OOF={conservative_scores['mean']:.6f}")


if __name__ == "__main__":
    main()

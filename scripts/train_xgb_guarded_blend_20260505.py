#!/usr/bin/env python3
"""Train XGBoost and create guarded blends against the current public best."""

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
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_SUBMISSIONS_DIR, ROOT, ensure_runtime_dirs
from etri_human_challenge.public_lgb import (
    get_public_lgb_feature_columns,
    load_public_lgb_feature_table,
    resolve_target_feature_views,
)
from etri_human_challenge.utils import binary_log_loss, write_json, write_markdown


EXPERIMENT_NAME = "xgb_guarded_blend_20260505"
XGB_RUN = "xgb_targetwise_histmix_guarded_20260505"
CURRENT_RUN = "public_lgb_targetwise_toward57_s4b650_20260504"
CURRENT_BEST_FILE = "lgb_temporal_s4b650.csv"
CURRENT_BEST_PUBLIC = 0.5829008297

READY_DIR = ROOT / "submissions" / "ready"
BLOCKED_ARCHIVE_DIR = ROOT / "submissions" / "archive" / "2026-05-05_xgb_guard_blocked"
LOG_DIR = ROOT / "logs"
CLIP_MIN = 0.02
CLIP_MAX = 0.98
SEEDS = [42, 1234]
N_FOLDS = 5
MIN_TARGET_GAIN = 2.5e-5
FREEZE_TARGETS = {"S4"}
MAX_DRIFT = {
    "guard003": 0.006,
    "guard005": 0.010,
}
CANDIDATES = [
    {
        "name": "guard003",
        "file": "submit_xgb_guard003_s4b650.csv",
        "weights": [0.00, 0.01, 0.02, 0.03],
        "decision": "First XGB upload if selected weights are non-zero: max 3% blend, S4 frozen.",
    },
    {
        "name": "guard005",
        "file": "submit_xgb_guard005_s4b650.csv",
        "weights": [0.00, 0.01, 0.02, 0.03, 0.04, 0.05],
        "decision": "Only if guard003 improves public: max 5% blend, S4 frozen.",
    },
]


BASE_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_estimators": 450,
    "learning_rate": 0.025,
    "max_depth": 2,
    "min_child_weight": 8.0,
    "subsample": 0.78,
    "colsample_bytree": 0.62,
    "reg_alpha": 0.7,
    "reg_lambda": 6.0,
    "gamma": 0.05,
    "max_bin": 128,
    "n_jobs": -1,
    "verbosity": 0,
}

TARGET_OVERRIDES: dict[str, dict[str, Any]] = {
    "Q1": {"max_depth": 2, "min_child_weight": 10.0, "reg_lambda": 8.0},
    "Q2": {"max_depth": 2, "min_child_weight": 10.0, "reg_lambda": 8.0},
    "Q3": {"max_depth": 2, "min_child_weight": 10.0, "reg_lambda": 8.0},
    "S1": {"max_depth": 3, "min_child_weight": 6.0, "reg_lambda": 5.0},
    "S2": {"max_depth": 2, "min_child_weight": 8.0, "reg_lambda": 6.0},
    "S3": {"max_depth": 2, "min_child_weight": 8.0, "reg_lambda": 6.0},
    "S4": {"max_depth": 2, "min_child_weight": 12.0, "reg_lambda": 10.0},
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
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _load_current() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{CURRENT_RUN}.parquet")
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(MODELS_DIR / f"test_predictions_{CURRENT_RUN}.csv")[TARGET_COLUMNS].astype(float)
    return raw[KEY_COLUMNS].copy(), labels, oof[TARGET_COLUMNS], test[TARGET_COLUMNS]


def _xgb_params(target: str, seed: int) -> dict[str, Any]:
    return {**BASE_XGB_PARAMS, **TARGET_OVERRIDES.get(target, {}), "random_state": int(seed)}


def _train_xgb() -> dict[str, Any]:
    ensure_runtime_dirs()
    frame = load_public_lgb_feature_table(rebuild=False)
    train = frame[frame["split"] == "train"].reset_index(drop=True).copy()
    test = frame[frame["split"] == "test"].reset_index(drop=True).copy()
    target_views = resolve_target_feature_views(default_feature_view="public_core", preset_name="histmix_guarded_v1")
    distinct_views = sorted(set(target_views.values()))
    feature_cols_by_view = {
        view: get_public_lgb_feature_columns(frame, feature_view=view)
        for view in distinct_views
    }

    oof_preds = np.zeros((len(train), len(TARGET_COLUMNS)), dtype=float)
    test_preds = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
    seed_scores: dict[str, list[dict[str, float]]] = {}
    feature_counts_by_target: dict[str, int] = {}

    for target_idx, target in enumerate(TARGET_COLUMNS):
        view = target_views[target]
        feature_cols = feature_cols_by_view[view]
        feature_counts_by_target[target] = len(feature_cols)
        X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).astype(float)
        X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).astype(float)
        y = train[target].astype(int).to_numpy()

        target_oof = np.zeros(len(train), dtype=float)
        target_test = np.zeros(len(test), dtype=float)
        target_seed_scores: list[dict[str, float]] = []

        for seed in SEEDS:
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed))
            seed_oof = np.zeros(len(train), dtype=float)
            seed_test = np.zeros(len(test), dtype=float)
            splits = list(splitter.split(X_train, y))
            for train_idx, valid_idx in splits:
                model = XGBClassifier(**_xgb_params(target, seed))
                model.fit(X_train.iloc[train_idx], y[train_idx])
                seed_oof[valid_idx] = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
                seed_test += model.predict_proba(X_test)[:, 1] / len(splits)
            target_seed_scores.append({"seed": float(seed), "log_loss": float(log_loss(y, np.clip(seed_oof, 1e-6, 1.0 - 1e-6)))})
            target_oof += seed_oof
            target_test += seed_test

        oof_preds[:, target_idx] = target_oof / len(SEEDS)
        test_preds[:, target_idx] = target_test / len(SEEDS)
        seed_scores[target] = target_seed_scores

    oof = pd.DataFrame(oof_preds, columns=TARGET_COLUMNS).clip(CLIP_MIN, CLIP_MAX)
    test_pred = pd.DataFrame(test_preds, columns=TARGET_COLUMNS).clip(CLIP_MIN, CLIP_MAX)
    scores = _score_targets(train[TARGET_COLUMNS], oof)

    oof_export = train[KEY_COLUMNS + TARGET_COLUMNS].copy()
    oof_export["split_scheme"] = "public_stratified_xgb"
    oof_export["model_family"] = XGB_RUN
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_xgb"] = oof[target].to_numpy(float)
    oof_path = OOF_DIR / f"oof_predictions_{XGB_RUN}.parquet"
    oof_export.to_parquet(oof_path, index=False)
    test_path = MODELS_DIR / f"test_predictions_{XGB_RUN}.csv"
    test_pred.to_csv(test_path, index=False)

    summary = {
        "run_name": XGB_RUN,
        "model": "xgboost.XGBClassifier",
        "target_feature_views": target_views,
        "n_features_by_target": feature_counts_by_target,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "scores": scores,
        "seed_scores": seed_scores,
        "params": BASE_XGB_PARAMS,
        "target_overrides": TARGET_OVERRIDES,
    }
    write_json(FEATURES_DIR / f"{XGB_RUN}_summary.json", summary)
    write_markdown(
        REPORT_SUBMISSIONS_DIR / f"{XGB_RUN}.md",
        "\n".join(
            [
                f"# XGBoost Targetwise Report ({XGB_RUN})",
                "",
                f"- OOF mean: `{scores['mean']:.6f}`",
                f"- Folds: `{N_FOLDS}`",
                f"- Seeds: `{SEEDS}`",
                "",
                "## Target Scores",
                "",
                *[f"- `{target}`: `{scores[target]:.6f}`" for target in TARGET_COLUMNS],
            ]
        ),
    )
    return {"oof": oof, "test": test_pred, "scores": scores, "summary": summary}


def _choose_weights(
    labels: pd.DataFrame,
    current_oof: pd.DataFrame,
    xgb_oof: pd.DataFrame,
    candidate_weights: list[float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    weights: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        current_score = binary_log_loss(labels[target].to_numpy(float), current_oof[target].to_numpy(float))
        if target in FREEZE_TARGETS:
            weights[target] = 0.0
            diagnostics[target] = {
                "current": current_score,
                "xgb": binary_log_loss(labels[target].to_numpy(float), xgb_oof[target].to_numpy(float)),
                "best_blend": current_score,
                "gain": 0.0,
                "selected_weight": 0.0,
            }
            continue

        best_weight = 0.0
        best_score = current_score
        for weight in candidate_weights:
            blended = np.clip((1.0 - weight) * current_oof[target].to_numpy(float) + weight * xgb_oof[target].to_numpy(float), CLIP_MIN, CLIP_MAX)
            score = binary_log_loss(labels[target].to_numpy(float), blended)
            if score < best_score:
                best_score = score
                best_weight = float(weight)
        gain = current_score - best_score
        if gain < MIN_TARGET_GAIN:
            best_weight = 0.0
            best_score = current_score
            gain = 0.0
        weights[target] = best_weight
        diagnostics[target] = {
            "current": current_score,
            "xgb": binary_log_loss(labels[target].to_numpy(float), xgb_oof[target].to_numpy(float)),
            "best_blend": best_score,
            "gain": gain,
            "selected_weight": best_weight,
        }
    return weights, diagnostics


def _apply_weights(current: pd.DataFrame, xgb: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    output = current.copy()
    for target in TARGET_COLUMNS:
        weight = float(weights[target])
        output[target] = np.clip((1.0 - weight) * current[target].to_numpy(float) + weight * xgb[target].to_numpy(float), CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _write_fixed_submission(path: Path, predictions: pd.DataFrame) -> None:
    template = pd.read_csv(ROOT / "data" / "ch2026_submission_sample.csv", dtype=str)
    output = template[KEY_COLUMNS].copy()
    values = predictions[TARGET_COLUMNS].astype(float).clip(1e-6, 1.0 - 1e-6)
    if values.shape != (250, 7):
        raise ValueError(f"Invalid prediction shape: {values.shape}")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    for target in TARGET_COLUMNS:
        output[target] = values[target].map(lambda value: f"{value:.10f}")
    output = output[template.columns]
    if output.shape != template.shape:
        raise ValueError(f"Invalid submission shape: {output.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, lineterminator="\n")


def _append_experiment(
    name: str,
    scores: dict[str, float],
    submission: Path,
    weights: dict[str, float],
    diagnostics: dict[str, dict[str, float]],
    notes: str,
) -> None:
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
            "validation_scheme": "xgb_guarded_blend",
            "seeds": ",".join(str(seed) for seed in SEEDS),
            "total_oof_logloss": scores["mean"],
            "target_logloss_Q1": scores["Q1"],
            "target_logloss_Q2": scores["Q2"],
            "target_logloss_Q3": scores["Q3"],
            "target_logloss_S1": scores["S1"],
            "target_logloss_S2": scores["S2"],
            "target_logloss_S3": scores["S3"],
            "target_logloss_S4": scores["S4"],
            "submission_file": str(submission.relative_to(ROOT)),
            "feature_view_by_target": json.dumps({"xgb_run": XGB_RUN, "current_run": CURRENT_RUN, "weights": weights, "diagnostics": diagnostics}, sort_keys=True),
            "notes": notes,
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(current_scores: dict[str, float], results: list[dict[str, Any]]) -> None:
    existing = _read_csv_rows(LOG_DIR / "candidate_scores.csv")
    keep = [
        row for row in existing
        if row.get("candidate") not in {result["file"] for result in results}
    ]
    if not keep:
        keep = [
            {
                "rank": 0,
                "candidate": CURRENT_BEST_FILE,
                "oof_mean": current_scores["mean"],
                "public_score": f"{CURRENT_BEST_PUBLIC:.10f}",
                "submission_file": f"submissions/ready/{CURRENT_BEST_FILE}",
                "notes": "Current public best.",
            }
        ]
    rows = keep[:1]
    for idx, result in enumerate(results, start=1):
        rows.append(
            {
                "rank": idx,
                "candidate": result["file"],
                "oof_mean": result["scores"]["mean"],
                "public_score": "",
                "submission_file": str(result["path"].relative_to(ROOT)),
                "notes": result["decision"] if result["guard_passed"] else "Blocked: XGB selected no non-zero guarded weights; do not upload.",
            }
        )
    for row in keep[1:]:
        row = dict(row)
        row["rank"] = len(rows)
        rows.append(row)
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_report(
    current_scores: dict[str, float],
    xgb_scores: dict[str, float],
    results: list[dict[str, Any]],
) -> None:
    lines = [
        "# XGB Guarded Blend 2026-05-05",
        "",
        f"- Current best: `{CURRENT_BEST_FILE}` public `{CURRENT_BEST_PUBLIC:.10f}`",
        f"- XGB run: `{XGB_RUN}`",
        "- Method: train target-wise XGBoost, then blend into current best only when OOF blend improves the target.",
        "- S4 is frozen to avoid damaging the public-proven S4 beta axis.",
        "",
        "## Self-Critique",
        "",
        "- XGBoost adds model diversity, but the dataset has only 450 train rows and many features, so raw XGB can overfit.",
        "- A direct XGB submission is not acceptable under the 'do not drop' requirement.",
        "- The guarded blend can still drop public if OOF does not match public, so the first file is capped at 3%.",
        "",
        "## OOF Scores",
        "",
        "| Target | Current | XGB raw |",
        "|---|---:|---:|",
    ]
    for target in TARGET_COLUMNS:
        lines.append(f"| {target} | `{current_scores[target]:.6f}` | `{xgb_scores[target]:.6f}` |")
    lines.extend(["", f"- Current mean: `{current_scores['mean']:.6f}`", f"- XGB raw mean: `{xgb_scores['mean']:.6f}`", "", "## Guarded Candidates", "", "| File | OOF mean | Selected weights | Decision |", "|---|---:|---|---|"])
    for result in results:
        weights = ", ".join(f"{target}={weight:.2f}" for target, weight in result["weights"].items() if weight > 0)
        if not weights:
            weights = "none"
        lines.append(f"| `{result['file']}` | `{result['scores']['mean']:.6f}` | `{weights}` | {result['decision']} |")
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Upload `submit_xgb_guard003_s4b650.csv` only if it selected at least one non-zero weight.",
            "- If no weight is selected, XGB did not pass the guard and should not be uploaded.",
            "- Upload `submit_xgb_guard005_s4b650.csv` only after the 3% file improves public.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "xgb_guarded_blend_20260505.md", "\n".join(lines))


def _update_readme(results: list[dict[str, Any]]) -> None:
    path = ROOT / "submissions" / "README.md"
    text = path.read_text() if path.exists() else "# Submissions\n"
    start = "<!-- xgb_guarded_blend_20260505:start -->"
    end = "<!-- xgb_guarded_blend_20260505:end -->"
    first = results[0]
    selected = [target for target, weight in first["weights"].items() if weight > 0]
    line = (
        f"XGB guard 통과 타깃: `{', '.join(selected)}`."
        if selected else
        "XGB guard 통과 타깃이 없으므로 제출하지 않습니다."
    )
    block = "\n".join(
        [
            start,
            "",
            "## XGB Guarded Blend",
            "",
            "- XGBoost는 단독 제출하지 않고 current best에 소량만 섞습니다.",
            "- S4는 고정합니다.",
            f"- 1순위 XGB 후보 위치: `{first['path'].relative_to(ROOT)}` OOF `{first['scores']['mean']:.6f}`.",
            f"- {line}",
            "- 5% 후보는 3% 후보가 public에서 개선될 때만 사용합니다.",
            "",
            end,
        ]
    )
    if start in text and end in text:
        before = text.split(start)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        text = f"{before}\n\n{block}\n\n{after}"
    else:
        text = f"{text.rstrip()}\n\n{block}\n"
    write_markdown(path, text)


def main() -> None:
    ensure_runtime_dirs()
    keys, labels, current_oof, current_test = _load_current()
    xgb = _train_xgb()
    xgb_oof = xgb["oof"]
    xgb_test = xgb["test"]
    current_scores = _score_targets(labels, current_oof)
    xgb_scores = xgb["scores"]

    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        weights, diagnostics = _choose_weights(labels, current_oof, xgb_oof, candidate["weights"])
        blend_oof = _apply_weights(current_oof, xgb_oof, weights)
        blend_test = _apply_weights(current_test, xgb_test, weights)
        scores = _score_targets(labels, blend_oof)
        max_drift = MAX_DRIFT[candidate["name"]]
        for target in TARGET_COLUMNS:
            drift = float(np.mean(np.abs(blend_test[target].to_numpy(float) - current_test[target].to_numpy(float))))
            if drift > max_drift:
                weights[target] = 0.0
        blend_oof = _apply_weights(current_oof, xgb_oof, weights)
        blend_test = _apply_weights(current_test, xgb_test, weights)
        scores = _score_targets(labels, blend_oof)

        guard_passed = any(float(weight) > 0.0 for weight in weights.values())
        output_path = (READY_DIR if guard_passed else BLOCKED_ARCHIVE_DIR) / candidate["file"]
        _write_fixed_submission(output_path, blend_test)
        name = f"{EXPERIMENT_NAME}_{candidate['name']}"
        _append_experiment(name, scores, output_path, weights, diagnostics, candidate["decision"])
        results.append({**candidate, "path": output_path, "weights": weights, "diagnostics": diagnostics, "scores": scores, "guard_passed": guard_passed})

    _write_candidate_scores(current_scores, results)
    _write_report(current_scores, xgb_scores, results)
    _update_readme(results)

    print(
        json.dumps(
            {
                "status": "ok",
                "xgb_oof": xgb_scores,
                "current_oof": current_scores,
                "results": [
                    {
                        "file": result["file"],
                        "oof": result["scores"]["mean"],
                        "weights": result["weights"],
                    }
                    for result in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

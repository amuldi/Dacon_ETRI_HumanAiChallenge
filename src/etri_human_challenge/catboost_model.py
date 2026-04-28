"""CatBoost model pipeline for ETRI challenge.

Why CatBoost over LightGBM for this dataset?
---------------------------------------------
1. Ordered boosting: CatBoost's default mode trains leaf splits using only
   the data that appeared *before* the current example — this is naturally
   aligned with our temporal/subject structure and reduces overfitting on
   small datasets (450 rows).
2. Symmetric (oblivious) trees: each tree level uses the same split across
   all branches, producing a lower-variance estimator — good when N is tiny.
3. No need for explicit early stopping tuning: CatBoost's built-in Bayesian
   bootstrap provides implicit regularization.

Per-target hyperparameter rationale
-------------------------------------
Based on OOF scores from the existing LGB pipeline:
  S1 (0.524) — easy, well-learnt → keep capacity, small depth
  S2, S3     — medium            → standard params
  Q1         — hard, binary      → increase regularization
  Q2, Q3     — hard, hist-aided  → aggressive regularization
  S4         — hardest (0.611)   → maximum regularization

CV scheme
---------
Uses group_time forward-chaining splits (3 folds) by default to match
the temporal validation methodology.  Can also use StratifiedKFold via
``cv_scheme="stratified"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    CatBoostClassifier = None  # type: ignore[assignment,misc]
    Pool = None                # type: ignore[assignment,misc]
    HAS_CATBOOST = False

from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .group_time_cv import group_time_split_iter
from .paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_OOF_DIR, ensure_runtime_dirs
from .utils import clip_probabilities, multi_target_log_loss, write_json, write_markdown


# ─────────────────────────────────────────────
# Default hyperparameters
# ─────────────────────────────────────────────

_CATBOOST_BASE = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5,
    "verbose": 0,
    "allow_writing_files": False,
    "task_type": "CPU",
    "thread_count": -1,
    "od_type": "Iter",
    "od_wait": 100,
}

# Per-target overrides — tuned for each target's difficulty profile
CATBOOST_TARGET_PARAMS: dict[str, dict[str, Any]] = {
    "Q1": {
        **_CATBOOST_BASE,
        "iterations": 1500,
        "learning_rate": 0.02,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "border_count": 128,
    },
    "Q2": {
        **_CATBOOST_BASE,
        "iterations": 1500,
        "learning_rate": 0.02,
        "depth": 4,
        "l2_leaf_reg": 8.0,
        "border_count": 64,
    },
    "Q3": {
        **_CATBOOST_BASE,
        "iterations": 1500,
        "learning_rate": 0.02,
        "depth": 4,
        "l2_leaf_reg": 8.0,
        "border_count": 64,
    },
    "S1": {
        **_CATBOOST_BASE,
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "border_count": 128,
    },
    "S2": {
        **_CATBOOST_BASE,
        "iterations": 1200,
        "learning_rate": 0.025,
        "depth": 5,
        "l2_leaf_reg": 4.0,
        "border_count": 128,
    },
    "S3": {
        **_CATBOOST_BASE,
        "iterations": 1200,
        "learning_rate": 0.025,
        "depth": 5,
        "l2_leaf_reg": 4.0,
        "border_count": 128,
    },
    "S4": {
        **_CATBOOST_BASE,
        "iterations": 2000,
        "learning_rate": 0.015,
        "depth": 4,
        "l2_leaf_reg": 10.0,
        "border_count": 64,
    },
}

CATBOOST_DEFAULT_SEEDS = [42, 1234, 9999, 7, 314, 2025, 777, 555]


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def _make_cv_splits(
    X_train: pd.DataFrame,
    y: np.ndarray,
    *,
    cv_scheme: str,
    n_folds: int,
    seed: int,
    full_frame: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if cv_scheme == "group_time":
        if full_frame is None:
            raise ValueError("full_frame required for group_time CV")
        return list(group_time_split_iter(full_frame, n_folds=n_folds))
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(skf.split(X_train, y))


def train_catboost_targetwise(
    *,
    frame: pd.DataFrame,
    feature_cols: list[str],
    run_name: str = "catboost_v1",
    n_folds: int = 3,
    seeds: list[int] | None = None,
    cv_scheme: str = "group_time",
    target_params: dict[str, dict[str, Any]] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """
    Train per-target CatBoost models with multi-seed averaging.

    Parameters
    ----------
    frame:
        Full feature table (train + test rows with ``split`` column).
    feature_cols:
        List of numeric feature column names.
    run_name:
        Unique identifier for this run.
    n_folds:
        Number of CV folds (3 for group_time, 5 for stratified).
    seeds:
        Random seeds to average over.
    cv_scheme:
        ``"group_time"`` (forward-chaining per subject) or ``"stratified"``.
    target_params:
        Per-target CatBoost params dict.  Defaults to ``CATBOOST_TARGET_PARAMS``.
    persist:
        Save OOF + test predictions to artifacts/.
    """
    if not HAS_CATBOOST:
        raise RuntimeError("catboost is not installed — add it to requirements.txt")

    ensure_runtime_dirs()
    tparams = target_params or CATBOOST_TARGET_PARAMS
    seed_list = seeds or CATBOOST_DEFAULT_SEEDS

    train = frame[frame["split"] == "train"].reset_index(drop=True).copy()
    test  = frame[frame["split"] == "test"].reset_index(drop=True).copy()

    X_train = train[feature_cols].copy()
    X_test  = test[feature_cols].copy()

    # Replace inf/large values — CatBoost is sensitive to these
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    oof_preds  = np.zeros((len(train), len(TARGET_COLUMNS)), dtype=float)
    test_preds = np.zeros((len(test),  len(TARGET_COLUMNS)), dtype=float)
    seed_scores: dict[str, list[dict[str, Any]]] = {}

    for target_idx, target in enumerate(TARGET_COLUMNS):
        y = train[target].astype(int).to_numpy()
        params = {**tparams.get(target, tparams["Q1"])}

        target_oof  = np.zeros(len(train), dtype=float)
        target_test = np.zeros(len(test),  dtype=float)
        target_seed_scores: list[dict[str, Any]] = []

        for seed in seed_list:
            params["random_seed"] = int(seed)
            splits = _make_cv_splits(
                X_train, y,
                cv_scheme=cv_scheme,
                n_folds=n_folds,
                seed=seed,
                full_frame=train if cv_scheme == "group_time" else None,
            )

            seed_oof  = np.zeros(len(train), dtype=float)
            seed_test = np.zeros(len(test),  dtype=float)

            for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                model = CatBoostClassifier(**params)
                model.fit(
                    X_train.iloc[tr_idx].to_numpy(),
                    y[tr_idx],
                    eval_set=(X_train.iloc[va_idx].to_numpy(), y[va_idx]),
                    verbose=False,
                )
                seed_oof[va_idx] = model.predict_proba(X_train.iloc[va_idx].to_numpy())[:, 1]
                seed_test += model.predict_proba(X_test.to_numpy())[:, 1] / len(splits)

            oof_ll = float(log_loss(y, clip_probabilities(seed_oof)))
            target_seed_scores.append({"seed": int(seed), "log_loss": oof_ll})
            target_oof  += seed_oof
            target_test += seed_test

        oof_preds[:, target_idx]  = target_oof  / len(seed_list)
        test_preds[:, target_idx] = target_test / len(seed_list)
        seed_scores[target] = target_seed_scores

    oof_frame = pd.DataFrame(oof_preds, columns=TARGET_COLUMNS)
    score_payload = multi_target_log_loss(train[TARGET_COLUMNS], oof_frame, TARGET_COLUMNS)

    result: dict[str, Any] = {
        "run_name":   run_name,
        "n_features": len(feature_cols),
        "n_folds":    n_folds,
        "cv_scheme":  cv_scheme,
        "seeds":      seed_list,
        "scores":     score_payload,
        "seed_scores": seed_scores,
    }

    if not persist:
        result["train_predictions"] = oof_frame
        result["test_predictions"]  = pd.DataFrame(test_preds, columns=TARGET_COLUMNS)
        return result

    # ── Persist ──────────────────────────────────
    test_pred_path = MODELS_DIR / f"test_predictions_{run_name}.csv"
    pd.DataFrame(test_preds, columns=TARGET_COLUMNS).to_csv(test_pred_path, index=False)

    oof_export = train[KEY_COLUMNS + TARGET_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_catboost"] = oof_frame[target].values
    oof_path = OOF_DIR / f"oof_predictions_{run_name}.parquet"
    oof_export.to_parquet(oof_path, index=False)

    summary_path = FEATURES_DIR / f"{run_name}_summary.json"
    write_json(summary_path, {
        "run_name":   run_name,
        "n_features": len(feature_cols),
        "n_folds":    n_folds,
        "cv_scheme":  cv_scheme,
        "seeds":      seed_list,
        "scores":     score_payload,
    })

    report_lines = [
        f"# CatBoost Report ({run_name})",
        "",
        f"- Train rows : {len(train)}",
        f"- Test rows  : {len(test)}",
        f"- Features   : {len(feature_cols)}",
        f"- Folds      : {n_folds} ({cv_scheme})",
        f"- Seeds      : {seed_list}",
        f"- Mean OOF log-loss: {score_payload['mean']:.6f}",
        "",
        "## Target Scores",
        "",
    ]
    for t in TARGET_COLUMNS:
        report_lines.append(f"- `{t}`: {score_payload[t]:.6f}")
    write_markdown(REPORT_OOF_DIR / f"{run_name}.md", "\n".join(report_lines))

    result["artifacts"] = {
        "test_prediction_path": str(test_pred_path),
        "oof_path":             str(oof_path),
        "summary_path":         str(summary_path),
    }
    return result

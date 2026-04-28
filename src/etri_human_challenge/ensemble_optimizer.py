"""Per-target ensemble weight optimizer.

Strategy
--------
Given K sets of OOF predictions (from LGB core, LGB hist365, LGB hist411,
CatBoost, etc.) we search for per-target blend weights that minimize OOF
binary log-loss.

The optimization is run with multiple random restarts to avoid local minima.
Weights are constrained to sum to 1 and each weight ∈ [0, 1].

Why per-target optimization beats uniform blending
---------------------------------------------------
Different targets are explained by different feature sets (hence histmix_guarded
beats public_core).  Extending this: some targets may benefit more from CatBoost
(ordered boosting handles temporal patterns better) while others benefit from
more LGB features.  Optimizing weights in OOF space finds this automatically
without manual tuning.

Isotonic calibration
--------------------
After blending, we apply isotonic regression calibration per target.  This
corrects for probability miscalibration (predictions too extreme or too
conservative) using the held-out OOF examples as calibration data.

Usage
-----
    from etri_human_challenge.ensemble_optimizer import (
        optimize_ensemble_weights,
        blend_predictions,
        IsotonicEnsembleCalibrator,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize as _scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sklearn.isotonic import IsotonicRegression

from .constants import TARGET_COLUMNS
from .utils import binary_log_loss, clip_probabilities


# ─────────────────────────────────────────────
# Weight optimization
# ─────────────────────────────────────────────

def _neg_log_loss_objective(
    weights: np.ndarray,
    oof_list: list[np.ndarray],
    y: np.ndarray,
) -> float:
    """Objective: -log_loss of weighted blend (to minimize)."""
    w = np.clip(weights, 0.0, 1.0)
    total = w.sum()
    if total < 1e-9:
        return 1e6
    w = w / total
    blended = sum(wi * oi for wi, oi in zip(w, oof_list))
    return binary_log_loss(y, clip_probabilities(blended))


def _optimize_weights_scipy(
    oof_list: list[np.ndarray],
    y: np.ndarray,
    n_restarts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    n = len(oof_list)
    best_w = np.ones(n) / n
    best_loss = _neg_log_loss_objective(best_w, oof_list, y)

    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n))
        result = _scipy_minimize(
            _neg_log_loss_objective,
            w0,
            args=(oof_list, y),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: float(w.sum()) - 1.0},
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        if result.success and result.fun < best_loss:
            best_loss = float(result.fun)
            w = np.clip(result.x, 0.0, 1.0)
            best_w = w / w.sum()

    return best_w, best_loss


def _optimize_weights_grid(
    oof_list: list[np.ndarray],
    y: np.ndarray,
    n_restarts: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Fallback: random restart hill-climbing (no scipy needed)."""
    n = len(oof_list)
    best_w = np.ones(n) / n
    best_loss = _neg_log_loss_objective(best_w, oof_list, y)

    # Grid search when n <= 3
    if n <= 3:
        steps = np.linspace(0.0, 1.0, 21)
        if n == 2:
            for w0 in steps:
                w = np.array([w0, 1.0 - w0])
                loss = _neg_log_loss_objective(w, oof_list, y)
                if loss < best_loss:
                    best_loss = loss
                    best_w = w
        elif n == 3:
            for w0 in steps:
                for w1 in steps:
                    w2 = 1.0 - w0 - w1
                    if w2 < 0:
                        continue
                    w = np.array([w0, w1, w2])
                    loss = _neg_log_loss_objective(w, oof_list, y)
                    if loss < best_loss:
                        best_loss = loss
                        best_w = w

    # Random restarts on top
    for _ in range(n_restarts):
        w = rng.dirichlet(np.ones(n))
        loss = _neg_log_loss_objective(w, oof_list, y)
        if loss < best_loss:
            best_loss = loss
            best_w = w.copy()

    return best_w, best_loss


def optimize_ensemble_weights(
    oof_predictions: list[pd.DataFrame],
    y_true: pd.DataFrame,
    *,
    n_restarts: int = 20,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """
    Find per-target optimal blend weights across K OOF DataFrames.

    Parameters
    ----------
    oof_predictions:
        List of K DataFrames, each with TARGET_COLUMNS.  All must have the
        same row order (aligned to the training rows).
    y_true:
        Ground-truth DataFrame with TARGET_COLUMNS.
    n_restarts:
        Number of random restarts for weight search.

    Returns
    -------
    dict mapping target -> list of K floats (normalized weights).
    """
    rng = np.random.default_rng(random_state)
    optimal: dict[str, list[float]] = {}

    for target in TARGET_COLUMNS:
        oof_list = [oof[target].to_numpy(dtype=float) for oof in oof_predictions]
        y = y_true[target].to_numpy(dtype=float)

        if HAS_SCIPY:
            w, _ = _optimize_weights_scipy(oof_list, y, n_restarts, rng)
        else:
            w, _ = _optimize_weights_grid(oof_list, y, n_restarts, rng)

        optimal[target] = w.tolist()

    return optimal


def blend_predictions(
    pred_frames: list[pd.DataFrame],
    weights: dict[str, list[float]],
) -> pd.DataFrame:
    """
    Blend K prediction DataFrames using per-target weights.

    Parameters
    ----------
    pred_frames:
        List of K DataFrames (same schema, TARGET_COLUMNS present).
    weights:
        Output of ``optimize_ensemble_weights`` or manually specified.
    """
    out: dict[str, np.ndarray] = {}
    for target in TARGET_COLUMNS:
        w = np.array(weights[target], dtype=float)
        blended = sum(
            w[i] * pred_frames[i][target].to_numpy(dtype=float)
            for i in range(len(pred_frames))
        )
        out[target] = blended
    return pd.DataFrame(out)


# ─────────────────────────────────────────────
# Isotonic regression calibration
# ─────────────────────────────────────────────

@dataclass
class IsotonicEnsembleCalibrator:
    """
    Fit isotonic regression on blended OOF predictions, then apply to test.

    Isotonic regression is a monotone function that maps raw predicted
    probabilities to calibrated probabilities.  It perfectly fits any
    monotone calibration curve and generalizes well when:
      - sample size ≥ 100 per target (we have ~450, so fine)
      - the raw predictor is already rank-ordered correctly

    CAUTION: Isotonic can overfit if applied to in-fold predictions.
    Only apply to OOF predictions (one global fit per target).
    """
    calibrators: dict[str, IsotonicRegression] = field(default_factory=dict)

    def fit(self, y_prob: pd.DataFrame, y_true: pd.DataFrame) -> "IsotonicEnsembleCalibrator":
        for target in TARGET_COLUMNS:
            p = clip_probabilities(y_prob[target].to_numpy(dtype=float))
            y = y_true[target].to_numpy(dtype=float)
            valid = np.isfinite(p) & np.isfinite(y)
            if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
                self.calibrators[target] = None  # type: ignore[assignment]
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p[valid], y[valid])
            self.calibrators[target] = iso
        return self

    def predict(self, y_prob: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        for target in TARGET_COLUMNS:
            p = clip_probabilities(y_prob[target].to_numpy(dtype=float))
            cal = self.calibrators.get(target)
            if cal is None:
                out[target] = p
            else:
                out[target] = clip_probabilities(cal.predict(p))
        return pd.DataFrame(out, index=y_prob.index)


# ─────────────────────────────────────────────
# Importance-stability feature selector
# ─────────────────────────────────────────────

def select_stable_features(
    importance_matrix: pd.DataFrame,
    *,
    top_k: int | None = None,
    stability_threshold: float = 0.0,
) -> list[str]:
    """
    Select features with stable importance across seeds/folds.

    Parameters
    ----------
    importance_matrix:
        DataFrame with features as index, seeds/folds as columns.
        Values are raw importance scores.
    top_k:
        Keep top-K features by mean importance.  None = keep all.
    stability_threshold:
        Discard features whose std/mean importance > threshold
        (coefficient of variation).  0.0 = no stability filter.

    Returns
    -------
    Sorted list of selected feature names.
    """
    mean_imp = importance_matrix.mean(axis=1)
    std_imp  = importance_matrix.std(axis=1)
    cv_imp   = std_imp / (mean_imp.abs() + 1e-9)

    selected = importance_matrix.index.tolist()

    if stability_threshold > 0.0:
        stable_mask = (cv_imp <= stability_threshold) | (mean_imp > mean_imp.quantile(0.5))
        selected = [f for f in selected if stable_mask.get(f, True)]

    mean_imp_sel = mean_imp[selected].sort_values(ascending=False)

    if top_k is not None:
        selected = mean_imp_sel.index.tolist()[:top_k]
    else:
        selected = mean_imp_sel.index.tolist()

    return selected


# ─────────────────────────────────────────────
# Summary reporting
# ─────────────────────────────────────────────

def report_ensemble_weights(
    weights: dict[str, list[float]],
    model_names: list[str],
) -> str:
    """Return a markdown table of per-target weights."""
    lines = ["# Ensemble Weights", "", "| Target | " + " | ".join(model_names) + " |"]
    lines.append("|--------|" + "|".join(["-------"] * len(model_names)) + "|")
    for target in TARGET_COLUMNS:
        w = weights.get(target, [1.0 / len(model_names)] * len(model_names))
        cells = " | ".join(f"{wi:.3f}" for wi in w)
        lines.append(f"| {target} | {cells} |")
    return "\n".join(lines)


def evaluate_oof_blend(
    oof_list: list[pd.DataFrame],
    y_true: pd.DataFrame,
    weights: dict[str, list[float]],
) -> dict[str, float]:
    """Return per-target log-loss of weighted OOF blend."""
    from .utils import multi_target_log_loss
    blended = blend_predictions(oof_list, weights)
    return multi_target_log_loss(y_true, blended, TARGET_COLUMNS)

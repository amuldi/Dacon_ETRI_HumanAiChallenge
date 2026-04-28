#!/usr/bin/env python3
"""Advanced v2 training pipeline.

What this script does (in order)
---------------------------------
1.  Build advanced feature table
    - Loads existing public_lgb_feature_table.parquet
    - Adds: 2nd-order temporal (accel, skew, kurt, weekly autocorr, trend sign)
    - Adds: Cross-feature interactions (HR efficiency, GPS×WiFi, night disruption, …)
    - Adds: Behavioral consistency (habit score, personal z-score, global anomaly)
    - Saves advanced_feature_table.parquet

2.  Run adversarial validation
    - Checks train/test distribution shift
    - Prints AUC (0.5 = good, >0.7 = concern)

3.  Train LGB models (3 configurations) using group_time CV
    - adv_core       : advanced features, public_core view
    - adv_hist411    : advanced features, public_hist411 view
    - adv_hist365    : advanced features, public_hist365 view
    NOTE: Uses group_time forward-chaining (no subject leakage)

4.  Train CatBoost (per-target hyperparameters, group_time CV)
    - catboost_adv_core : advanced features, all targets

5.  Optimize ensemble weights (per-target SLSQP/grid search on OOF)
    - Combines LGB adv_core, adv_hist411, adv_hist365, CatBoost adv_core

6.  Apply isotonic calibration on blended OOF → fit calibrators

7.  Generate final test predictions + submission CSV

Run
---
    cd <repo_root>
    python scripts/train_advanced_v2.py [--rebuild-features] [--skip-catboost]

Arguments
---------
--rebuild-features  Force rebuild of feature tables even if cached.
--skip-catboost     Skip CatBoost training (for faster iteration).
--seeds             Comma-separated seed list (default: 42,1234,9999,7,314,2025,777,555).
--n-folds           Number of folds for group_time CV (default: 3).
--tag               Submission tag prefix (default: v2_adv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── bootstrap path ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / ".vendor"))

import numpy as np
import pandas as pd

from etri_human_challenge.advanced_features import (
    build_advanced_feature_table,
    load_advanced_feature_table,
)
from etri_human_challenge.catboost_model import train_catboost_targetwise
from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.ensemble_optimizer import (
    IsotonicEnsembleCalibrator,
    blend_predictions,
    evaluate_oof_blend,
    optimize_ensemble_weights,
    report_ensemble_weights,
)
from etri_human_challenge.group_time_cv import compute_adversarial_auc
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import (
    FEATURES_DIR,
    MODELS_DIR,
    OOF_DIR,
    REPORT_OOF_DIR,
    REPORT_SUBMISSIONS_DIR,
    SUBMISSIONS_DIR,
    ensure_runtime_dirs,
)
from etri_human_challenge.public_lgb import (
    PUBLIC_LGB_PARAMS,
    get_public_lgb_feature_columns,
    load_public_lgb_feature_table,
    PUBLIC_LGB_FEATURE_VIEWS,
)
from etri_human_challenge.utils import (
    clip_probabilities,
    multi_target_log_loss,
    write_json,
    write_markdown,
)


# ─────────────────────────────────────────────
# Per-target LGB hyperparameter overrides
# ─────────────────────────────────────────────

_LGB_TARGET_PARAMS: dict[str, dict] = {
    # Hard targets: more regularization, lower LR
    "Q1": dict(num_leaves=31, learning_rate=0.02, min_child_samples=20, reg_alpha=0.3, reg_lambda=2.0),
    "Q2": dict(num_leaves=15, learning_rate=0.015, min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0),
    "Q3": dict(num_leaves=15, learning_rate=0.015, min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0),
    # Easier targets: more capacity
    "S1": dict(num_leaves=31, learning_rate=0.02, min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0),
    "S2": dict(num_leaves=31, learning_rate=0.02, min_child_samples=20, reg_alpha=0.2, reg_lambda=1.5),
    "S3": dict(num_leaves=31, learning_rate=0.02, min_child_samples=20, reg_alpha=0.2, reg_lambda=1.5),
    # Hardest target: maximum regularization
    "S4": dict(num_leaves=15, learning_rate=0.01, min_child_samples=35, reg_alpha=0.8, reg_lambda=4.0),
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_advanced_feature_cols(
    frame: pd.DataFrame,
    feature_view: str,
) -> list[str]:
    """
    Extend the standard public_lgb feature list with new advanced columns.
    New columns follow the naming: __accel_, __roll_skew_, __roll_kurt_,
    __weekly_autocorr, __trend_sign_, __habit_score_, __personal_zscore,
    x_hr_per_step_log, x_gps_wifi_social, etc.
    """
    base_cols = set(get_public_lgb_feature_columns(frame, feature_view=feature_view))

    # Advanced column patterns to include
    advanced_patterns = [
        "__accel_",
        "__roll_skew_",
        "__roll_kurt_",
        "__weekly_autocorr",
        "__trend_sign_",
        "__habit_score_",
        "__personal_zscore",
        "__expanding_rank",
        "x_hr_per_step_log",
        "x_gps_wifi_social",
        "x_ble_wifi_indoor",
        "x_night_disruption",
        "x_activity_entropy_hr",
        "x_screen_session_density",
        "x_speech_mobility",
        "x_hr_rest_day_gap",
        "x_global_anomaly_score",
    ]

    all_numeric = [
        col for col in frame.columns
        if pd.api.types.is_numeric_dtype(frame[col])
        and col not in set(KEY_COLUMNS + TARGET_COLUMNS + ["split"])
    ]

    extra_cols = {
        col for col in all_numeric
        if any(pat in col for pat in advanced_patterns)
    }

    selected = sorted(base_cols | extra_cols, key=lambda c: list(frame.columns).index(c) if c in frame.columns else 9999)
    # Keep only columns that exist in frame
    return [c for c in selected if c in frame.columns]


# ─────────────────────────────────────────────
# LGB training with group_time CV
# ─────────────────────────────────────────────

def _train_lgb_group_time(
    *,
    frame: pd.DataFrame,
    feature_cols_by_target: dict[str, list[str]],
    run_name: str,
    seeds: list[int],
    n_folds: int,
) -> dict:
    """Train LGB with group_time forward-chaining CV."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("lightgbm is not installed")

    from etri_human_challenge.group_time_cv import group_time_split_iter

    ensure_runtime_dirs()
    train = frame[frame["split"] == "train"].reset_index(drop=True).copy()
    test  = frame[frame["split"] == "test"].reset_index(drop=True).copy()

    oof_preds  = np.zeros((len(train), len(TARGET_COLUMNS)), dtype=float)
    test_preds = np.zeros((len(test),  len(TARGET_COLUMNS)), dtype=float)
    seed_scores: dict[str, list[dict]] = {}

    for target_idx, target in enumerate(TARGET_COLUMNS):
        feature_cols = feature_cols_by_target[target]
        X_train = train[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
        X_test  = test[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
        y = train[target].astype(int).to_numpy()

        # Per-target LGB params
        t_overrides = _LGB_TARGET_PARAMS.get(target, {})
        base_params = {
            **PUBLIC_LGB_PARAMS,
            **t_overrides,
            "n_estimators": 3000,  # more room, early stopping will find optimum
        }

        target_oof  = np.zeros(len(train), dtype=float)
        target_test = np.zeros(len(test),  dtype=float)
        target_seed_scores: list[dict] = []

        for seed in seeds:
            params = {**base_params, "random_state": int(seed)}
            splits = list(group_time_split_iter(train, n_folds=n_folds))

            seed_oof  = np.zeros(len(train), dtype=float)
            seed_test = np.zeros(len(test),  dtype=float)

            for tr_idx, va_idx in splits:
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train.iloc[tr_idx],
                    y[tr_idx],
                    eval_set=[(X_train.iloc[va_idx], y[va_idx])],
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
                )
                seed_oof[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]
                seed_test += model.predict_proba(X_test)[:, 1] / len(splits)

            from sklearn.metrics import log_loss as _log_loss
            oof_ll = float(_log_loss(y, clip_probabilities(seed_oof)))
            target_seed_scores.append({"seed": int(seed), "log_loss": oof_ll})
            target_oof  += seed_oof
            target_test += seed_test

        oof_preds[:, target_idx]  = target_oof  / len(seeds)
        test_preds[:, target_idx] = target_test / len(seeds)
        seed_scores[target] = target_seed_scores

    oof_frame = pd.DataFrame(oof_preds, columns=TARGET_COLUMNS)
    score_payload = multi_target_log_loss(train[TARGET_COLUMNS], oof_frame, TARGET_COLUMNS)

    # Persist
    test_pred_path = MODELS_DIR / f"test_predictions_{run_name}.csv"
    pd.DataFrame(test_preds, columns=TARGET_COLUMNS).to_csv(test_pred_path, index=False)

    oof_export = train[KEY_COLUMNS + TARGET_COLUMNS].copy()
    for t in TARGET_COLUMNS:
        oof_export[f"{t}_lgb"] = oof_frame[t].values
    oof_path = OOF_DIR / f"oof_predictions_{run_name}.parquet"
    oof_export.to_parquet(oof_path, index=False)

    write_json(FEATURES_DIR / f"{run_name}_summary.json", {
        "run_name": run_name,
        "n_folds": n_folds,
        "cv_scheme": "group_time",
        "seeds": seeds,
        "scores": score_payload,
    })

    return {
        "run_name": run_name,
        "scores": score_payload,
        "train_predictions": oof_frame,
        "test_predictions": pd.DataFrame(test_preds, columns=TARGET_COLUMNS),
        "test_pred_path": str(test_pred_path),
        "oof_path": str(oof_path),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced v2 training pipeline")
    p.add_argument("--rebuild-features", action="store_true",
                   help="Force rebuild of feature tables even if cached")
    p.add_argument("--skip-catboost", action="store_true",
                   help="Skip CatBoost training (faster iteration)")
    p.add_argument("--seeds", default="42,1234,9999,7,314,2025,777,555",
                   help="Comma-separated seed list")
    p.add_argument("--n-folds", type=int, default=3,
                   help="Number of group_time CV folds")
    p.add_argument("--tag", default="v2_adv",
                   help="Submission tag prefix")
    p.add_argument("--n-restarts", type=int, default=20,
                   help="Optimizer restarts for ensemble weight search")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip isotonic calibration")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ensure_runtime_dirs()

    print("=" * 60)
    print("ETRI Advanced v2 Pipeline")
    print("=" * 60)

    # ── Step 1: Build / load advanced features ────────────────────
    print("\n[1/6] Building advanced feature table …")
    if args.rebuild_features or not (FEATURES_DIR / "advanced_feature_table.parquet").exists():
        base = load_public_lgb_feature_table(rebuild=False)
        frame = build_advanced_feature_table(base, persist=True)
    else:
        frame = load_advanced_feature_table(rebuild=False)

    train = frame[frame["split"] == "train"].reset_index(drop=True)
    test  = frame[frame["split"] == "test"].reset_index(drop=True)
    print(f"  Frame: {frame.shape}  |  train={len(train)}  test={len(test)}")

    # ── Step 2: Adversarial validation ────────────────────────────
    print("\n[2/6] Adversarial validation …")
    adv_cols = get_public_lgb_feature_columns(frame, feature_view="public_core")
    adv_auc = compute_adversarial_auc(frame, adv_cols)
    print(f"  Train/test AUC = {adv_auc:.4f}  (0.50 ideal, >0.70 = distribution shift)")
    if adv_auc > 0.70:
        print("  ⚠  Significant distribution shift detected — consider feature selection")

    # ── Step 3: LGB with group_time CV ────────────────────────────
    print("\n[3/6] Training LGB (group_time CV, 3 views) …")
    lgb_results: dict[str, dict] = {}

    # Target-feature-view mapping (best from histmix_guarded_v1)
    target_view_map = {
        "Q1": "public_core",
        "Q2": "public_hist411",
        "Q3": "public_hist365",
        "S1": "public_core",
        "S2": "public_core",
        "S3": "public_core",
        "S4": "public_hist411",
    }

    # Build per-target feature columns (with advanced features appended)
    for view_name, run_label in [
        ("public_core",   "lgb_adv_core"),
        ("public_hist411","lgb_adv_hist411"),
        ("public_hist365","lgb_adv_hist365"),
    ]:
        adv_feat_cols = _get_advanced_feature_cols(frame, view_name)
        feature_cols_by_target: dict[str, list[str]] = {
            target: adv_feat_cols for target in TARGET_COLUMNS
        }
        print(f"  Training {run_label} (n_features={len(adv_feat_cols)}) …")
        result = _train_lgb_group_time(
            frame=frame,
            feature_cols_by_target=feature_cols_by_target,
            run_name=run_label,
            seeds=seeds,
            n_folds=args.n_folds,
        )
        lgb_results[run_label] = result
        scores = result["scores"]
        print(f"    OOF mean log-loss = {scores['mean']:.6f}")
        for t in TARGET_COLUMNS:
            print(f"      {t}: {scores[t]:.6f}")

    # ── Target-wise best LGB predictions ──────────────────────────
    print("\n  Building target-wise LGB blend (same logic as histmix_guarded) …")
    targetwise_oof  = pd.DataFrame(index=range(len(train)), columns=TARGET_COLUMNS, dtype=float)
    targetwise_test = pd.DataFrame(index=range(len(test)),  columns=TARGET_COLUMNS, dtype=float)
    view_to_run = {
        "public_core":    "lgb_adv_core",
        "public_hist411": "lgb_adv_hist411",
        "public_hist365": "lgb_adv_hist365",
    }
    for target in TARGET_COLUMNS:
        run_key = view_to_run[target_view_map[target]]
        targetwise_oof[target]  = lgb_results[run_key]["train_predictions"][target].values
        targetwise_test[target] = lgb_results[run_key]["test_predictions"][target].values

    tw_scores = multi_target_log_loss(train[TARGET_COLUMNS], targetwise_oof, TARGET_COLUMNS)
    print(f"  Target-wise LGB OOF = {tw_scores['mean']:.6f}")

    # ── Step 4: CatBoost ──────────────────────────────────────────
    cat_result: dict | None = None
    if not args.skip_catboost:
        print("\n[4/6] Training CatBoost (group_time CV) …")
        cat_feat_cols = _get_advanced_feature_cols(frame, "public_core")
        cat_result = train_catboost_targetwise(
            frame=frame,
            feature_cols=cat_feat_cols,
            run_name="catboost_adv_v2",
            n_folds=args.n_folds,
            seeds=seeds,
            cv_scheme="group_time",
            persist=True,
        )
        cat_scores = cat_result["scores"]
        print(f"  CatBoost OOF mean = {cat_scores['mean']:.6f}")
        for t in TARGET_COLUMNS:
            print(f"    {t}: {cat_scores[t]:.6f}")
    else:
        print("\n[4/6] CatBoost skipped.")

    # ── Step 5: Ensemble weight optimization ──────────────────────
    print("\n[5/6] Optimizing ensemble weights …")

    oof_list: list[pd.DataFrame] = [targetwise_oof]
    test_list: list[pd.DataFrame] = [targetwise_test]
    model_names = ["lgb_targetwise"]

    # Always add core and hist411 as separate ensemble components
    for run_key, label in [
        ("lgb_adv_core",    "lgb_core"),
        ("lgb_adv_hist411", "lgb_hist411"),
        ("lgb_adv_hist365", "lgb_hist365"),
    ]:
        oof_list.append(lgb_results[run_key]["train_predictions"])
        test_list.append(lgb_results[run_key]["test_predictions"])
        model_names.append(label)

    if cat_result is not None:
        oof_list.append(cat_result["train_predictions"])
        test_list.append(cat_result["test_predictions"])
        model_names.append("catboost")

    optimal_weights = optimize_ensemble_weights(
        oof_list,
        train[TARGET_COLUMNS],
        n_restarts=args.n_restarts,
    )

    blend_oof  = blend_predictions(oof_list,  optimal_weights)
    blend_test = blend_predictions(test_list, optimal_weights)
    blend_scores = multi_target_log_loss(train[TARGET_COLUMNS], blend_oof, TARGET_COLUMNS)
    print(f"  Optimized blend OOF = {blend_scores['mean']:.6f}")
    for t in TARGET_COLUMNS:
        print(f"    {t}: {blend_scores[t]:.6f}")

    weight_report = report_ensemble_weights(optimal_weights, model_names)
    print("\n" + weight_report)
    write_markdown(REPORT_OOF_DIR / "ensemble_weights_v2.md", weight_report)

    # ── Step 6: Isotonic calibration ──────────────────────────────
    if not args.no_calibration:
        print("\n[6/6] Applying isotonic calibration …")
        calibrator = IsotonicEnsembleCalibrator()
        calibrator.fit(blend_oof, train[TARGET_COLUMNS])
        calibrated_oof  = calibrator.predict(blend_oof)
        calibrated_test = calibrator.predict(blend_test)
        cal_scores = multi_target_log_loss(train[TARGET_COLUMNS], calibrated_oof, TARGET_COLUMNS)
        print(f"  Calibrated OOF = {cal_scores['mean']:.6f}  (pre-calib: {blend_scores['mean']:.6f})")
    else:
        calibrated_test = blend_test
        cal_scores = blend_scores
        print("\n[6/6] Calibration skipped.")

    # ── Generate submission ────────────────────────────────────────
    print("\nGenerating submission …")
    template = load_submission_template()
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        raw = calibrated_test[target].to_numpy(dtype=float)
        submission[target] = np.clip(raw, 0.02, 0.98)

    submission_path = SUBMISSIONS_DIR / f"submission_{args.tag}_ensemble.csv"
    submission.to_csv(submission_path, index=False)
    print(f"  Saved: {submission_path}")

    # ── Save ensemble weights for reproducibility ──────────────────
    write_json(
        FEATURES_DIR / f"{args.tag}_ensemble_weights.json",
        {
            "tag": args.tag,
            "model_names": model_names,
            "optimal_weights": optimal_weights,
            "oof_scores_pre_calibration": blend_scores,
            "oof_scores_post_calibration": cal_scores,
        },
    )

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  LGB target-wise OOF  : {tw_scores['mean']:.6f}")
    print(f"  Ensemble blended OOF : {blend_scores['mean']:.6f}")
    print(f"  Calibrated OOF       : {cal_scores['mean']:.6f}")
    if cat_result:
        print(f"  CatBoost OOF         : {cat_result['scores']['mean']:.6f}")
    print(f"\n  Submission → {submission_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablation study: measure marginal gain of each new component.

For each ablation variant, trains a minimal model and reports OOF.
Use this to decide which features/strategies to keep before final submission.

Variants tested
---------------
A  baseline_v1          : LGB, public_core, StratifiedKFold (existing best)
B  group_time_only      : LGB, public_core, group_time CV
C  adv_features_only    : LGB, public_core + advanced, StratifiedKFold
D  group_time_adv       : LGB, public_core + advanced, group_time CV
E  targetwise_adv       : LGB, target-wise views + advanced, group_time CV
F  catboost_adv         : CatBoost, public_core + advanced, group_time CV
G  ensemble_lgb_only    : blend D + E
H  ensemble_full        : blend D + E + F (+ calibration)

Expected progression: A > B (stricter CV gives lower apparent OOF but better
generalization) → C slightly better or equal → D better → E slightly better
→ F adds diversity → G,H top ensemble.

Run
---
    python scripts/run_ablation_study.py [--fast] [--seeds 42,1234]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / ".vendor"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss as sk_log_loss

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from etri_human_challenge.advanced_features import load_advanced_feature_table
from etri_human_challenge.catboost_model import train_catboost_targetwise
from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.ensemble_optimizer import (
    IsotonicEnsembleCalibrator,
    blend_predictions,
    optimize_ensemble_weights,
)
from etri_human_challenge.group_time_cv import group_time_split_iter
from etri_human_challenge.paths import FEATURES_DIR, REPORT_EXPERIMENTS_DIR, ensure_runtime_dirs
from etri_human_challenge.public_lgb import (
    PUBLIC_LGB_PARAMS,
    get_public_lgb_feature_columns,
    load_public_lgb_feature_table,
)
from etri_human_challenge.utils import clip_probabilities, multi_target_log_loss, write_markdown


RESULTS: list[dict] = []


def _make_lgb_model(seed: int) -> "lgb.LGBMClassifier":
    params = {**PUBLIC_LGB_PARAMS, "random_state": seed, "n_estimators": 2000}
    return lgb.LGBMClassifier(**params)


def _train_single_view(
    frame: pd.DataFrame,
    feature_cols: list[str],
    seeds: list[int],
    n_folds: int,
    cv_scheme: str,
) -> pd.DataFrame:
    """Return OOF DataFrame with TARGET_COLUMNS columns."""
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    X = train[feature_cols].copy().replace([np.inf, -np.inf], np.nan)

    oof = np.zeros((len(train), len(TARGET_COLUMNS)), dtype=float)

    for tidx, target in enumerate(TARGET_COLUMNS):
        y = train[target].astype(int).to_numpy()
        target_oof = np.zeros(len(train), dtype=float)

        for seed in seeds:
            if cv_scheme == "stratified":
                splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y))
            else:
                splits = list(group_time_split_iter(train, n_folds=n_folds))

            seed_oof = np.zeros(len(train), dtype=float)
            for tr_idx, va_idx in splits:
                model = _make_lgb_model(seed)
                model.fit(
                    X.iloc[tr_idx], y[tr_idx],
                    eval_set=[(X.iloc[va_idx], y[va_idx])],
                    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
                )
                seed_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            target_oof += seed_oof

        oof[:, tidx] = target_oof / len(seeds)

    return pd.DataFrame(oof, columns=TARGET_COLUMNS)


def _score(label: str, oof: pd.DataFrame, y_true: pd.DataFrame) -> None:
    scores = multi_target_log_loss(y_true, oof, TARGET_COLUMNS)
    row = {"variant": label, "mean": scores["mean"]}
    for t in TARGET_COLUMNS:
        row[t] = scores[t]
    RESULTS.append(row)
    print(f"  {label:40s}: {scores['mean']:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="Use only 2 seeds for speed")
    p.add_argument("--seeds", default="42,1234,9999,7,314,2025,777,555")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--skip-catboost", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.fast:
        seeds = seeds[:2]
    n_folds = args.n_folds
    ensure_runtime_dirs()

    print("Loading feature tables …")
    base_frame = load_public_lgb_feature_table(rebuild=False)
    adv_frame  = load_advanced_feature_table(rebuild=False)
    train_base = base_frame[base_frame["split"] == "train"].reset_index(drop=True)
    train_adv  = adv_frame[adv_frame["split"]  == "train"].reset_index(drop=True)
    y_true = train_base[TARGET_COLUMNS]

    core_cols = get_public_lgb_feature_columns(base_frame, feature_view="public_core")
    h411_cols = get_public_lgb_feature_columns(base_frame, feature_view="public_hist411")
    h365_cols = get_public_lgb_feature_columns(base_frame, feature_view="public_hist365")

    # Advanced column names (same as train_advanced_v2.py)
    adv_patterns = [
        "__accel_", "__roll_skew_", "__roll_kurt_", "__weekly_autocorr",
        "__trend_sign_", "__habit_score_", "__personal_zscore", "__expanding_rank",
        "x_hr_per_step_log", "x_gps_wifi_social", "x_ble_wifi_indoor",
        "x_night_disruption", "x_activity_entropy_hr", "x_screen_session_density",
        "x_speech_mobility", "x_hr_rest_day_gap", "x_global_anomaly_score",
    ]
    adv_extra = [
        c for c in adv_frame.columns
        if pd.api.types.is_numeric_dtype(adv_frame[c])
        and any(pat in c for pat in adv_patterns)
        and c not in set(KEY_COLUMNS + TARGET_COLUMNS + ["split"])
    ]
    core_adv_cols = sorted(set(core_cols) | set(adv_extra),
                           key=lambda c: list(adv_frame.columns).index(c) if c in adv_frame.columns else 9999)
    core_adv_cols = [c for c in core_adv_cols if c in adv_frame.columns]

    print("\nAblation study")
    print("-" * 60)

    # A – baseline: public_core, StratifiedKFold (mirrors existing pipeline)
    print("A: baseline_v1 …")
    oof_a = _train_single_view(base_frame, core_cols, seeds, n_folds=5, cv_scheme="stratified")
    _score("A baseline_v1 (LGB/core/stratified-5fold)", oof_a, y_true)

    # B – group_time CV only change
    print("B: group_time CV …")
    oof_b = _train_single_view(base_frame, core_cols, seeds, n_folds, cv_scheme="group_time")
    _score(f"B group_time only (LGB/core/{n_folds}-fold)", oof_b, y_true)

    # C – advanced features only
    print("C: advanced features (stratified) …")
    oof_c = _train_single_view(adv_frame, core_adv_cols, seeds, n_folds=5, cv_scheme="stratified")
    _score("C adv features (LGB/core+adv/stratified)", oof_c, y_true)

    # D – advanced + group_time
    print("D: advanced + group_time …")
    oof_d = _train_single_view(adv_frame, core_adv_cols, seeds, n_folds, cv_scheme="group_time")
    _score(f"D adv + group_time (LGB/core+adv/{n_folds}-fold)", oof_d, y_true)

    # E – target-wise views + advanced + group_time
    print("E: target-wise + advanced …")
    view_map = {"Q1": "public_core", "Q2": "public_hist411", "Q3": "public_hist365",
                "S1": "public_core", "S2": "public_core",    "S3": "public_core",   "S4": "public_hist411"}
    oof_e_parts: dict[str, pd.DataFrame] = {}
    for view in ["public_core", "public_hist411", "public_hist365"]:
        base_v = set(get_public_lgb_feature_columns(adv_frame, feature_view=view))
        cols_v = sorted(base_v | set(adv_extra), key=lambda c: list(adv_frame.columns).index(c) if c in adv_frame.columns else 9999)
        cols_v = [c for c in cols_v if c in adv_frame.columns]
        oof_e_parts[view] = _train_single_view(adv_frame, cols_v, seeds, n_folds, "group_time")

    oof_e = pd.DataFrame(index=range(len(train_adv)), columns=TARGET_COLUMNS, dtype=float)
    for target in TARGET_COLUMNS:
        oof_e[target] = oof_e_parts[view_map[target]][target].values
    _score("E target-wise+adv+group_time", oof_e, y_true)

    # F – CatBoost
    if not args.skip_catboost:
        print("F: CatBoost …")
        cat_r = train_catboost_targetwise(
            frame=adv_frame, feature_cols=core_adv_cols,
            run_name="catboost_ablation", n_folds=n_folds,
            seeds=seeds[:4] if args.fast else seeds,
            cv_scheme="group_time", persist=False,
        )
        oof_f = cat_r["train_predictions"]
        _score("F CatBoost/core+adv/group_time", oof_f, y_true)
    else:
        oof_f = None

    # G – ensemble D + E
    print("G: ensemble LGB (D+E) …")
    w_g = optimize_ensemble_weights([oof_d, oof_e], y_true, n_restarts=20)
    oof_g = blend_predictions([oof_d, oof_e], w_g)
    _score("G ensemble(D+E)", oof_g, y_true)

    # H – full ensemble + calibration
    print("H: full ensemble + calibration …")
    oof_h_parts = [oof_d, oof_e]
    if oof_f is not None:
        oof_h_parts.append(oof_f)
    w_h = optimize_ensemble_weights(oof_h_parts, y_true, n_restarts=20)
    oof_h_raw = blend_predictions(oof_h_parts, w_h)
    cal = IsotonicEnsembleCalibrator().fit(oof_h_raw, y_true)
    oof_h = cal.predict(oof_h_raw)
    _score("H full ensemble + calibration", oof_h, y_true)

    # ── Report ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    results_df = pd.DataFrame(RESULTS).set_index("variant")
    print(results_df.to_string(float_format="{:.6f}".format))

    lines = ["# Ablation Study Results", "", results_df.to_markdown()]
    write_markdown(REPORT_EXPERIMENTS_DIR / "ablation_v2.md", "\n".join(lines))
    results_df.reset_index().to_csv(FEATURES_DIR / "ablation_v2_results.csv", index=False)
    print(f"\nSaved to {FEATURES_DIR / 'ablation_v2_results.csv'}")


if __name__ == "__main__":
    main()

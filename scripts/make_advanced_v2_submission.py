#!/usr/bin/env python3
"""Generate final advanced v2 submission from cached artifacts.

If training artifacts exist, reuses them.  Otherwise triggers training.

Quick usage
-----------
    # First time (train + submit):
    python scripts/train_advanced_v2.py

    # Subsequent (reuse cached, just rebuild submission):
    python scripts/make_advanced_v2_submission.py --tag v2_final

    # Try different calibration clip:
    python scripts/make_advanced_v2_submission.py --clip-min 0.01 --clip-max 0.99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / ".vendor"))

import numpy as np
import pandas as pd

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.ensemble_optimizer import (
    IsotonicEnsembleCalibrator,
    blend_predictions,
)
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, SUBMISSIONS_DIR, ensure_runtime_dirs
from etri_human_challenge.utils import clip_probabilities, multi_target_log_loss, write_markdown
from etri_human_challenge.public_lgb import load_public_lgb_feature_table


def _load_oof(run_name: str) -> pd.DataFrame | None:
    path = OOF_DIR / f"oof_predictions_{run_name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Rename prediction columns to TARGET_COLUMNS if needed
    rename = {}
    for t in TARGET_COLUMNS:
        for suffix in ["_lgb", "_catboost"]:
            if f"{t}{suffix}" in df.columns:
                rename[f"{t}{suffix}"] = t
    if rename:
        df = df.rename(columns=rename)
    return df[TARGET_COLUMNS] if all(t in df.columns for t in TARGET_COLUMNS) else None


def _load_test_preds(run_name: str) -> pd.DataFrame | None:
    path = MODELS_DIR / f"test_predictions_{run_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v2_final")
    p.add_argument("--clip-min", type=float, default=0.02)
    p.add_argument("--clip-max", type=float, default=0.98)
    p.add_argument("--weights-json", default=None,
                   help="Path to ensemble weights JSON from train_advanced_v2.py")
    p.add_argument("--no-calibration", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()

    # ── Locate weights file ────────────────────────────────────────
    weights_path = Path(args.weights_json) if args.weights_json else None
    if weights_path is None:
        # Try default location written by train_advanced_v2.py
        candidates = sorted(FEATURES_DIR.glob("*_ensemble_weights.json"))
        if candidates:
            weights_path = candidates[-1]
            print(f"Using weights: {weights_path}")

    if weights_path is None or not weights_path.exists():
        print("No weights file found. Run train_advanced_v2.py first.")
        sys.exit(1)

    weights_data = json.loads(weights_path.read_text())
    model_names = weights_data["model_names"]
    optimal_weights = weights_data["optimal_weights"]

    print(f"Models in ensemble: {model_names}")
    print(f"Pre-calibration OOF: {weights_data.get('oof_scores_pre_calibration', {}).get('mean', 'n/a')}")

    # ── Load OOF + test predictions ───────────────────────────────
    oof_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    run_map = {
        "lgb_targetwise":  None,  # built dynamically
        "lgb_core":        "lgb_adv_core",
        "lgb_hist411":     "lgb_adv_hist411",
        "lgb_hist365":     "lgb_adv_hist365",
        "catboost":        "catboost_adv_v2",
    }

    for name in model_names:
        run_name = run_map.get(name)
        if run_name is None:
            print(f"  Skipping {name} (no run_map entry)")
            # Use a dummy — will weight to 0 if optimizer did its job
            oof_frames.append(None)
            test_frames.append(None)
            continue

        oof = _load_oof(run_name)
        test = _load_test_preds(run_name)
        if oof is None or test is None:
            print(f"  Missing artifacts for {run_name} — skipping")
            oof_frames.append(None)
            test_frames.append(None)
        else:
            oof_frames.append(oof)
            test_frames.append(test)
            print(f"  Loaded {run_name}: oof={len(oof)} test={len(test)}")

    # Drop None entries and their weights
    valid_idx = [i for i, o in enumerate(oof_frames) if o is not None]
    oof_frames  = [oof_frames[i]  for i in valid_idx]
    test_frames = [test_frames[i] for i in valid_idx]
    valid_names = [model_names[i] for i in valid_idx]

    # Renormalize weights over valid models
    filtered_weights: dict[str, list[float]] = {}
    for target in TARGET_COLUMNS:
        w = np.array([optimal_weights[target][i] for i in valid_idx])
        s = w.sum()
        filtered_weights[target] = (w / s if s > 0 else np.ones(len(w)) / len(w)).tolist()

    blend_test = blend_predictions(test_frames, filtered_weights)

    # ── Optional calibration ───────────────────────────────────────
    if not args.no_calibration and oof_frames:
        blend_oof = blend_predictions(oof_frames, filtered_weights)
        base = load_public_lgb_feature_table(rebuild=False)
        y_true = base[base["split"] == "train"][TARGET_COLUMNS].reset_index(drop=True)

        if len(blend_oof) == len(y_true):
            cal = IsotonicEnsembleCalibrator().fit(blend_oof, y_true)
            blend_test = cal.predict(blend_test)
            print("Isotonic calibration applied.")
        else:
            print(f"  OOF length mismatch ({len(blend_oof)} vs {len(y_true)}) — skipping calibration")

    # ── Build submission ───────────────────────────────────────────
    template = load_submission_template()
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        raw = blend_test[target].to_numpy(dtype=float)
        submission[target] = np.clip(raw, args.clip_min, args.clip_max)

    out_path = SUBMISSIONS_DIR / f"submission_{args.tag}.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path}")

    # Quick sanity check
    print("\nPrediction statistics:")
    for target in TARGET_COLUMNS:
        vals = submission[target].to_numpy(dtype=float)
        print(f"  {target}: mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.4f} max={vals.max():.4f}")


if __name__ == "__main__":
    main()

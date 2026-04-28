#!/usr/bin/env python3
"""Prepare short-named submission candidates for the public track."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import MODELS_DIR, OOF_DIR, ROOT
from etri_human_challenge.utils import binary_log_loss


CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"

RUN_CORE = "public_lgb_public_core"
RUN_H411 = "public_lgb_public_hist411"
RUN_H365 = "public_lgb_public_hist365"
RUN_GUARDED = "public_lgb_targetwise_histmix_guarded_v1"
RUN_HOLDOUT = "public_lgb_targetwise_histmix_guarded_v1_subject_holdout"


def _load_oof(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet")
    labels = frame[TARGET_COLUMNS].copy()
    predictions = pd.DataFrame(
        {target: frame[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS}
    )
    return labels, predictions


def _load_test(run_name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / f"test_predictions_{run_name}.csv")[TARGET_COLUMNS].astype(float)


def _score(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _softblend(source: dict[str, pd.DataFrame], weight: float) -> pd.DataFrame:
    output = pd.DataFrame(index=source["core"].index)
    output["Q1"] = source["core"]["Q1"]
    output["Q2"] = (1.0 - weight) * source["core"]["Q2"] + weight * source["hist411"]["Q2"]
    output["Q3"] = (1.0 - weight) * source["core"]["Q3"] + weight * source["hist365"]["Q3"]
    output["S1"] = source["core"]["S1"]
    output["S2"] = source["core"]["S2"]
    output["S3"] = source["core"]["S3"]
    output["S4"] = source["hist411"]["S4"]
    return output[TARGET_COLUMNS].clip(CLIP_MIN, CLIP_MAX)


def _add_holdout_s2s3(base: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    output = base.copy()
    output["S2"] = 0.88 * base["S2"] + 0.12 * holdout["S2"]
    output["S3"] = 0.94 * base["S3"] + 0.06 * holdout["S3"]
    return output[TARGET_COLUMNS].clip(CLIP_MIN, CLIP_MAX)


def _make_submission(predictions: pd.DataFrame) -> pd.DataFrame:
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = predictions[target].to_numpy(float)
    if submission.shape != (250, 10):
        raise ValueError(f"Invalid submission shape: {submission.shape}")
    if submission.columns.tolist() != template.columns.tolist():
        raise ValueError("Submission columns do not match sample_submission.")
    values = submission[TARGET_COLUMNS].to_numpy(float)
    if values.min() < CLIP_MIN or values.max() > CLIP_MAX:
        raise ValueError(f"Predictions outside [{CLIP_MIN}, {CLIP_MAX}].")
    return submission


def _write_submission(path: Path, predictions: pd.DataFrame, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _make_submission(predictions).to_csv(path, index=False)


def _write_candidate_log(rows: list[dict[str, object]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "candidate_scores.csv"
    columns = [
        "rank",
        "candidate",
        "oof_mean",
        "public_score",
        "submission_file",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_public_score_log() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "timestamp": "2026-04-25",
            "submission_file": "submission_public_lgb_v3_public_lgb_targetwise_histmix_guarded_v1.csv",
            "experiment_name": "histmix_guarded_v1",
            "public_score": "0.5960566585",
            "delta_vs_best": "0.0000000000",
            "notes": "Current best. Keep as anchor.",
        },
        {
            "timestamp": "2026-04-28 00:49:31",
            "submission_file": "submission_public_lgb_v4_softblend_w090.csv",
            "experiment_name": "softblend_w090",
            "public_score": "0.5962890684",
            "delta_vs_best": "0.0002324099",
            "notes": "Worse than current best; stop softblend direction.",
        },
        {
            "timestamp": "2026-04-28 00:38:26",
            "submission_file": "submission_public_lgb_v5_public_lgb_targetwise_histmix_guarded_v1_subject_holdout.csv",
            "experiment_name": "histmix_guarded_v1_subject_holdout",
            "public_score": "0.5968333841",
            "delta_vs_best": "0.0007767256",
            "notes": "Worse than current best; do not promote.",
        },
    ]
    path = LOG_DIR / "public_scores.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Ready Submission Candidates",
        "",
        "Use this folder for short, readable submission filenames. Existing artifacts are preserved elsewhere.",
        "",
        "## Public Score Notes",
        "",
        "- Current best: `0.5960566585` (`histmix_guarded_v1`).",
        "- `soft_w090`: `0.5962890684`, worse by `+0.0002324099`; softblend direction is not promoted.",
        "- Subject-holdout retrain: `0.5968333841`, worse by `+0.0007767256`; not promoted.",
        "",
        "## Recommended Order",
        "",
        "| Rank | File | OOF mean | Public score | Notes |",
        "|---:|---|---:|---:|---|",
    ]
    for row in rows:
        public_score = row["public_score"] or ""
        lines.append(
            f"| {row['rank']} | `{Path(str(row['submission_file'])).name}` | "
            f"{float(row['oof_mean']):.6f} | {public_score} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Naming",
            "",
            "- `soft_w090`: Q2/Q3 use 90% history view and 10% core.",
            "- `guarded_s2s3`: keeps the current best guarded target map, then blends only S2 with 12% subject-holdout and S3 with 6% subject-holdout.",
        ]
    )
    (ROOT / "submissions" / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels, oof_core = _load_oof(RUN_CORE)
    _, oof_h411 = _load_oof(RUN_H411)
    _, oof_h365 = _load_oof(RUN_H365)
    _, oof_guarded = _load_oof(RUN_GUARDED)
    _, oof_holdout = _load_oof(RUN_HOLDOUT)

    test_core = _load_test(RUN_CORE)
    test_h411 = _load_test(RUN_H411)
    test_h365 = _load_test(RUN_H365)
    test_guarded = _load_test(RUN_GUARDED)
    test_holdout = _load_test(RUN_HOLDOUT)

    oof_source = {"core": oof_core, "hist411": oof_h411, "hist365": oof_h365}
    test_source = {"core": test_core, "hist411": test_h411, "hist365": test_h365}

    candidates: list[tuple[int, str, pd.DataFrame, pd.DataFrame, str, str]] = [
        (
            0,
            "00_best_guarded.csv",
            oof_guarded,
            test_guarded,
            "0.5960566585",
            "Reference current best; keep as anchor, not a new probe.",
        ),
        (
            1,
            "next_guarded_s2s3.csv",
            _add_holdout_s2s3(oof_guarded, oof_holdout),
            _add_holdout_s2s3(test_guarded, test_holdout),
            "",
            "Next low-risk probe: keep current best Q1/Q2/Q3/S1/S4, blend only S2/S3 with tiny subject-holdout weights.",
        ),
        (
            2,
            "01_soft_w090.csv",
            _softblend(oof_source, 0.90),
            _softblend(test_source, 0.90),
            "0.5962890684",
            "Submitted and worse than best; stop this softblend direction.",
        ),
        (
            3,
            "02_soft_w090_s2s3.csv",
            _add_holdout_s2s3(_softblend(oof_source, 0.90), oof_holdout),
            _add_holdout_s2s3(_softblend(test_source, 0.90), test_holdout),
            "",
            "Demoted because it inherits failed Q2/Q3 softblend direction.",
        ),
        (
            4,
            "03_soft_w085_s2s3.csv",
            _add_holdout_s2s3(_softblend(oof_source, 0.85), oof_holdout),
            _add_holdout_s2s3(_softblend(test_source, 0.85), test_holdout),
            "",
            "Demoted because it moves farther in failed softblend direction.",
        ),
        (
            5,
            "04_soft_w085.csv",
            _softblend(oof_source, 0.85),
            _softblend(test_source, 0.85),
            "",
            "Do not submit for now; likely same failure direction as w090.",
        ),
        (
            6,
            "05_soft_w095.csv",
            _softblend(oof_source, 0.95),
            _softblend(test_source, 0.95),
            "",
            "Lower-risk than w085 but still same failed softblend family.",
        ),
    ]

    existing = [str((READY_DIR / item[1]).relative_to(ROOT)) for item in candidates if (READY_DIR / item[1]).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing ready submissions. "
            f"Use --overwrite to regenerate: {existing}"
        )

    log_rows: list[dict[str, object]] = []
    for rank, filename, oof_pred, test_pred, public_score, notes in candidates:
        output_path = READY_DIR / filename
        _write_submission(output_path, test_pred, overwrite=args.overwrite)
        scores = _score(labels, oof_pred)
        log_rows.append(
            {
                "rank": rank,
                "candidate": filename,
                "oof_mean": scores["mean"],
                "public_score": public_score,
                "submission_file": str(output_path.relative_to(ROOT)),
                "notes": notes,
            }
        )

    _write_candidate_log(log_rows)
    _write_public_score_log()
    _write_readme(log_rows)

    print(json.dumps({"status": "ok", "candidates": log_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

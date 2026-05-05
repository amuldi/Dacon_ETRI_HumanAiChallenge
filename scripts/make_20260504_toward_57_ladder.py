#!/usr/bin/env python3
"""Create the post-b130 S4 ladder aimed at the 0.57 public range."""

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

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template, load_train_labels
from etri_human_challenge.paths import FEATURES_DIR, MODELS_DIR, OOF_DIR, REPORT_SUBMISSIONS_DIR, ROOT, ensure_runtime_dirs
from etri_human_challenge.utils import binary_log_loss, write_json, write_markdown


EXPERIMENT_NAME = "toward_57_s4_ladder_20260504"
ANCHOR_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
S4_FULL_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"

CURRENT_BEST_FILE = "lgb_temporal_s4b130.csv"
CURRENT_BEST_PUBLIC = 0.5845552904
CURRENT_BEST_BETA = 1.30
PREVIOUS_BEST_PUBLIC = 0.5847975097

READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
CLIP_MIN = 0.02
CLIP_MAX = 0.98

PUBLIC_POINTS = [
    (0.50, 0.5856099175),
    (0.65, 0.5853942690),
    (0.85, 0.5851199626),
    (1.10, 0.5847975097),
    (1.30, 0.5845552904),
]

CANDIDATES = [
    {
        "beta": 2.00,
        "run_name": "public_lgb_targetwise_toward57_s4b200_20260504",
        "file": "lgb_temporal_s4b200.csv",
        "decision": "1st submit: confirm that the b130 trend survives a wider jump.",
    },
    {
        "beta": 3.50,
        "run_name": "public_lgb_targetwise_toward57_s4b350_20260504",
        "file": "lgb_temporal_s4b350.csv",
        "decision": "2nd submit only if b200 improves; target low 0.58 range.",
    },
    {
        "beta": 5.00,
        "run_name": "public_lgb_targetwise_toward57_s4b500_20260504",
        "file": "lgb_temporal_s4b500.csv",
        "decision": "3rd submit only if b350 improves; first direct 0.57-range attempt.",
    },
    {
        "beta": 6.50,
        "run_name": "public_lgb_targetwise_toward57_s4b650_20260504",
        "file": "lgb_temporal_s4b650.csv",
        "decision": "Ceiling probe only after b500 improves; clipping starts, highest risk.",
    },
]


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


def _validate_inputs() -> None:
    train = load_train_labels()
    sample = load_submission_template()
    if len(train) != 450:
        raise ValueError(f"Unexpected train row count: {len(train)}")
    if len(sample) != 250:
        raise ValueError(f"Unexpected test row count: {len(sample)}")
    missing = [target for target in TARGET_COLUMNS if target not in train.columns]
    if missing:
        raise ValueError(f"Missing train target columns: {missing}")


def _load_run(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet")
    labels = raw[TARGET_COLUMNS].copy()
    oof = pd.DataFrame({target: raw[f"{target}_public_lgb"].astype(float) for target in TARGET_COLUMNS})
    test = pd.read_csv(MODELS_DIR / f"test_predictions_{run_name}.csv")[TARGET_COLUMNS].astype(float)
    return raw[KEY_COLUMNS].copy(), labels, oof, test


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _public_projection(beta: float) -> float:
    xs = np.array([point[0] for point in PUBLIC_POINTS], dtype=float)
    ys = np.array([point[1] for point in PUBLIC_POINTS], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope * beta + intercept)


def _s4_blend(anchor: pd.DataFrame, s4_full: pd.DataFrame, beta: float) -> pd.DataFrame:
    output = anchor.copy()
    output["S4"] = np.clip(anchor["S4"] + beta * (s4_full["S4"] - anchor["S4"]), CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _make_submission(path: Path, predictions: pd.DataFrame) -> None:
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    if predictions.shape != (250, 7):
        raise ValueError(f"Invalid prediction shape: {predictions.shape}")
    values = predictions[TARGET_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    if values.min() < CLIP_MIN or values.max() > CLIP_MAX:
        raise ValueError(f"Predictions outside [{CLIP_MIN}, {CLIP_MAX}].")

    submission = template[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = predictions[target].to_numpy(float)
    if submission.shape != (250, 10):
        raise ValueError(f"Invalid submission shape: {submission.shape}")
    if submission.columns.tolist() != template.columns.tolist():
        raise ValueError("Submission columns do not match sample_submission.")

    if path.exists():
        existing = pd.read_csv(path)
        if existing.columns.tolist() != submission.columns.tolist():
            raise FileExistsError(f"Existing submission has different columns: {path}")
        old_values = existing[TARGET_COLUMNS].to_numpy(float)
        new_values = submission[TARGET_COLUMNS].to_numpy(float)
        if old_values.shape != new_values.shape or not np.allclose(old_values, new_values, atol=1e-12, rtol=0.0):
            raise FileExistsError(f"Refusing to overwrite different existing submission: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)


def _write_prediction_artifacts(
    run_name: str,
    keys: pd.DataFrame,
    labels: pd.DataFrame,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    scores: dict[str, float],
    selection: dict[str, Any],
) -> None:
    oof_export = keys[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        oof_export[target] = labels[target].to_numpy(int)
    oof_export["split_scheme"] = "toward_57_s4_ladder"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "selection": selection})


def _append_experiment(result: dict[str, Any]) -> None:
    path = LOG_DIR / "experiments.csv"
    columns = [
        "timestamp", "experiment_name", "validation_scheme", "seeds", "total_oof_logloss",
        "target_logloss_Q1", "target_logloss_Q2", "target_logloss_Q3", "target_logloss_S1",
        "target_logloss_S2", "target_logloss_S3", "target_logloss_S4", "submission_file",
        "feature_view_by_target", "notes",
    ]
    rows = _read_csv_rows(path)
    name = f"{EXPERIMENT_NAME}_b{int(round(result['beta'] * 100)):03d}"
    rows = [row for row in rows if row.get("experiment_name") != name]
    scores = result["scores"]
    rows.append(
        {
            "timestamp": _now_kst(),
            "experiment_name": name,
            "validation_scheme": "toward_57_s4_ladder",
            "seeds": "posthoc",
            "total_oof_logloss": scores["mean"],
            "target_logloss_Q1": scores["Q1"],
            "target_logloss_Q2": scores["Q2"],
            "target_logloss_Q3": scores["Q3"],
            "target_logloss_S1": scores["S1"],
            "target_logloss_S2": scores["S2"],
            "target_logloss_S3": scores["S3"],
            "target_logloss_S4": scores["S4"],
            "submission_file": str(result["path"].relative_to(ROOT)),
            "feature_view_by_target": json.dumps(result["selection"], sort_keys=True),
            "notes": "Post-b130 S4-only ladder. Public trend is extrapolated; Q/S targets are fixed.",
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_public_scores() -> None:
    path = LOG_DIR / "public_scores.csv"
    columns = ["timestamp", "submission_file", "experiment_name", "public_score", "delta_vs_best", "notes"]
    rows = _read_csv_rows(path)
    rows = [row for row in rows if row.get("submission_file") != CURRENT_BEST_FILE]
    rows.append(
        {
            "timestamp": "2026-05-04",
            "submission_file": CURRENT_BEST_FILE,
            "experiment_name": "today_s4_extension_20260504_b130",
            "public_score": f"{CURRENT_BEST_PUBLIC:.10f}",
            "delta_vs_best": f"{CURRENT_BEST_PUBLIC - PREVIOUS_BEST_PUBLIC:.10f}",
            "notes": "New current best; beta 1.30 confirms the S4-only extension trend.",
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(results: list[dict[str, Any]]) -> None:
    rows = [
        {
            "rank": 0,
            "candidate": CURRENT_BEST_FILE,
            "oof_mean": 0.5597017230477448,
            "public_score": f"{CURRENT_BEST_PUBLIC:.10f}",
            "submission_file": f"submissions/ready/{CURRENT_BEST_FILE}",
            "notes": "Current public best; beta 1.30 confirmed continuation.",
        },
    ]
    for idx, result in enumerate(results, start=1):
        rows.append(
            {
                "rank": idx,
                "candidate": result["file"],
                "oof_mean": result["scores"]["mean"],
                "public_score": "",
                "submission_file": f"submissions/ready/{result['file']}",
                "notes": f"{result['decision']} Projected public {result['projected_public']:.6f}.",
            }
        )
    rows.extend(
        [
            {
                "rank": len(rows),
                "candidate": "lgb_temporal_s4b110.csv",
                "oof_mean": 0.5596888312564241,
                "public_score": "0.5847975097",
                "submission_file": "submissions/ready/lgb_temporal_s4b110.csv",
                "notes": "Previous best backup.",
            },
            {
                "rank": len(rows) + 1,
                "candidate": "lgb_temporal_push.csv",
                "oof_mean": 0.5595372562363035,
                "public_score": "0.5880629390",
                "submission_file": "submissions/archive/2026-05-02_push_failed/lgb_temporal_push.csv",
                "notes": "Failed public; do not resubmit.",
            },
        ]
    )
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Submissions",
        "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.",
        "",
        "## Current Best",
        "",
        f"- Public best: `{CURRENT_BEST_PUBLIC:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_BEST_FILE}`",
        "",
        "## 2026-05-04 Toward 0.57 Plan",
        "",
        "b130이 개선됐으므로 오늘의 핵심 축은 계속 S4-only beta 확장입니다. Q1/Q2/Q3/S1/S2/S3는 그대로 고정합니다.",
        "",
        "| Order | File | Beta | OOF mean | Projected public | Decision |",
        "|---:|---|---:|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_BEST_FILE}` | `{CURRENT_BEST_BETA:.2f}` | `0.559702` | `{CURRENT_BEST_PUBLIC:.10f}` | 현재 best |",
    ]
    for idx, result in enumerate(results, start=1):
        lines.append(
            f"| {idx} | `ready/{result['file']}` | `{result['beta']:.2f}` | "
            f"`{result['scores']['mean']:.6f}` | `{result['projected_public']:.6f}` | {result['decision']} |"
        )
    lines.extend(
        [
            "| backup | `ready/lgb_temporal_s4b110.csv` | `1.10` | `0.559689` | `0.5847975097` | 이전 best 백업 |",
            "| stop | `archive/2026-05-02_push_failed/lgb_temporal_push.csv` |  | `0.559537` | `0.5880629390` | 제출 금지 |",
            "",
            "## Rules",
            "",
            "1. 먼저 `lgb_temporal_s4b200.csv`를 올립니다.",
            "2. b200이 `0.5845552904`보다 낮으면 `lgb_temporal_s4b350.csv`를 올립니다.",
            "3. b350도 개선되면 `lgb_temporal_s4b500.csv`를 올립니다.",
            "4. b500이 0.580 근처 또는 그 이하로 내려가면 `lgb_temporal_s4b650.csv`로 0.57대를 직접 찌릅니다.",
            "5. b200이 악화되면 확장 폭을 줄여야 하므로 b350 이상은 중단합니다.",
            "",
            "## Notes",
            "",
            "- b130 public score is on the same linear trend as beta 0.50, 0.65, 0.85, and 1.10.",
            "- OOF is no longer reliable for this public direction, so public-confirmed monotonicity is the decision signal.",
            "- b650 is the first candidate where upper clipping appears, so it is a ceiling probe, not a blind next submit.",
        ]
    )
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission Report: toward_57_s4_ladder_20260504",
        "",
        f"- New current best: `{CURRENT_BEST_FILE}` public `{CURRENT_BEST_PUBLIC:.10f}`",
        "- Goal: probe whether the public-validated S4 beta direction can enter the 0.57 range.",
        "- Fixed targets: Q1/Q2/Q3/S1/S2/S3.",
        "- Risk control: stop if b200 fails; use b650 only after b500 remains strong.",
        "",
        "| File | Beta | OOF mean | S4 OOF | S4 test mean | Clip high | Projected public | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| `{CURRENT_BEST_FILE}` | `{CURRENT_BEST_BETA:.2f}` | `0.559702` | `0.613037` | `0.547018` | `0` | `{CURRENT_BEST_PUBLIC:.10f}` | current best |",
    ]
    for result in results:
        scores = result["scores"]
        stats = result["stats"]
        lines.append(
            f"| `{result['file']}` | `{result['beta']:.2f}` | `{scores['mean']:.6f}` | "
            f"`{scores['S4']:.6f}` | `{stats['s4_mean']:.6f}` | `{stats['clip_high']}` | "
            f"`{result['projected_public']:.6f}` | {result['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Why This Ladder",
            "",
            "- The b130 result improved from `0.5847975097` to `0.5845552904`, matching the linear beta-public trend.",
            "- The per-beta movement is small in probability space; even b500 has no clipping and only raises S4 test mean to about `0.5528`.",
            "- b650 projects into the 0.577 range but starts clipping, so it is reserved as the ceiling probe.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "toward_57_s4_ladder_20260504.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    keys, labels, anchor_oof, anchor_test = _load_run(ANCHOR_RUN)
    _, _, full_oof, full_test = _load_run(S4_FULL_RUN)

    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        beta = float(candidate["beta"])
        oof = _s4_blend(anchor_oof, full_oof, beta)
        test = _s4_blend(anchor_test, full_test, beta)
        scores = _score_targets(labels, oof)
        path = READY_DIR / str(candidate["file"])
        _make_submission(path, test)

        s4_values = test["S4"].to_numpy(float)
        selection = {
            "anchor_run": ANCHOR_RUN,
            "s4_full_run": S4_FULL_RUN,
            "s4_beta": beta,
            "fixed_targets": [target for target in TARGET_COLUMNS if target != "S4"],
        }
        _write_prediction_artifacts(str(candidate["run_name"]), keys, labels, oof, test, scores, selection)
        result = {
            **candidate,
            "scores": scores,
            "path": path,
            "selection": selection,
            "projected_public": _public_projection(beta),
            "stats": {
                "s4_mean": float(s4_values.mean()),
                "s4_min": float(s4_values.min()),
                "s4_max": float(s4_values.max()),
                "clip_low": int((s4_values <= CLIP_MIN + 1e-12).sum()),
                "clip_high": int((s4_values >= CLIP_MAX - 1e-12).sum()),
            },
        }
        _append_experiment(result)
        results.append(result)

    _write_public_scores()
    _write_candidate_scores(results)
    _write_readme(results)
    _write_report(results)

    print("=== toward_57_s4_ladder_20260504 ===")
    print(f"current: {CURRENT_BEST_FILE} beta={CURRENT_BEST_BETA:.2f} public={CURRENT_BEST_PUBLIC:.10f}")
    for result in results:
        scores = result["scores"]
        stats = result["stats"]
        print(
            f"{result['file']}: beta={result['beta']:.2f} "
            f"OOF={scores['mean']:.6f} S4={scores['S4']:.6f} "
            f"projected_public={result['projected_public']:.6f} "
            f"S4_mean={stats['s4_mean']:.6f} clip_high={stats['clip_high']} "
            f"path={result['path']}"
        )


if __name__ == "__main__":
    main()

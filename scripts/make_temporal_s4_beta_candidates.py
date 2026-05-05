#!/usr/bin/env python3
"""Create S4 beta continuation candidates after S4 half improved public."""

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


EXPERIMENT_NAME = "temporal_s4_beta_v1"
CURRENT_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
S4_FULL_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"
CURRENT_BEST_FILE = "lgb_temporal_s4b085.csv"
CURRENT_BEST_PUBLIC = 0.5851199626
CURRENT_BEST_OOF = 0.5596903927971973

CANDIDATES = [
    {"beta": 0.95, "run_name": "public_lgb_targetwise_temporal_s4b095_v1", "file": "lgb_temporal_s4b095.csv", "label": "S4 beta 0.95"},
    {"beta": 1.00, "run_name": "public_lgb_targetwise_temporal_s4b100_v1", "file": "lgb_temporal_s4b100.csv", "label": "S4 beta 1.00"},
    {"beta": 1.10, "run_name": "public_lgb_targetwise_temporal_s4b110_v1", "file": "lgb_temporal_s4b110.csv", "label": "S4 beta 1.10"},
]

CLIP_MIN = 0.02
CLIP_MAX = 0.98
READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"


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


def _s4_blend(current: pd.DataFrame, s4_full: pd.DataFrame, beta: float) -> pd.DataFrame:
    output = current.copy()
    output["S4"] = np.clip(current["S4"] + beta * (s4_full["S4"] - current["S4"]), CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _make_submission(path: Path, predictions: pd.DataFrame) -> None:
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    if predictions.shape != (250, 7):
        raise ValueError(f"Invalid prediction shape: {predictions.shape}")
    values = predictions[TARGET_COLUMNS].to_numpy(float)
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


def _write_artifacts(
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
    oof_export["split_scheme"] = "temporal_s4_beta"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "selection": selection})


def _append_experiment(name: str, scores: dict[str, float], submission: Path, selection: dict[str, Any]) -> None:
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
            "validation_scheme": "temporal_s4_beta",
            "seeds": "posthoc",
            "total_oof_logloss": scores["mean"],
            "target_logloss_Q1": scores["Q1"],
            "target_logloss_Q2": scores["Q2"],
            "target_logloss_Q3": scores["Q3"],
            "target_logloss_S1": scores["S1"],
            "target_logloss_S2": scores["S2"],
            "target_logloss_S3": scores["S3"],
            "target_logloss_S4": scores["S4"],
            "submission_file": str(submission.relative_to(ROOT)),
            "feature_view_by_target": json.dumps(selection, sort_keys=True),
            "notes": "S4-only beta continuation after beta 0.50 improved public.",
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(results: list[dict[str, Any]]) -> None:
    rows = [
        {
            "rank": 0,
            "candidate": CURRENT_BEST_FILE,
            "oof_mean": CURRENT_BEST_OOF,
            "public_score": f"{CURRENT_BEST_PUBLIC:.10f}",
            "submission_file": f"submissions/ready/{CURRENT_BEST_FILE}",
            "notes": "Current public best; S4 beta 0.85 worked.",
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
                "notes": result["notes"],
            }
        )
    rows.extend(
        [
            {
                "rank": len(rows),
                "candidate": "lgb_temporal_targetwise_s4.csv",
                "oof_mean": 0.559687095376793,
                "public_score": "",
                "submission_file": "submissions/ready/lgb_temporal_targetwise_s4.csv",
                "notes": "Full S4 temporal probe; same as beta 1.00 reference.",
            },
            {
                "rank": len(rows) + 1,
                "candidate": "lgb_temporal_targetwise.csv",
                "oof_mean": 0.5598445121450588,
                "public_score": "0.5863944910",
                "submission_file": "submissions/ready/lgb_temporal_targetwise.csv",
                "notes": "Previous public best.",
            },
            {
                "rank": len(rows) + 2,
                "candidate": "lgb_temporal_push.csv",
                "oof_mean": 0.5595372562363035,
                "public_score": "0.5880629390",
                "submission_file": "submissions/archive/2026-05-02_push_failed/lgb_temporal_push.csv",
                "notes": "Failed public; archived, do not resubmit.",
            },
        ]
    )
    _write_csv_rows(LOG_DIR / "candidate_scores.csv", rows, ["rank", "candidate", "oof_mean", "public_score", "submission_file", "notes"])


def _write_readme(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Submissions", "",
        "DACON에 올릴 파일은 `submissions/ready/` 기준으로 보면 됩니다.", "",
        "## Current Best", "",
        f"- Public best: `{CURRENT_BEST_PUBLIC:.10f}`",
        f"- Best file: `submissions/ready/{CURRENT_BEST_FILE}`", "",
        "## Recommended Order", "",
        "| Priority | File | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_BEST_FILE}` | `{CURRENT_BEST_OOF:.6f}` | `{CURRENT_BEST_PUBLIC:.10f}` | 현재 best |",
    ]
    for idx, result in enumerate(results, start=1):
        lines.append(f"| {idx} | `ready/{result['file']}` | `{result['scores']['mean']:.6f}` |  | {result['decision']} |")
    lines.extend(
        [
            f"| {len(results) + 1} | `ready/lgb_temporal_targetwise_s4.csv` | `0.559687` |  | full S4, 고위험 |",
            f"| {len(results) + 2} | `ready/lgb_temporal_targetwise.csv` | `0.559845` | `0.5863944910` | 이전 best |",
            f"| {len(results) + 3} | `archive/2026-05-02_push_failed/lgb_temporal_push.csv` | `0.559537` | `0.5880629390` | 실패, 제출 금지 |",
            "",
            "## Notes",
            "",
        "- `lgb_temporal_s4b085`가 public best를 갱신했으므로 S4 이동은 full 근처까지만 이어갑니다.",
        "- Q1/Q2/Q3/S1/S2/S3는 current best 그대로 고정합니다.",
        "- `lgb_temporal_push` 계열은 public 악화로 중단합니다.",
        ]
    )
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(current_scores: dict[str, float], full_scores: dict[str, float], results: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission Report: temporal_s4_beta_v1", "",
        f"- Current public best: `{CURRENT_BEST_PUBLIC:.10f}` from `{CURRENT_BEST_FILE}`",
        "- Strategy: keep Q1/Q2/Q3/S1/S2/S3 fixed and finish the S4 beta ladder near full S4.",
        "- Primary next file: `submissions/ready/lgb_temporal_s4b095.csv`", "",
        "| File | Beta | OOF mean | S4 OOF | Decision |",
        "|---|---:|---:|---:|---|",
        f"| `{CURRENT_BEST_FILE}` | `0.85` | `{CURRENT_BEST_OOF:.6f}` | `{current_scores['S4']:.6f}` | current best |",
    ]
    for result in results:
        lines.append(
            f"| `{result['file']}` | `{result['beta']:.2f}` | `{result['scores']['mean']:.6f}` | "
            f"`{result['scores']['S4']:.6f}` | {result['decision']} |"
        )
    lines.append(f"| `lgb_temporal_targetwise_s4.csv` | `1.00` | `{full_scores['mean']:.6f}` | `{full_scores['S4']:.6f}` | full S4, highest risk |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Submit `lgb_temporal_s4b095.csv` first.",
            "- Use `lgb_temporal_s4b100.csv` if beta 0.95 improves public.",
            "- Use `lgb_temporal_s4b110.csv` only if full S4 still improves public.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "temporal_s4_beta_v1.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    keys, labels, current_oof, current_test = _load_run(CURRENT_RUN)
    _, _, full_oof, full_test = _load_run(S4_FULL_RUN)

    current_best_oof = _s4_blend(current_oof, full_oof, 0.85)
    current_best_scores = _score_targets(labels, current_best_oof)
    full_scores = _score_targets(labels, full_oof)
    results: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        beta = float(candidate["beta"])
        oof = _s4_blend(current_oof, full_oof, beta)
        test = _s4_blend(current_test, full_test, beta)
        scores = _score_targets(labels, oof)
        path = READY_DIR / str(candidate["file"])
        _make_submission(path, test)
        selection = {"current_run": CURRENT_RUN, "s4_full_run": S4_FULL_RUN, "s4_beta": beta}
        _write_artifacts(str(candidate["run_name"]), keys, labels, oof, test, scores, selection)
        _append_experiment(f"{EXPERIMENT_NAME}_b{int(round(beta * 100)):03d}", scores, path, selection)
        results.append(
            {
                **candidate,
                "scores": scores,
                "notes": f"{candidate['label']}; Q/S fixed, S4-only continuation.",
                "decision": "다음 제출" if beta == 0.95 else ("2순위, full S4" if beta == 1.00 else "고위험, full 초과"),
            }
        )

    _write_candidate_scores(results)
    _write_readme(results)
    _write_report(current_best_scores, full_scores, results)

    print("=== temporal_s4_beta_v1 ===")
    print(f"current: {CURRENT_BEST_FILE} OOF={CURRENT_BEST_OOF:.6f} public={CURRENT_BEST_PUBLIC:.10f}")
    for result in results:
        scores = result["scores"]
        print(f"{result['file']}: beta={result['beta']:.2f} OOF={scores['mean']:.6f} S4={scores['S4']:.6f}")
    print(f"full_reference: lgb_temporal_targetwise_s4.csv OOF={full_scores['mean']:.6f} S4={full_scores['S4']:.6f}")


if __name__ == "__main__":
    main()

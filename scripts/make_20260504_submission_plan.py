#!/usr/bin/env python3
"""Create the 2026-05-04 upload candidates around the current S4 beta best."""

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


EXPERIMENT_NAME = "today_s4_extension_20260504"
ANCHOR_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
S4_FULL_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"

CURRENT_BEST_FILE = "lgb_temporal_s4b110.csv"
CURRENT_BEST_PUBLIC = 0.5847975097
CURRENT_BEST_BETA = 1.10

READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
CLIP_MIN = 0.02
CLIP_MAX = 0.98

CANDIDATES = [
    {
        "beta": 1.15,
        "run_name": "public_lgb_targetwise_today_s4b115_20260504",
        "file": "lgb_temporal_s4b115.csv",
        "decision": "1순위: b110에서 가장 작은 추가 연장",
    },
    {
        "beta": 1.20,
        "run_name": "public_lgb_targetwise_today_s4b120_20260504",
        "file": "lgb_temporal_s4b120.csv",
        "decision": "2순위: b115가 개선될 때만 제출",
    },
    {
        "beta": 1.30,
        "run_name": "public_lgb_targetwise_today_s4b130_20260504",
        "file": "lgb_temporal_s4b130.csv",
        "decision": "3순위: b120까지 개선될 때만 제출, 고위험",
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
    oof_export["split_scheme"] = "today_s4_extension"
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
    selection = result["selection"]
    rows.append(
        {
            "timestamp": _now_kst(),
            "experiment_name": name,
            "validation_scheme": "today_s4_extension",
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
            "feature_view_by_target": json.dumps(selection, sort_keys=True),
            "notes": "2026-05-04 upload candidate: isolated S4 beta extension from the current public best.",
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_candidate_scores(results: list[dict[str, Any]]) -> None:
    rows = [
        {
            "rank": 0,
            "candidate": CURRENT_BEST_FILE,
            "oof_mean": 0.5596888312564241,
            "public_score": f"{CURRENT_BEST_PUBLIC:.10f}",
            "submission_file": f"submissions/ready/{CURRENT_BEST_FILE}",
            "notes": "Current public best; already uploaded.",
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
                "notes": result["decision"],
            }
        )
    rows.extend(
        [
            {
                "rank": len(rows),
                "candidate": "lgb_temporal_s4b100.csv",
                "oof_mean": 0.559687095376793,
                "public_score": "",
                "submission_file": "submissions/ready/lgb_temporal_s4b100.csv",
                "notes": "Fallback diagnostic if b115 worsens; checks whether b110 over-shot the public optimum.",
            },
            {
                "rank": len(rows) + 1,
                "candidate": "lgb_temporal_s4b085.csv",
                "oof_mean": 0.5596903927971973,
                "public_score": "0.5851199626",
                "submission_file": "submissions/ready/lgb_temporal_s4b085.csv",
                "notes": "Previous public best backup.",
            },
            {
                "rank": len(rows) + 2,
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
        "## 2026-05-04 Upload Plan",
        "",
        "오늘은 S4만 더 움직입니다. Q1/Q2/Q3/S1/S2/S3는 `lgb_temporal_s4b110.csv`와 동일하게 고정합니다.",
        "",
        "| Order | File | Beta | OOF mean | Public | Decision |",
        "|---:|---|---:|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_BEST_FILE}` | `{CURRENT_BEST_BETA:.2f}` | `0.559689` | `{CURRENT_BEST_PUBLIC:.10f}` | 현재 best, 기준점 |",
    ]
    for idx, result in enumerate(results, start=1):
        lines.append(
            f"| {idx} | `ready/{result['file']}` | `{result['beta']:.2f}` | "
            f"`{result['scores']['mean']:.6f}` |  | {result['decision']} |"
        )
    lines.extend(
        [
            "| fallback | `ready/lgb_temporal_s4b100.csv` | `1.00` | `0.559687` |  | b115가 악화될 때만 확인 |",
            "| backup | `ready/lgb_temporal_s4b085.csv` | `0.85` | `0.559690` | `0.5851199626` | 이전 best 백업 |",
            "| stop | `archive/2026-05-02_push_failed/lgb_temporal_push.csv` |  | `0.559537` | `0.5880629390` | 제출 금지 |",
            "",
            "## Rules",
            "",
            "1. 먼저 `lgb_temporal_s4b115.csv`를 올립니다.",
            "2. b115가 `0.5847975097`보다 낮으면 `lgb_temporal_s4b120.csv`를 올립니다.",
            "3. b120도 개선되면 마지막으로 `lgb_temporal_s4b130.csv`를 올립니다.",
            "4. b115가 악화되면 추가 확장은 중단하고, 필요할 때만 `lgb_temporal_s4b100.csv`로 과확장 여부를 확인합니다.",
            "",
            "## Notes",
            "",
            "- 공개 점수 개선은 S4-only beta에서만 이어졌습니다.",
            "- `temporal_push`처럼 Q/S까지 같이 민 방향은 public에서 실패했으므로 제외합니다.",
            "- `b130`은 탐색 폭을 남기기 위한 고위험 파일입니다. b115/b120가 모두 좋아질 때만 씁니다.",
        ]
    )
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission Report: today_s4_extension_20260504",
        "",
        f"- Current best: `{CURRENT_BEST_FILE}` public `{CURRENT_BEST_PUBLIC:.10f}`",
        "- Strategy: continue the proven S4-only beta axis with smaller steps beyond beta 1.10.",
        "- Fixed targets: Q1/Q2/Q3/S1/S2/S3.",
        "",
        "| File | Beta | OOF mean | S4 OOF | S4 test mean | Decision |",
        "|---|---:|---:|---:|---:|---|",
        f"| `{CURRENT_BEST_FILE}` | `{CURRENT_BEST_BETA:.2f}` | `0.559689` | `0.612947` | `0.546705` | current best |",
    ]
    for result in results:
        scores = result["scores"]
        test = result["test"]
        lines.append(
            f"| `{result['file']}` | `{result['beta']:.2f}` | `{scores['mean']:.6f}` | "
            f"`{scores['S4']:.6f}` | `{test['S4'].mean():.6f}` | {result['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Upload Logic",
            "",
            "- First upload: `lgb_temporal_s4b115.csv`.",
            "- Continue to `b120` only if `b115` beats `0.5847975097`.",
            "- Continue to `b130` only if `b120` also improves.",
            "- If `b115` is worse, stop extension and optionally test `lgb_temporal_s4b100.csv` as the conservative diagnostic.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "today_s4_extension_20260504.md", "\n".join(lines))


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

        selection = {
            "anchor_run": ANCHOR_RUN,
            "s4_full_run": S4_FULL_RUN,
            "s4_beta": beta,
            "fixed_targets": [target for target in TARGET_COLUMNS if target != "S4"],
        }
        _write_prediction_artifacts(str(candidate["run_name"]), keys, labels, oof, test, scores, selection)
        result = {**candidate, "scores": scores, "path": path, "selection": selection, "test": test}
        _append_experiment(result)
        results.append(result)

    _write_candidate_scores(results)
    _write_readme(results)
    _write_report(results)

    print("=== today_s4_extension_20260504 ===")
    print(f"current: {CURRENT_BEST_FILE} beta={CURRENT_BEST_BETA:.2f} public={CURRENT_BEST_PUBLIC:.10f}")
    for result in results:
        scores = result["scores"]
        print(
            f"{result['file']}: beta={result['beta']:.2f} "
            f"OOF={scores['mean']:.6f} S4={scores['S4']:.6f} "
            f"path={result['path']}"
        )


if __name__ == "__main__":
    main()

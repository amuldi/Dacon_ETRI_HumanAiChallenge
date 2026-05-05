#!/usr/bin/env python3
"""Create non-S4 recovery candidates after S4 directional probes failed."""

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


EXPERIMENT_NAME = "qs_recovery_20260505"
PRIOR_RUN = "public_lgb_targetwise_temporal_prior_v1"
TARGETWISE_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
CURRENT_RUN = "public_lgb_targetwise_toward57_s4b650_20260504"
QSCAL_RUN = "public_lgb_targetwise_postb650_s4b650_qscal_20260504"

CURRENT_BEST_FILE = "lgb_temporal_s4b650.csv"
CURRENT_BEST_PUBLIC = 0.5829008297
CURRENT_BEST_OOF = 0.5652961542103195
FAILED_DOWN_FILE = "submit_s4down650_fixed.csv"
FAILED_DOWN_PUBLIC = 0.5838666368

READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
CLIP_MIN = 0.02
CLIP_MAX = 0.98

CANDIDATES = [
    {
        "file": "submit_qshead160_s4b650.csv",
        "run_name": "public_lgb_targetwise_qs_recovery_head160_s4b650_20260505",
        "kind": "scaled_temporal_axis",
        "scales": {"Q1": 1.60, "Q2": 1.60, "Q3": 1.40, "S1": 1.25, "S2": 1.00, "S3": 1.00},
        "decision": "First upload: push only Q1/Q2/Q3/S1; S2/S3/S4 fixed after public failures.",
    },
    {
        "file": "submit_qshead135_s4b650.csv",
        "run_name": "public_lgb_targetwise_qs_recovery_head135_s4b650_20260505",
        "kind": "scaled_temporal_axis",
        "scales": {"Q1": 1.35, "Q2": 1.35, "Q3": 1.25, "S1": 1.15, "S2": 1.00, "S3": 1.00},
        "decision": "Safer fallback if head160 is too aggressive.",
    },
    {
        "file": "submit_qsmicro_s4b650.csv",
        "run_name": "public_lgb_targetwise_qs_recovery_micro_s4b650_20260505",
        "kind": "scaled_temporal_axis",
        "scales": {"Q1": 1.35, "Q2": 1.35, "Q3": 1.25, "S1": 1.15, "S2": 1.035, "S3": 1.02},
        "decision": "Small all-Q/S push; use only if head-only direction improves.",
    },
    {
        "file": "submit_qscal_s4b650_fixed.csv",
        "run_name": QSCAL_RUN,
        "kind": "existing_qscal",
        "scales": {},
        "decision": "Conservative fallback: keep b650 S4 and apply OOF-fitted Q/S calibration.",
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
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


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


def _apply_scales(
    prior: pd.DataFrame,
    targetwise: pd.DataFrame,
    current: pd.DataFrame,
    scales: dict[str, float],
) -> pd.DataFrame:
    output = current.copy()
    for target, scale in scales.items():
        output[target] = np.clip(prior[target] + float(scale) * (targetwise[target] - prior[target]), CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _write_fixed_submission(path: Path, predictions: pd.DataFrame) -> None:
    template = pd.read_csv(ROOT / "data" / "ch2026_submission_sample.csv", dtype=str)
    if predictions.shape != (250, 7):
        raise ValueError(f"Invalid prediction shape: {predictions.shape}")
    output = template[KEY_COLUMNS].copy()
    values = predictions[TARGET_COLUMNS].astype(float).clip(1e-6, 1.0 - 1e-6)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    for target in TARGET_COLUMNS:
        output[target] = values[target].map(lambda value: f"{value:.10f}")
    output = output[template.columns]
    if output.shape != template.shape:
        raise ValueError(f"Invalid submission shape: {output.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, lineterminator="\n")


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
    oof_export["split_scheme"] = "qs_recovery"
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
    rows = [row for row in rows if row.get("experiment_name") != result["experiment_name"]]
    scores = result["scores"]
    rows.append(
        {
            "timestamp": _now_kst(),
            "experiment_name": result["experiment_name"],
            "validation_scheme": "qs_recovery",
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
            "notes": result["decision"],
        }
    )
    _write_csv_rows(path, rows, columns)


def _write_public_scores() -> None:
    path = LOG_DIR / "public_scores.csv"
    columns = ["timestamp", "submission_file", "experiment_name", "public_score", "delta_vs_best", "notes"]
    rows = _read_csv_rows(path)
    rows = [row for row in rows if row.get("submission_file") != FAILED_DOWN_FILE]
    rows.append(
        {
            "timestamp": "2026-05-05",
            "submission_file": FAILED_DOWN_FILE,
            "experiment_name": "post_b650_strategy_20260504_s4down650",
            "public_score": f"{FAILED_DOWN_PUBLIC:.10f}",
            "delta_vs_best": f"{FAILED_DOWN_PUBLIC - CURRENT_BEST_PUBLIC:.10f}",
            "notes": "Worse than current best; stop S4 directional probes and switch to Q/S recovery.",
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
            "notes": "Current public best.",
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
                "candidate": FAILED_DOWN_FILE,
                "oof_mean": 0.5621850602177,
                "public_score": f"{FAILED_DOWN_PUBLIC:.10f}",
                "submission_file": "submissions/archive/2026-05-05_s4_direction_failed/submit_s4down650_fixed.csv",
                "notes": "Failed public; do not resubmit.",
            },
            {
                "rank": len(rows) + 1,
                "candidate": "lgb_temporal_s4b130.csv",
                "oof_mean": 0.5597017230477448,
                "public_score": "0.5845552904",
                "submission_file": "submissions/ready/lgb_temporal_s4b130.csv",
                "notes": "Previous best backup.",
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
        "## 2026-05-05 Q/S Recovery Plan",
        "",
        "`s4up650`과 `s4down650`이 모두 악화됐으므로 S4 방향성 실험은 중단합니다. current best의 S4는 고정하고 Q1/Q2/Q3/S1만 제한적으로 다시 움직입니다.",
        "",
        "| Order | File | OOF mean | S4 mean | Decision |",
        "|---:|---|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_BEST_FILE}` | `{CURRENT_BEST_OOF:.6f}` | `0.554090` | 현재 best |",
    ]
    for idx, result in enumerate(results, start=1):
        lines.append(
            f"| {idx} | `ready/{result['file']}` | `{result['scores']['mean']:.6f}` | "
            f"`{result['test']['S4'].mean():.6f}` | {result['decision']} |"
        )
    lines.extend(
        [
            f"| failed | `archive/2026-05-05_s4_direction_failed/{FAILED_DOWN_FILE}` | `0.562185` | `0.487833` | public `{FAILED_DOWN_PUBLIC:.10f}`, 제출 금지 |",
            "",
            "## Rules",
            "",
            "1. 먼저 `submit_qshead160_s4b650.csv`를 올립니다.",
            "2. 이 파일이 `0.5829008297`보다 좋아지면 Q/S head push 방향을 더 세분화합니다.",
            "3. 악화되면 `submit_qshead135_s4b650.csv`로 강도를 낮춥니다.",
            "4. Q/S push도 실패하면 `submit_qscal_s4b650_fixed.csv`만 확인하고, 새 모델/새 feature 축으로 넘어갑니다.",
        ]
    )
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission Report: qs_recovery_20260505",
        "",
        f"- Current best: `{CURRENT_BEST_FILE}` public `{CURRENT_BEST_PUBLIC:.10f}`",
        f"- Failed latest: `{FAILED_DOWN_FILE}` public `{FAILED_DOWN_PUBLIC:.10f}`",
        "- Strategy: freeze S4 and probe a constrained non-S4 temporal-prior axis.",
        "",
        "| File | OOF mean | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        scores = result["scores"]
        lines.append(
            f"| `{result['file']}` | `{scores['mean']:.6f}` | `{scores['Q1']:.6f}` | "
            f"`{scores['Q2']:.6f}` | `{scores['Q3']:.6f}` | `{scores['S1']:.6f}` | "
            f"`{scores['S2']:.6f}` | `{scores['S3']:.6f}` | `{scores['S4']:.6f}` | {result['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Rationale",
            "",
            "- Full Q/S over-push failed earlier, so S2/S3 are frozen in the first two files.",
            "- The first file moves only Q1/Q2/Q3/S1 where OOF still improves along the temporal axis.",
            "- All output CSVs use fixed-format sample keys and 10-decimal probabilities to avoid submission Data Error.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "qs_recovery_20260505.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    keys, labels, prior_oof, prior_test = _load_run(PRIOR_RUN)
    _, _, targetwise_oof, targetwise_test = _load_run(TARGETWISE_RUN)
    _, _, current_oof, current_test = _load_run(CURRENT_RUN)
    _, _, qscal_oof, qscal_test = _load_run(QSCAL_RUN)

    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        if candidate["kind"] == "existing_qscal":
            oof = qscal_oof.copy()
            test = qscal_test.copy()
            selection = {"source_run": QSCAL_RUN}
        else:
            scales = dict(candidate["scales"])
            oof = _apply_scales(prior_oof, targetwise_oof, current_oof, scales)
            test = _apply_scales(prior_test, targetwise_test, current_test, scales)
            selection = {
                "prior_run": PRIOR_RUN,
                "targetwise_run": TARGETWISE_RUN,
                "current_run": CURRENT_RUN,
                "scales": scales,
                "fixed_targets": [target for target in TARGET_COLUMNS if target not in scales],
            }
            _write_prediction_artifacts(str(candidate["run_name"]), keys, labels, oof, test, _score_targets(labels, oof), selection)

        scores = _score_targets(labels, oof)
        path = READY_DIR / str(candidate["file"])
        _write_fixed_submission(path, test)
        result = {
            **candidate,
            "path": path,
            "scores": scores,
            "selection": selection,
            "test": test,
            "experiment_name": f"{EXPERIMENT_NAME}_{str(candidate['file']).replace('submit_', '').replace('.csv', '')}",
        }
        _append_experiment(result)
        results.append(result)

    _write_public_scores()
    _write_candidate_scores(results)
    _write_readme(results)
    _write_report(results)

    print("=== qs_recovery_20260505 ===")
    print(f"current: {CURRENT_BEST_FILE} public={CURRENT_BEST_PUBLIC:.10f} OOF={CURRENT_BEST_OOF:.6f}")
    for result in results:
        print(
            f"{result['file']}: OOF={result['scores']['mean']:.6f} "
            f"S4_mean={result['test']['S4'].mean():.6f} path={result['path']}"
        )


if __name__ == "__main__":
    main()

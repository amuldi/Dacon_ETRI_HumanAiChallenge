#!/usr/bin/env python3
"""Create post-b650 candidates after the S4-only trend flattened."""

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


EXPERIMENT_NAME = "post_b650_strategy_20260504"
ANCHOR_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
S4_FULL_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"

CURRENT_BEST_FILE = "lgb_temporal_s4b650.csv"
CURRENT_BEST_PUBLIC = 0.5829008297
CURRENT_BEST_OOF = 0.5652961542103195
PREVIOUS_BEST_PUBLIC = 0.5845552904

READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
CLIP_MIN = 0.02
CLIP_MAX = 0.98

QS_CAL_PARAMS = {
    "Q1": {"temp": 1.15, "bias": 0.00},
    "Q2": {"temp": 1.05, "bias": 0.00},
    "Q3": {"temp": 1.05, "bias": -0.025},
    "S1": {"temp": 1.05, "bias": -0.025},
    "S2": {"temp": 1.10, "bias": -0.075},
    "S3": {"temp": 1.10, "bias": -0.050},
}

CANDIDATES = [
    {
        "file": "lgb_temporal_s4up650.csv",
        "run_name": "public_lgb_targetwise_postb650_s4up650_20260504",
        "mode": "s4_directional",
        "s4_mode": "up",
        "beta": 6.50,
        "decision": "First submit: decomposes b650 and tests whether the public gain came from raising S4 only.",
    },
    {
        "file": "lgb_temporal_s4b1200.csv",
        "run_name": "public_lgb_targetwise_postb650_s4b1200_20260504",
        "mode": "s4_directional",
        "s4_mode": "symmetric",
        "beta": 12.00,
        "decision": "Second submit only if S4 still looks alive; symmetric plateau check.",
    },
    {
        "file": "lgb_temporal_s4b650_qscal.csv",
        "run_name": "public_lgb_targetwise_postb650_s4b650_qscal_20260504",
        "mode": "s4_b650_qs_cal",
        "s4_mode": "symmetric",
        "beta": 6.50,
        "decision": "Non-S4 fallback: keep b650 S4 and add conservative OOF calibration on Q/S targets.",
    },
    {
        "file": "lgb_temporal_s4down650.csv",
        "run_name": "public_lgb_targetwise_postb650_s4down650_20260504",
        "mode": "s4_directional",
        "s4_mode": "down",
        "beta": 6.50,
        "decision": "Diagnostic only if up650 fails; tests whether lowering S4 rows was the useful half.",
    },
    {
        "file": "lgb_temporal_s4b1800.csv",
        "run_name": "public_lgb_targetwise_postb650_s4b1800_20260504",
        "mode": "s4_directional",
        "s4_mode": "symmetric",
        "beta": 18.00,
        "decision": "Ceiling probe only if b1200 improves meaningfully; high clipping risk.",
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


def _logit(values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _transform(values: np.ndarray | pd.Series, temp: float, bias: float) -> np.ndarray:
    return np.clip(_sigmoid(temp * _logit(values) + bias), CLIP_MIN, CLIP_MAX)


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _apply_s4(anchor: pd.DataFrame, s4_full: pd.DataFrame, beta: float, s4_mode: str) -> pd.DataFrame:
    output = anchor.copy()
    delta = s4_full["S4"] - anchor["S4"]
    if s4_mode == "symmetric":
        step = delta
    elif s4_mode == "up":
        step = np.maximum(delta, 0.0)
    elif s4_mode == "down":
        step = np.minimum(delta, 0.0)
    else:
        raise ValueError(f"Unsupported S4 mode: {s4_mode}")
    output["S4"] = np.clip(anchor["S4"] + beta * step, CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _apply_qs_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for target, params in QS_CAL_PARAMS.items():
        output[target] = _transform(frame[target], params["temp"], params["bias"])
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
    oof_export["split_scheme"] = "post_b650_strategy"
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
    name = str(result["experiment_name"])
    rows = [row for row in rows if row.get("experiment_name") != name]
    scores = result["scores"]
    rows.append(
        {
            "timestamp": _now_kst(),
            "experiment_name": name,
            "validation_scheme": "post_b650_strategy",
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
    rows = [row for row in rows if row.get("submission_file") != CURRENT_BEST_FILE]
    rows.append(
        {
            "timestamp": "2026-05-04",
            "submission_file": CURRENT_BEST_FILE,
            "experiment_name": "toward_57_s4_ladder_20260504_b650",
            "public_score": f"{CURRENT_BEST_PUBLIC:.10f}",
            "delta_vs_best": f"{CURRENT_BEST_PUBLIC - PREVIOUS_BEST_PUBLIC:.10f}",
            "notes": "New current best, but S4-only gain flattened; switch to directional and non-S4 probes.",
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
            "notes": "Current public best; S4-only extension improved but flattened.",
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
                "candidate": "lgb_temporal_s4b130.csv",
                "oof_mean": 0.5597017230477448,
                "public_score": "0.5845552904",
                "submission_file": "submissions/ready/lgb_temporal_s4b130.csv",
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
        "## 2026-05-04 Post-b650 Plan",
        "",
        "b650은 개선됐지만 S4-only가 포화됐습니다. 이제 목표는 S4 개선분을 분해하고, 비-S4 보정 축을 동시에 여는 것입니다.",
        "",
        "| Order | File | Mode | OOF mean | S4 mean | Clip low/high | Decision |",
        "|---:|---|---|---:|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_BEST_FILE}` | current | `{CURRENT_BEST_OOF:.6f}` | `0.554090` | `0/7` | 현재 best |",
    ]
    for idx, result in enumerate(results, start=1):
        stats = result["stats"]
        lines.append(
            f"| {idx} | `ready/{result['file']}` | `{result['mode_label']}` | "
            f"`{result['scores']['mean']:.6f}` | `{stats['s4_mean']:.6f}` | "
            f"`{stats['clip_low']}/{stats['clip_high']}` | {result['decision']} |"
        )
    lines.extend(
        [
            "| backup | `ready/lgb_temporal_s4b130.csv` | previous | `0.559702` | `0.547018` | `0/0` | 이전 best 백업 |",
            "| stop | `archive/2026-05-02_push_failed/lgb_temporal_push.csv` | failed | `0.559537` |  |  | 제출 금지 |",
            "",
            "## Rules",
            "",
            "1. 먼저 `lgb_temporal_s4up650.csv`를 올립니다.",
            "2. up650이 개선되면 이후는 up-only ladder로 전환합니다.",
            "3. up650이 악화되면 `lgb_temporal_s4b1200.csv`로 symmetric 잔여 개선폭만 확인합니다.",
            "4. S4 후보가 모두 포화되면 `lgb_temporal_s4b650_qscal.csv`로 Q/S 보정축을 엽니다.",
            "5. `lgb_temporal_s4b1800.csv`는 b1200이 의미 있게 개선될 때만 씁니다.",
        ]
    )
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(results: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission Report: post_b650_strategy_20260504",
        "",
        f"- New current best: `{CURRENT_BEST_FILE}` public `{CURRENT_BEST_PUBLIC:.10f}`",
        "- b650 improved, but the observed gain is far above the old linear projection and now looks saturated.",
        "- A S4-only path probably cannot reach 0.57 by itself; we need directional S4 diagnostics plus a small Q/S calibration branch.",
        "",
        "| File | Mode | Beta | OOF mean | S4 OOF | S4 test mean | Min | Max | Clip low | Clip high | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| `{CURRENT_BEST_FILE}` | current | `6.50` | `{CURRENT_BEST_OOF:.6f}` | `0.652198` | `0.554090` | `0.071653` | `0.980000` | `0` | `7` | current best |",
    ]
    for result in results:
        stats = result["stats"]
        scores = result["scores"]
        lines.append(
            f"| `{result['file']}` | `{result['mode_label']}` | `{result['beta']:.2f}` | "
            f"`{scores['mean']:.6f}` | `{scores['S4']:.6f}` | `{stats['s4_mean']:.6f}` | "
            f"`{stats['s4_min']:.6f}` | `{stats['s4_max']:.6f}` | `{stats['clip_low']}` | "
            f"`{stats['clip_high']}` | {result['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `s4up650` and `s4down650` split the b650 move into its two halves.",
            "- `s4b1200` checks whether symmetric S4 has any useful residual slope.",
            "- `s4b650_qscal` keeps the current best S4 and only applies conservative OOF-fitted Q/S calibration.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "post_b650_strategy_20260504.md", "\n".join(lines))


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()
    keys, labels, anchor_oof, anchor_test = _load_run(ANCHOR_RUN)
    _, _, full_oof, full_test = _load_run(S4_FULL_RUN)

    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        beta = float(candidate["beta"])
        oof = _apply_s4(anchor_oof, full_oof, beta, str(candidate["s4_mode"]))
        test = _apply_s4(anchor_test, full_test, beta, str(candidate["s4_mode"]))
        if candidate["mode"] == "s4_b650_qs_cal":
            oof = _apply_qs_calibration(oof)
            test = _apply_qs_calibration(test)
        scores = _score_targets(labels, oof)
        path = READY_DIR / str(candidate["file"])
        _make_submission(path, test)
        s4_values = test["S4"].to_numpy(float)
        selection = {
            "anchor_run": ANCHOR_RUN,
            "s4_full_run": S4_FULL_RUN,
            "s4_beta": beta,
            "s4_mode": candidate["s4_mode"],
            "qs_cal_params": QS_CAL_PARAMS if candidate["mode"] == "s4_b650_qs_cal" else {},
        }
        _write_prediction_artifacts(str(candidate["run_name"]), keys, labels, oof, test, scores, selection)
        experiment_name = f"{EXPERIMENT_NAME}_{str(candidate['file']).replace('lgb_temporal_', '').replace('.csv', '')}"
        result = {
            **candidate,
            "path": path,
            "scores": scores,
            "selection": selection,
            "experiment_name": experiment_name,
            "mode_label": "S4 " + str(candidate["s4_mode"]) if candidate["mode"] == "s4_directional" else "S4 b650 + Q/S cal",
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

    print("=== post_b650_strategy_20260504 ===")
    print(f"current: {CURRENT_BEST_FILE} public={CURRENT_BEST_PUBLIC:.10f} OOF={CURRENT_BEST_OOF:.6f}")
    for result in results:
        stats = result["stats"]
        print(
            f"{result['file']}: mode={result['mode_label']} beta={result['beta']:.2f} "
            f"OOF={result['scores']['mean']:.6f} S4={result['scores']['S4']:.6f} "
            f"S4_mean={stats['s4_mean']:.6f} clip={stats['clip_low']}/{stats['clip_high']} "
            f"path={result['path']}"
        )


if __name__ == "__main__":
    main()

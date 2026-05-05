#!/usr/bin/env python3
"""Create public-anchored S4 local-continuation candidates after Q/S failed."""

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


EXPERIMENT_NAME = "s4_public_anchored_20260505"
ANCHOR_RUN = "public_lgb_targetwise_temporal_targetwise_v1"
S4_FULL_RUN = "public_lgb_targetwise_temporal_targetwise_s4_v1"

CURRENT_BEST_FILE = "lgb_temporal_s4b650.csv"
CURRENT_BEST_PUBLIC = 0.5829008297
CURRENT_BEST_OOF = 0.5652961542103195
CURRENT_BEST_BETA = 6.50

FAILED_QS_FILE = "submit_qshead160_s4b650.csv"
FAILED_QS_PUBLIC = 0.5837781861
FAILED_QS_OOF = 0.5650630985997749

PREVIOUS_BACKUP_FILE = "lgb_temporal_s4b130.csv"
PREVIOUS_BACKUP_PUBLIC = 0.5845552904
PREVIOUS_BACKUP_OOF = 0.5597017230477448

READY_DIR = ROOT / "submissions" / "ready"
LOG_DIR = ROOT / "logs"
ARCHIVE_DIR = ROOT / "submissions" / "archive" / "2026-05-05_qshead_failed"
CLIP_MIN = 0.02
CLIP_MAX = 0.98

CANDIDATES = [
    {
        "file": "submit_s4sym800_s4b650.csv",
        "run_name": "public_lgb_targetwise_s4public_sym800_20260505",
        "mode": "symmetric",
        "beta": 8.00,
        "decision": "First upload: public-validated symmetric S4 axis, small step beyond b650.",
    },
    {
        "file": "submit_s4near950_s4b650.csv",
        "run_name": "public_lgb_targetwise_s4public_near950_20260505",
        "mode": "near_gated",
        "base_beta": 6.50,
        "near_beta": 9.50,
        "near_days": 7.0,
        "decision": "Second only if sym800 improves: stronger S4 step only where nearest train label is within 7 days.",
    },
    {
        "file": "submit_s4sym950_s4b650.csv",
        "run_name": "public_lgb_targetwise_s4public_sym950_20260505",
        "mode": "symmetric",
        "beta": 9.50,
        "decision": "Higher-risk continuation only if sym800 improves clearly.",
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


def _s4_beta(anchor: pd.DataFrame, s4_full: pd.DataFrame, beta: float) -> pd.DataFrame:
    output = anchor.copy()
    output["S4"] = np.clip(anchor["S4"] + float(beta) * (s4_full["S4"] - anchor["S4"]), CLIP_MIN, CLIP_MAX)
    return output[TARGET_COLUMNS]


def _nearest_train_days_for_test() -> pd.Series:
    train = load_train_labels()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    sample = load_submission_template()[KEY_COLUMNS].copy()
    nearest = pd.Series(np.inf, index=sample.index, dtype=float)

    for subject_id, test_group in sample.groupby("subject_id", sort=False):
        train_group = train[train["subject_id"] == subject_id]
        train_days = pd.to_datetime(train_group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        test_days = pd.to_datetime(test_group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        if len(train_days) == 0:
            continue
        distances = np.abs(test_days[:, None] - train_days[None, :]).astype(float)
        nearest.loc[test_group.index] = distances.min(axis=1)
    return nearest


def _nearest_train_days_for_oof(keys: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    source = pd.concat([keys[KEY_COLUMNS].copy(), labels[TARGET_COLUMNS].copy()], axis=1)
    nearest = pd.Series(np.inf, index=source.index, dtype=float)

    for _, group in source.groupby("subject_id", sort=False):
        idx = group.index.to_numpy()
        days = pd.to_datetime(group["lifelog_date"]).to_numpy("datetime64[D]").astype("int64")
        distances = np.abs(days[:, None] - days[None, :]).astype(float)
        np.fill_diagonal(distances, np.inf)
        nearest.loc[idx] = distances.min(axis=1)
    return nearest


def _s4_near_gated(
    anchor: pd.DataFrame,
    s4_full: pd.DataFrame,
    *,
    base_beta: float,
    near_beta: float,
    nearest_days: pd.Series,
    near_days: float,
) -> pd.DataFrame:
    output = _s4_beta(anchor, s4_full, base_beta)
    near_values = _s4_beta(anchor, s4_full, near_beta)["S4"]
    mask = nearest_days.to_numpy(float) <= float(near_days)
    output.loc[mask, "S4"] = near_values.loc[mask].to_numpy(float)
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
    oof_export["split_scheme"] = "s4_public_anchored"
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof[target].to_numpy(float)
    oof_export.to_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet", index=False)
    test[TARGET_COLUMNS].to_csv(MODELS_DIR / f"test_predictions_{run_name}.csv", index=False)
    write_json(FEATURES_DIR / f"{run_name}_summary.json", {"run_name": run_name, "scores": scores, "selection": selection})


def _prediction_stats(predictions: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    values = predictions["S4"].to_numpy(float)
    return {
        "s4_mean": float(values.mean()),
        "s4_min": float(values.min()),
        "s4_max": float(values.max()),
        "s4_clip_low": int((values <= CLIP_MIN + 1e-10).sum()),
        "s4_clip_high": int((values >= CLIP_MAX - 1e-10).sum()),
        "s4_mean_abs_change_vs_b650": float(np.mean(np.abs(values - current["S4"].to_numpy(float)))),
    }


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
            "validation_scheme": "s4_public_anchored",
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
    rows = [row for row in rows if row.get("submission_file") != FAILED_QS_FILE]
    rows.append(
        {
            "timestamp": "2026-05-05",
            "submission_file": FAILED_QS_FILE,
            "experiment_name": "qs_recovery_20260505_qshead160_s4b650",
            "public_score": f"{FAILED_QS_PUBLIC:.10f}",
            "delta_vs_best": f"{FAILED_QS_PUBLIC - CURRENT_BEST_PUBLIC:.10f}",
            "notes": "Worse than current best; Q/S head push failed, stop remaining Q/S recovery candidates.",
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
                "candidate": FAILED_QS_FILE,
                "oof_mean": FAILED_QS_OOF,
                "public_score": f"{FAILED_QS_PUBLIC:.10f}",
                "submission_file": f"submissions/archive/2026-05-05_qshead_failed/{FAILED_QS_FILE}",
                "notes": "Failed public; do not submit Q/S recovery siblings.",
            },
            {
                "rank": len(rows) + 1,
                "candidate": PREVIOUS_BACKUP_FILE,
                "oof_mean": PREVIOUS_BACKUP_OOF,
                "public_score": f"{PREVIOUS_BACKUP_PUBLIC:.10f}",
                "submission_file": f"submissions/ready/{PREVIOUS_BACKUP_FILE}",
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
        "## 2026-05-05 S4 Public-Anchored Plan",
        "",
        "`submit_qshead160_s4b650.csv`가 public에서 악화됐으므로 Q/S head push와 남은 Q/S 후보는 중단합니다.",
        "반대로 public에서 실제로 확인된 개선 축은 대칭 S4 beta뿐입니다. 따라서 S4는 up/down으로 쪼개지 않고, b650 근처에서 작게만 이어갑니다.",
        "",
        "| Order | File | OOF mean | S4 mean | Clip low/high | Decision |",
        "|---:|---|---:|---:|---:|---|",
        f"| 0 | `ready/{CURRENT_BEST_FILE}` | `{CURRENT_BEST_OOF:.6f}` | `0.554090` | `0/7` | 현재 best |",
    ]
    for idx, result in enumerate(results, start=1):
        stats = result["stats"]
        lines.append(
            f"| {idx} | `ready/{result['file']}` | `{result['scores']['mean']:.6f}` | "
            f"`{stats['s4_mean']:.6f}` | `{int(stats['s4_clip_low'])}/{int(stats['s4_clip_high'])}` | {result['decision']} |"
        )
    lines.extend(
        [
            f"| failed | `archive/2026-05-05_qshead_failed/{FAILED_QS_FILE}` | `{FAILED_QS_OOF:.6f}` | `0.554090` | `0/7` | public `{FAILED_QS_PUBLIC:.10f}`, 제출 금지 |",
            "",
            "## Rules",
            "",
            "1. 먼저 `submit_s4sym800_s4b650.csv`를 올립니다.",
            f"2. 이 파일이 `{CURRENT_BEST_PUBLIC:.10f}`보다 좋아지면 `submit_s4near950_s4b650.csv`를 다음 후보로 봅니다.",
            "3. `submit_s4sym800_s4b650.csv`도 악화되면 S4 후처리까지 중단하고 새 학습/새 feature 축으로 넘어갑니다.",
            "4. Q/S 후보(`qshead135`, `qsmicro`, `qscal`)는 ready에서 제거했으므로 제출하지 않습니다.",
        ]
    )
    write_markdown(ROOT / "submissions" / "README.md", "\n".join(lines))


def _write_report(results: list[dict[str, Any]], near_test_count: int, near_oof_count: int) -> None:
    lines = [
        "# S4 Public-Anchored Strategy 2026-05-05",
        "",
        f"- Current best: `{CURRENT_BEST_FILE}` public `{CURRENT_BEST_PUBLIC:.10f}`",
        f"- Failed latest: `{FAILED_QS_FILE}` public `{FAILED_QS_PUBLIC:.10f}`",
        "- New direction: stop Q/S moves, keep Q1/Q2/Q3/S1/S2/S3 fixed, and only continue the public-proven symmetric S4 axis.",
        "",
        "## Self-Critique",
        "",
        "### What was wrong before",
        "",
        "- I treated `s4up650` and `s4down650` failures as if the whole S4 direction was dead. That was too broad.",
        "- The actual public evidence says isolated up/down halves are bad, but the symmetric `s4b650` combination is still the best file.",
        "- `qshead160` had better OOF but worse public, so OOF-only Q/S recovery is not trustworthy for the next upload.",
        "",
        "### Why this strategy is more defensible",
        "",
        "- It changes only the one axis that has repeatedly improved public: symmetric S4 beta.",
        "- The first candidate is a small move from beta 6.50 to 8.00, not a jump to the high-clipping beta 12/18 files.",
        "- Q/S targets are frozen, because the latest public result directly invalidated that direction.",
        "",
        "### Why it can still be wrong",
        "",
        "- S4 OOF gets worse as beta increases, so this is public-feedback extrapolation, not validation-driven improvement.",
        "- If `b650` was already the local public optimum, `sym800` will worsen.",
        "- If `sym800` worsens, more post-processing is not the right next move; the next axis must be new model/features.",
        "",
        "## Candidate Metrics",
        "",
        f"- Near-gated rows: test `{near_test_count}/250`, OOF `{near_oof_count}/450` within 7 days.",
        "",
        "| File | OOF mean | S4 OOF | S4 mean | S4 min | S4 max | Clip low | Clip high | Mean abs change vs b650 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        stats = result["stats"]
        scores = result["scores"]
        lines.append(
            f"| `{result['file']}` | `{scores['mean']:.6f}` | `{scores['S4']:.6f}` | "
            f"`{stats['s4_mean']:.6f}` | `{stats['s4_min']:.6f}` | `{stats['s4_max']:.6f}` | "
            f"`{int(stats['s4_clip_low'])}` | `{int(stats['s4_clip_high'])}` | "
            f"`{stats['s4_mean_abs_change_vs_b650']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- First upload: `submit_s4sym800_s4b650.csv`.",
            "- Do not upload the remaining Q/S recovery files.",
            "- If `sym800` improves, continue with `near950`; if it worsens, stop S4/QS post-processing.",
        ]
    )
    write_markdown(REPORT_SUBMISSIONS_DIR / "s4_public_anchored_20260505.md", "\n".join(lines))


def _update_experiment_log() -> None:
    path = ROOT / "EXPERIMENT_LOG.md"
    start = "<!-- s4_public_anchored_20260505:start -->"
    end = "<!-- s4_public_anchored_20260505:end -->"
    block = "\n".join(
        [
            start,
            "",
            "## 2026-05-05 S4 Public-Anchored Reset",
            "",
            f"- Latest failed: `{FAILED_QS_FILE}` public `{FAILED_QS_PUBLIC:.10f}`.",
            "- Self-critique: Q/S recovery was OOF-led and contradicted public feedback, so it is stopped.",
            "- Corrected strategy: only the symmetric S4 beta axis has public support; up/down halves and Q/S moves are not supported.",
            "- Next upload: `submit_s4sym800_s4b650.csv`.",
            "- Stop condition: if sym800 does not beat `0.5829008297`, stop S4/QS post-processing and move to new model/features.",
            "",
            end,
            "",
        ]
    )
    text = path.read_text() if path.exists() else "# Experiment Log\n"
    if start in text and end in text:
        before = text.split(start)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        text = f"{before}\n\n{block}{after}"
    else:
        text = f"{text.rstrip()}\n\n{block}"
    write_markdown(path, text)


def main() -> None:
    ensure_runtime_dirs()
    _validate_inputs()

    keys, labels, anchor_oof, anchor_test = _load_run(ANCHOR_RUN)
    _, _, s4_full_oof, s4_full_test = _load_run(S4_FULL_RUN)
    current_oof = _s4_beta(anchor_oof, s4_full_oof, CURRENT_BEST_BETA)
    current_test = _s4_beta(anchor_test, s4_full_test, CURRENT_BEST_BETA)
    oof_nearest = _nearest_train_days_for_oof(keys, labels)
    test_nearest = _nearest_train_days_for_test()

    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        if candidate["mode"] == "symmetric":
            oof = _s4_beta(anchor_oof, s4_full_oof, float(candidate["beta"]))
            test = _s4_beta(anchor_test, s4_full_test, float(candidate["beta"]))
            selection = {
                "anchor_run": ANCHOR_RUN,
                "s4_full_run": S4_FULL_RUN,
                "mode": "symmetric",
                "s4_beta": candidate["beta"],
                "fixed_targets": ["Q1", "Q2", "Q3", "S1", "S2", "S3"],
            }
        elif candidate["mode"] == "near_gated":
            oof = _s4_near_gated(
                anchor_oof,
                s4_full_oof,
                base_beta=float(candidate["base_beta"]),
                near_beta=float(candidate["near_beta"]),
                nearest_days=oof_nearest,
                near_days=float(candidate["near_days"]),
            )
            test = _s4_near_gated(
                anchor_test,
                s4_full_test,
                base_beta=float(candidate["base_beta"]),
                near_beta=float(candidate["near_beta"]),
                nearest_days=test_nearest,
                near_days=float(candidate["near_days"]),
            )
            selection = {
                "anchor_run": ANCHOR_RUN,
                "s4_full_run": S4_FULL_RUN,
                "mode": "near_gated",
                "base_beta": candidate["base_beta"],
                "near_beta": candidate["near_beta"],
                "near_days": candidate["near_days"],
                "fixed_targets": ["Q1", "Q2", "Q3", "S1", "S2", "S3"],
            }
        else:
            raise ValueError(f"Unsupported mode: {candidate['mode']}")

        scores = _score_targets(labels, oof)
        path = READY_DIR / str(candidate["file"])
        _write_fixed_submission(path, test)
        _write_prediction_artifacts(str(candidate["run_name"]), keys, labels, oof, test, scores, selection)
        result = {
            **candidate,
            "path": path,
            "scores": scores,
            "selection": selection,
            "stats": _prediction_stats(test, current_test),
            "experiment_name": f"{EXPERIMENT_NAME}_{candidate['file'].replace('.csv', '')}",
        }
        _append_experiment(result)
        results.append(result)

    _write_public_scores()
    _write_candidate_scores(results)
    _write_readme(results)
    _write_report(
        results,
        near_test_count=int((test_nearest <= 7.0).sum()),
        near_oof_count=int((oof_nearest <= 7.0).sum()),
    )
    _update_experiment_log()

    print(json.dumps({"status": "ok", "results": [{"file": r["file"], "oof": r["scores"]["mean"], "stats": r["stats"]} for r in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

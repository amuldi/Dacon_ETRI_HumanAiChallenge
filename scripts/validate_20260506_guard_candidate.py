#!/usr/bin/env python3
"""Validate whether a candidate run may produce a 2026-05-06 CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

import numpy as np
import pandas as pd

from etri_human_challenge.constants import TARGET_COLUMNS
from etri_human_challenge.paths import MODELS_DIR, OOF_DIR
from etri_human_challenge.utils import binary_log_loss


CURRENT_RUN = "public_lgb_targetwise_toward57_s4b650_20260504"
CURRENT_PUBLIC = 0.5829008297
MIN_NON_S4_IMPROVED = 2
MAX_TARGET_LOSS = 0.00005
MAX_TARGET_TEST_DRIFT = 0.005
FORBIDDEN_NAME_TOKENS = [
    "s4sym",
    "s4near",
    "s4up",
    "s4down",
    "qshead",
    "qsmicro",
    "qscal",
    "guard003",
    "guard005",
]


def _prediction_columns(frame: pd.DataFrame, *, allow_plain_targets: bool) -> pd.DataFrame:
    output: dict[str, pd.Series] = {}
    for target in TARGET_COLUMNS:
        candidates = [
            f"{target}_public_lgb",
            f"{target}_xgb",
            f"{target}_lgb",
            f"{target}_catboost",
            f"{target}_prior_v2",
        ]
        if allow_plain_targets:
            candidates.append(target)
        for column in candidates:
            if column in frame.columns:
                output[target] = frame[column].astype(float)
                break
        else:
            raise ValueError(f"Could not find prediction column for {target}.")
    return pd.DataFrame(output)[TARGET_COLUMNS]


def _load_oof(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(OOF_DIR / f"oof_predictions_{run_name}.parquet")
    labels = raw[TARGET_COLUMNS].astype(float).copy()
    preds = _prediction_columns(raw, allow_plain_targets=False)
    return labels, preds


def _load_test(run_name: str) -> pd.DataFrame:
    raw = pd.read_csv(MODELS_DIR / f"test_predictions_{run_name}.csv")
    return _prediction_columns(raw, allow_plain_targets=True)


def _score_targets(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(float), predictions[target].to_numpy(float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def _collect_numeric_weights(payload: Any) -> list[float]:
    if isinstance(payload, bool):
        return []
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, list):
        output: list[float] = []
        for item in payload:
            output.extend(_collect_numeric_weights(item))
        return output
    if isinstance(payload, dict):
        output = []
        for value in payload.values():
            output.extend(_collect_numeric_weights(value))
        return output
    return []


def _load_weight_check(weights_json: str | None) -> dict[str, Any]:
    if not weights_json:
        return {"provided": False, "non_zero_count": None, "all_zero": None}
    payload = json.loads(Path(weights_json).read_text())
    if isinstance(payload, dict):
        for key in ["weights", "selected_weights", "blend_weights", "optimal_weights"]:
            if key in payload:
                payload = payload[key]
                break
    values = _collect_numeric_weights(payload)
    non_zero_count = int(sum(abs(value) > 1e-12 for value in values))
    return {
        "provided": True,
        "weight_count": len(values),
        "non_zero_count": non_zero_count,
        "all_zero": non_zero_count == 0,
    }


def _guard_candidate(
    candidate_run: str,
    candidate_name: str,
    allow_forbidden_name: bool,
    weights_json: str | None,
) -> dict[str, Any]:
    labels, current_oof = _load_oof(CURRENT_RUN)
    candidate_labels, candidate_oof = _load_oof(candidate_run)
    if len(labels) != len(candidate_labels):
        raise ValueError(f"OOF length mismatch: current={len(labels)} candidate={len(candidate_labels)}")
    current_test = _load_test(CURRENT_RUN)
    candidate_test = _load_test(candidate_run)
    if current_test.shape != candidate_test.shape:
        raise ValueError(f"Test prediction shape mismatch: current={current_test.shape} candidate={candidate_test.shape}")

    current_scores = _score_targets(labels, current_oof)
    candidate_scores = _score_targets(labels, candidate_oof)

    target_delta = {
        target: float(candidate_scores[target] - current_scores[target])
        for target in TARGET_COLUMNS
    }
    improved_non_s4 = [
        target for target in TARGET_COLUMNS
        if target != "S4" and target_delta[target] < 0.0
    ]
    max_loss = max(max(delta, 0.0) for delta in target_delta.values())
    test_drift = {
        target: float(np.mean(np.abs(candidate_test[target].to_numpy(float) - current_test[target].to_numpy(float))))
        for target in TARGET_COLUMNS
    }
    max_drift = max(test_drift.values())
    has_effect = any(drift > 1e-12 for drift in test_drift.values())
    lower_name = candidate_name.lower()
    forbidden_hits = [token for token in FORBIDDEN_NAME_TOKENS if token in lower_name]
    weight_check = _load_weight_check(weights_json)

    failures: list[str] = []
    if forbidden_hits and not allow_forbidden_name:
        failures.append(f"forbidden axis token(s): {', '.join(forbidden_hits)}")
    if len(improved_non_s4) < MIN_NON_S4_IMPROVED:
        failures.append(f"non-S4 improved targets {len(improved_non_s4)} < {MIN_NON_S4_IMPROVED}")
    if max_loss > MAX_TARGET_LOSS:
        failures.append(f"max target OOF loss {max_loss:.8f} > {MAX_TARGET_LOSS:.8f}")
    if max_drift > MAX_TARGET_TEST_DRIFT:
        failures.append(f"max target test drift {max_drift:.8f} > {MAX_TARGET_TEST_DRIFT:.8f}")
    if not has_effect:
        failures.append("candidate has no test-prediction effect versus current")
    if weight_check["provided"] and weight_check["all_zero"]:
        failures.append("provided weight metadata contains no non-zero weights")

    return {
        "candidate_run": candidate_run,
        "candidate_name": candidate_name,
        "current_run": CURRENT_RUN,
        "current_public": CURRENT_PUBLIC,
        "approved": not failures,
        "failures": failures,
        "weight_check": weight_check,
        "improved_non_s4_targets": improved_non_s4,
        "target_delta_candidate_minus_current": target_delta,
        "test_mean_abs_drift": test_drift,
        "current_scores": current_scores,
        "candidate_scores": candidate_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-name", default="")
    parser.add_argument("--weights-json", default=None)
    parser.add_argument("--allow-forbidden-name", action="store_true")
    args = parser.parse_args()

    candidate_name = args.candidate_name or args.candidate_run
    result = _guard_candidate(args.candidate_run, candidate_name, args.allow_forbidden_name, args.weights_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

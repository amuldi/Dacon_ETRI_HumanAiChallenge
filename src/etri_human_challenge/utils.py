"""Utility helpers shared across modules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import EPSILON, TIME_BUCKETS
from .paths import ensure_parent


def hour_to_bucket(hour: int) -> str:
    for name, (start, end) in TIME_BUCKETS.items():
        if start <= hour < end:
            return name
    return "unknown"


def flatten_columns(columns: list[tuple[str, str]]) -> list[str]:
    output: list[str] = []
    for left, right in columns:
        if right:
            output.append(f"{left}__{right}")
        else:
            output.append(left)
    return output


def clip_probabilities(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1.0 - EPSILON)


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = clip_probabilities(y_prob)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def multi_target_log_loss(y_true: pd.DataFrame, y_prob: pd.DataFrame, targets: list[str]) -> dict[str, Any]:
    scores = {target: binary_log_loss(y_true[target].values, y_prob[target].values) for target in targets}
    scores["mean"] = float(np.mean([scores[target] for target in targets]))
    scores["std"] = float(np.std([scores[target] for target in targets]))
    return scores


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_markdown(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content.rstrip() + "\n")


def to_date_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        if math.isnan(float(value)):
            return None
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


"""Target-wise submission mixing helpers."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

from pathlib import Path

import pandas as pd

from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .io import load_submission_template
from .paths import REPORT_SUBMISSIONS_DIR, SUBMISSIONS_DIR, ensure_runtime_dirs
from .utils import write_markdown


def mix_submission_files(
    left_path: Path,
    right_path: Path,
    *,
    tag: str,
    target_weights: dict[str, float],
) -> Path:
    ensure_runtime_dirs()
    template = load_submission_template()[KEY_COLUMNS + TARGET_COLUMNS].copy()
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    expected_columns = list(template.columns)
    if list(left.columns) != expected_columns or list(right.columns) != expected_columns:
        raise ValueError("Submission columns do not match the sample submission.")
    if len(left) != len(template) or len(right) != len(template):
        raise ValueError("Submission row counts do not match the sample submission.")

    mixed = left.copy()
    for target, weight in target_weights.items():
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Unknown target: {target}")
        mixed[target] = (1.0 - weight) * left[target] + weight * right[target]

    output_path = SUBMISSIONS_DIR / f"submission_{tag}_mixed.csv"
    mixed.to_csv(output_path, index=False)

    report_lines = [
        "# Submission Mix Report",
        "",
        f"- Left: `{left_path}`",
        f"- Right: `{right_path}`",
        f"- Output: `{output_path}`",
        "",
        "## Target Weights",
        "",
    ]
    for target in TARGET_COLUMNS:
        weight = float(target_weights.get(target, 0.0))
        report_lines.append(f"- `{target}`: right_weight={weight:g}")
    write_markdown(REPORT_SUBMISSIONS_DIR / f"submission_{tag}_mixed.md", "\n".join(report_lines))
    return output_path

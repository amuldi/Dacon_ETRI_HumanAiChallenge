"""Schema auditing and data contract generation."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .io import build_key_frame, load_modality_frame, load_train_labels, modality_paths
from .paths import CONTRACTS_DIR, REPORT_CONTRACTS_DIR, ensure_runtime_dirs
from .utils import safe_float, to_date_string, write_json, write_markdown


def _compact_sample(value: Any) -> str:
    text = repr(value)
    if len(text) > 220:
        return text[:217] + "..."
    return text


def summarize_train_labels() -> dict[str, Any]:
    train = load_train_labels()
    day_delta = (train["sleep_date"] - train["lifelog_date"]).dt.days
    return {
        "rows": int(len(train)),
        "subjects": int(train["subject_id"].nunique()),
        "sleep_dates": int(train["sleep_date"].nunique()),
        "lifelog_dates": int(train["lifelog_date"].nunique()),
        "lifelog_to_sleep_delta_days": {str(k): int(v) for k, v in day_delta.value_counts().sort_index().items()},
        "target_positive_rate": {target: float(train[target].mean()) for target in TARGET_COLUMNS},
    }


def summarize_modality(path: Path) -> dict[str, Any]:
    schema = pq.read_schema(path)
    frame = load_modality_frame(path).head(2)
    first_timestamp = pd.to_datetime(frame["timestamp"]).min()
    last_timestamp = pd.to_datetime(frame["timestamp"]).max()
    sample_records = frame.to_dict(orient="records")

    timestamp_probe = load_modality_frame(path)[["subject_id", "timestamp"]].copy()
    timestamp_probe["lifelog_date"] = pd.to_datetime(timestamp_probe["timestamp"]).dt.normalize()

    return {
        "file": path.name,
        "rows": int(pq.ParquetFile(path).metadata.num_rows),
        "columns": [field.name for field in schema],
        "schema": {field.name: str(field.type) for field in schema},
        "subjects": int(timestamp_probe["subject_id"].nunique()),
        "lifelog_date_min": to_date_string(timestamp_probe["lifelog_date"].min()),
        "lifelog_date_max": to_date_string(timestamp_probe["lifelog_date"].max()),
        "daily_observations_mean": float(timestamp_probe.groupby(["subject_id", "lifelog_date"]).size().mean()),
        "sample_records": [{key: _compact_sample(value) for key, value in row.items()} for row in sample_records],
        "first_timestamp_sample": to_date_string(first_timestamp),
        "last_timestamp_sample": to_date_string(last_timestamp),
    }


def build_schema_contract() -> dict[str, Any]:
    key_frame = build_key_frame()
    modalities = {path.stem: summarize_modality(path) for path in modality_paths()}
    return {
        "keys": KEY_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "train_summary": summarize_train_labels(),
        "combined_key_rows": int(len(key_frame)),
        "combined_train_rows": int((key_frame["split"] == "train").sum()),
        "combined_test_rows": int((key_frame["split"] == "test").sum()),
        "aggregation_contract": {
            "primary_join": ["subject_id", "lifelog_date"],
            "final_row_grain": ["subject_id", "sleep_date"],
            "date_rule": "lifelog_date must equal sleep_date minus one day",
            "allowed_feature_types": ["numeric", "missingness_flag"],
        },
        "leakage_risks": [
            "Do not aggregate logs from sleep_date or later when predicting that sleep_date.",
            "Personal baseline features must use only prior lifelog dates for the same subject.",
            "OOF calibration must be fit inside each training fold, not on the full OOF table.",
        ],
        "modalities": modalities,
    }


def render_schema_contract_markdown(contract: dict[str, Any]) -> str:
    train_summary = contract["train_summary"]
    lines = [
        "# Data Contract",
        "",
        "## Core Grain",
        "",
        f"- Training rows: {train_summary['rows']}",
        f"- Subjects: {train_summary['subjects']}",
        f"- Final feature row grain: `{', '.join(contract['aggregation_contract']['final_row_grain'])}`",
        f"- Primary modality join: `{', '.join(contract['aggregation_contract']['primary_join'])}`",
        f"- Date rule: {contract['aggregation_contract']['date_rule']}",
        "",
        "## Target Balance",
        "",
    ]
    for target, rate in train_summary["target_positive_rate"].items():
        lines.append(f"- `{target}` positive rate: {rate:.4f}")
    lines.extend(["", "## Leakage Risks", ""])
    for risk in contract["leakage_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", "## Modalities", ""])
    for modality, summary in contract["modalities"].items():
        lines.append(f"### `{modality}`")
        lines.append("")
        lines.append(f"- Rows: {summary['rows']}")
        lines.append(f"- Columns: {', '.join(summary['columns'])}")
        lines.append(f"- Date range: {summary['lifelog_date_min']} to {summary['lifelog_date_max']}")
        lines.append(f"- Mean observations per subject-day: {summary['daily_observations_mean']:.2f}")
        lines.append(f"- Sample record: `{summary['sample_records'][0]}`")
        lines.append("")
    return "\n".join(lines)


def run_schema_audit() -> dict[str, Any]:
    ensure_runtime_dirs()
    contract = build_schema_contract()
    write_json(CONTRACTS_DIR / "schema_contract.json", contract)
    write_markdown(REPORT_CONTRACTS_DIR / "schema_contract.md", render_schema_contract_markdown(contract))
    return contract


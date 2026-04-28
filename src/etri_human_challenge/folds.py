"""Fold manifest generation."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .constants import DEFAULT_N_SPLITS, KEY_COLUMNS, RANDOM_STATE
from .io import load_train_labels
from .paths import FEATURES_DIR, FOLDS_DIR, REPORT_FOLDS_DIR, ensure_runtime_dirs
from .utils import write_markdown


def build_group_manifest(train: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS) -> pd.DataFrame:
    n_splits = min(n_splits, train["subject_id"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    records: list[dict[str, object]] = []
    for fold_id, (_, valid_idx) in enumerate(splitter.split(train, groups=train["subject_id"])):
        valid_keys = train.iloc[valid_idx][KEY_COLUMNS]
        for _, row in valid_keys.iterrows():
            records.append(
                {
                    "subject_id": row["subject_id"],
                    "sleep_date": row["sleep_date"],
                    "lifelog_date": row["lifelog_date"],
                    "split_scheme": "group",
                    "fold_id": fold_id,
                    "role": "valid",
                }
            )
    return pd.DataFrame(records)


def build_group_time_manifest(train: pd.DataFrame, n_splits: int = DEFAULT_N_SPLITS) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = train.sort_values(["subject_id", "lifelog_date"]).copy()
    ordered["position"] = ordered.groupby("subject_id").cumcount()
    ordered["group_size"] = ordered.groupby("subject_id")["subject_id"].transform("size")

    for subject_id, group in ordered.groupby("subject_id", sort=False):
        size = len(group)
        cut_40 = int(np.floor(size * 0.4))
        cut_60 = int(np.floor(size * 0.6))
        cut_80 = int(np.floor(size * 0.8))
        slices = {
            0: group.iloc[cut_40:cut_60],
            1: group.iloc[cut_60:cut_80],
            2: group.iloc[cut_80:],
        }
        for fold_id, valid_slice in slices.items():
            for _, row in valid_slice.iterrows():
                records.append(
                    {
                        "subject_id": row["subject_id"],
                        "sleep_date": row["sleep_date"],
                        "lifelog_date": row["lifelog_date"],
                        "split_scheme": "group_time",
                        "fold_id": fold_id,
                        "role": "valid",
                    }
                )
    return pd.DataFrame(records)


def build_fold_manifest(n_splits: int = DEFAULT_N_SPLITS) -> pd.DataFrame:
    train = load_train_labels().copy()
    manifest = pd.concat(
        [
            build_group_manifest(train, n_splits=n_splits),
            build_group_time_manifest(train, n_splits=n_splits),
        ],
        ignore_index=True,
    )
    return manifest.sort_values(["split_scheme", "fold_id", "subject_id", "sleep_date"]).reset_index(drop=True)


def render_fold_report(manifest: pd.DataFrame) -> str:
    lines = ["# Fold Report", ""]
    for scheme, scheme_df in manifest.groupby("split_scheme"):
        lines.append(f"## `{scheme}`")
        lines.append("")
        counts = scheme_df.groupby("fold_id").size().to_dict()
        for fold_id, count in counts.items():
            lines.append(f"- Fold {fold_id}: {count} validation rows")
        lines.append("")
    return "\n".join(lines)


def run_fold_build(n_splits: int = DEFAULT_N_SPLITS) -> pd.DataFrame:
    ensure_runtime_dirs()
    manifest = build_fold_manifest(n_splits=n_splits)
    manifest.to_parquet(FOLDS_DIR / "fold_manifest.parquet", index=False)
    write_markdown(REPORT_FOLDS_DIR / "fold_manifest.md", render_fold_report(manifest))
    return manifest

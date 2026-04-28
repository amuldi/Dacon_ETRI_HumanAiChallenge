"""I/O helpers for raw data and generated artifacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import bootstrap as _bootstrap  # noqa: F401

import pandas as pd

from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .paths import DATA_DIR, RAW_MODALITY_DIR


TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
SUBMISSION_PATH = DATA_DIR / "ch2026_submission_sample.csv"


@lru_cache(maxsize=1)
def load_train_labels() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    df["split"] = "train"
    return df


@lru_cache(maxsize=1)
def load_submission_template() -> pd.DataFrame:
    df = pd.read_csv(SUBMISSION_PATH, parse_dates=["sleep_date", "lifelog_date"])
    for target in TARGET_COLUMNS:
        df[target] = pd.NA
    df["split"] = "test"
    return df


def build_key_frame() -> pd.DataFrame:
    train = load_train_labels()
    test = load_submission_template()
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined = combined.sort_values(KEY_COLUMNS + ["split"]).drop_duplicates(KEY_COLUMNS, keep="first")
    return combined.reset_index(drop=True)


def load_modality_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def modality_paths() -> list[Path]:
    return sorted(RAW_MODALITY_DIR.glob("*.parquet"))


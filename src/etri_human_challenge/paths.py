"""Filesystem paths used by the pipeline."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_MODALITY_DIR = DATA_DIR / "ch2025_data_items"

ARTIFACTS_DIR = ROOT / "artifacts"
CONTRACTS_DIR = ARTIFACTS_DIR / "contracts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
FOLDS_DIR = ARTIFACTS_DIR / "folds"
OOF_DIR = ARTIFACTS_DIR / "oof"
MODELS_DIR = ARTIFACTS_DIR / "models"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"

REPORTS_DIR = ROOT / "reports"
REPORT_CONTRACTS_DIR = REPORTS_DIR / "contracts"
REPORT_FEATURES_DIR = REPORTS_DIR / "features"
REPORT_FOLDS_DIR = REPORTS_DIR / "folds"
REPORT_OOF_DIR = REPORTS_DIR / "oof"
REPORT_EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
REPORT_MODELS_DIR = REPORTS_DIR / "models"
REPORT_SUBMISSIONS_DIR = REPORTS_DIR / "submissions"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_runtime_dirs() -> None:
    for directory in [
        CONTRACTS_DIR,
        FEATURES_DIR,
        FOLDS_DIR,
        OOF_DIR,
        MODELS_DIR,
        EXPERIMENTS_DIR,
        SUBMISSIONS_DIR,
        REPORT_CONTRACTS_DIR,
        REPORT_FEATURES_DIR,
        REPORT_FOLDS_DIR,
        REPORT_OOF_DIR,
        REPORT_EXPERIMENTS_DIR,
        REPORT_MODELS_DIR,
        REPORT_SUBMISSIONS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

"""Group-time cross-validation split for temporal subject data.

Replaces StratifiedKFold used in the current pipeline, which allows the
same subject's records to appear in both train and validation folds.  That
leaks subject-level behavioral patterns and inflates OOF scores.

Design
------
For each subject the timeline is partitioned into sections:
  • Fold 0 (validation): rows at positions [40%, 60%) of that subject's timeline
  • Fold 1 (validation): rows at positions [60%, 80%)
  • Fold 2 (validation): rows at positions [80%, 100%)

Training rows for fold F are all rows that precede the validation rows for
that subject (pure forward-chaining, no future look-ahead).  This mirrors
real deployment: predict tomorrow using only what happened today and earlier.

For 5-fold compatibility the class also supports a subject-stratified split
(GroupKFold) where each group = one subject.  Both schemes are provided to
allow ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from .constants import KEY_COLUMNS


# ─────────────────────────────────────────────
# Pure forward-chaining group-time splits
# ─────────────────────────────────────────────

@dataclass
class FoldSlice:
    fold_id: int
    train_idx: np.ndarray   # indices into the *original* train frame
    valid_idx: np.ndarray


def build_group_time_splits(
    frame: pd.DataFrame,
    *,
    n_folds: int = 3,
    valid_fraction: float = 0.20,
) -> list[FoldSlice]:
    """
    Return list of FoldSlice for a subject-grouped, time-ordered frame.

    Parameters
    ----------
    frame:
        Must have columns ``subject_id`` and ``lifelog_date``.
    n_folds:
        Number of folds (default 3).  Each fold uses the *last* ``valid_fraction``
        of each subject's remaining timeline as validation.
    valid_fraction:
        Fraction of each subject's timeline used as validation per fold.

    Forward-chaining guarantee
    --------------------------
    For fold F, training rows for subject S are all rows with position
    < cut_F(S).  Validation rows are positions [cut_F(S), cut_{F+1}(S)).
    No future data is visible during training or validation.
    """
    ordered = (
        frame
        .sort_values(["subject_id", "lifelog_date"])
        .reset_index(drop=False)     # keep original integer index
    )
    # We will collect per-fold (train_idx, valid_idx) lists
    fold_train: list[list[int]] = [[] for _ in range(n_folds)]
    fold_valid: list[list[int]] = [[] for _ in range(n_folds)]

    for _, subject_df in ordered.groupby("subject_id", sort=False):
        orig_indices = subject_df["index"].tolist()   # original frame indices
        n = len(orig_indices)
        if n < 4:
            # Too few rows to make meaningful splits — add to every training set
            for f in range(n_folds):
                fold_train[f].extend(orig_indices)
            continue

        # Cut points (fraction-based)
        cuts = [int(np.floor(n * (1.0 - valid_fraction * (n_folds - f)))) for f in range(n_folds)]
        cuts.append(n)

        for f in range(n_folds):
            train_end = cuts[f]
            valid_start = cuts[f]
            valid_end = cuts[f + 1]
            train_rows = orig_indices[:train_end]
            valid_rows = orig_indices[valid_start:valid_end]
            if not valid_rows:
                # Edge case: tiny subject — fallback
                valid_rows = orig_indices[max(0, n - 1):]
                train_rows = orig_indices[: max(0, n - 1)]
            fold_train[f].extend(train_rows)
            fold_valid[f].extend(valid_rows)

    slices = [
        FoldSlice(
            fold_id=f,
            train_idx=np.array(sorted(fold_train[f]), dtype=int),
            valid_idx=np.array(sorted(fold_valid[f]), dtype=int),
        )
        for f in range(n_folds)
    ]
    return slices


def group_time_split_iter(
    frame: pd.DataFrame,
    *,
    n_folds: int = 3,
    valid_fraction: float = 0.20,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """sklearn-style iterator: yields (train_idx, valid_idx) pairs."""
    for fs in build_group_time_splits(frame, n_folds=n_folds, valid_fraction=valid_fraction):
        yield fs.train_idx, fs.valid_idx


# ─────────────────────────────────────────────
# Leave-One-Subject-Out (for diagnostics)
# ─────────────────────────────────────────────

def loso_split_iter(frame: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Subject-Out CV.

    Yields (train_idx, valid_idx) for each unique subject.
    Best for bias estimation but expensive (~30 folds for this dataset).
    """
    frame = frame.reset_index(drop=True)
    subjects = frame["subject_id"].unique()
    for subj in subjects:
        valid_idx = frame.index[frame["subject_id"] == subj].to_numpy()
        train_idx = frame.index[frame["subject_id"] != subj].to_numpy()
        yield train_idx, valid_idx


# ─────────────────────────────────────────────
# Adversarial validation helper
# ─────────────────────────────────────────────

def compute_adversarial_auc(
    feature_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_folds: int = 5,
    random_state: int = 42,
) -> float:
    """
    Train a classifier to distinguish train rows from test rows.

    AUC ≈ 0.5 means train/test distributions are similar (good).
    AUC > 0.7 means significant distribution shift (feature engineering
    or selection may be needed to improve generalization).

    Returns
    -------
    mean OOF AUC across folds.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    train_mask = feature_frame["split"] == "train"
    test_mask = feature_frame["split"] == "test"
    X = feature_frame[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = np.zeros(len(feature_frame), dtype=int)
    y[test_mask.to_numpy()] = 1

    # Only rows that are in train or test
    mask = (train_mask | test_mask).to_numpy()
    X, y = X[mask], y[mask]

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    for tr_idx, va_idx in skf.split(X, y):
        clf.fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[va_idx])[:, 1]
        if len(np.unique(y[va_idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[va_idx], proba)))

    return float(np.mean(aucs)) if aucs else float("nan")

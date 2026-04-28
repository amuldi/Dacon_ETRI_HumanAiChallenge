"""
실제 대회 구조를 재현하는 CV 스킴.

대회 test = 각 subject 날짜를 비시간순으로 ~36% 랜덤 선택.
→ 올바른 CV: 각 subject의 날짜를 n_folds 등분하여 순환 holdout.
   시간 순서 없음. subject 간 완전 독립.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator


def subject_stratified_holdout_iter(
    train_frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    subject별로 행을 n_folds에 균등 배정 후 순환 holdout.

    Parameters
    ----------
    train_frame : train split DataFrame. 반드시 reset_index(drop=True) 상태.
    n_folds     : fold 수 (기본 5).
    random_state: 재현성용 seed.

    Yields
    ------
    (train_idx, val_idx) : 정수 numpy 배열 쌍.

    보장사항
    --------
    - 각 행은 정확히 1번 val에 등장.
    - 같은 subject의 행이 한 fold에서 train/val에 동시 등장하지 않음.
    - 시간 순서 제약 없음 (대회 split 구조 반영).
    """
    rng = np.random.default_rng(random_state)
    frame = train_frame.reset_index(drop=True)
    fold_assignment = np.full(len(frame), -1, dtype=int)

    for subj in frame["subject_id"].unique():
        idx = frame.index[frame["subject_id"] == subj].to_numpy()
        shuffled = rng.permutation(idx)
        for fold_i, chunk in enumerate(np.array_split(shuffled, n_folds)):
            fold_assignment[chunk] = fold_i

    for fold_id in range(n_folds):
        val_idx   = np.where(fold_assignment == fold_id)[0]
        train_idx = np.where(fold_assignment != fold_id)[0]
        yield train_idx, val_idx

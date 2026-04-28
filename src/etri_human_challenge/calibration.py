"""Probability calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .constants import EPSILON, RANDOM_STATE
from .utils import clip_probabilities


@dataclass
class IdentityCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return clip_probabilities(values)


class PlattCalibrator:
    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> "PlattCalibrator":
        unique_targets = np.unique(targets)
        if len(unique_targets) < 2:
            self._model = None
            return self
        logits = np.log(clip_probabilities(probabilities) / (1.0 - clip_probabilities(probabilities)))
        self._model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        self._model.fit(logits.reshape(-1, 1), targets)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self._model is None:
            return clip_probabilities(probabilities)
        logits = np.log(clip_probabilities(probabilities) / (1.0 - clip_probabilities(probabilities)))
        return clip_probabilities(self._model.predict_proba(logits.reshape(-1, 1))[:, 1])


"""Structured artifacts for experiments and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import write_json


@dataclass
class ExperimentCard:
    name: str
    model_family: str
    feature_view: str
    split_scheme: str
    mean_log_loss: float
    std_log_loss: float
    target_scores: dict[str, float]
    calibration: str
    improvement_over_dummy: float
    accepted: bool
    paper_relevance: str
    score_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_experiment_card(path: Path, card: ExperimentCard) -> None:
    write_json(path, card.to_dict())

"""Lightweight sequence models for short-window experiments."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .features import build_daily_feature_table
from .paths import EXPERIMENTS_DIR, FEATURES_DIR, REPORT_EXPERIMENTS_DIR, REPORT_MODELS_DIR, ensure_runtime_dirs
from .utils import multi_target_log_loss, write_markdown


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.targets[index]


class SequenceMLP(nn.Module):
    def __init__(self, feature_dim: int, window_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim * window_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(TARGET_COLUMNS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SequenceTCN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_dim, len(TARGET_COLUMNS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features.transpose(1, 2)).squeeze(-1)
        return self.head(encoded)


def _select_sequence_features(frame: pd.DataFrame) -> list[str]:
    selected = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and column not in TARGET_COLUMNS
        and not column.endswith("__delta_vs_expanding")
    ]
    return selected[:256]


def build_sequence_arrays(frame: pd.DataFrame, window_size: int = 7) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    selected = _select_sequence_features(frame)
    ordered = frame[frame["split"] == "train"].sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    keys: list[dict[str, Any]] = []
    for _, group in ordered.groupby("subject_id"):
        values = group[selected].fillna(0.0).to_numpy(dtype=np.float32)
        target_values = group[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        for index in range(len(group)):
            start = max(0, index - window_size + 1)
            window = values[start : index + 1]
            if len(window) < window_size:
                pad = np.zeros((window_size - len(window), values.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            sequences.append(window)
            targets.append(target_values[index])
            keys.append(group.iloc[index][KEY_COLUMNS].to_dict())
    return np.asarray(sequences), np.asarray(targets), pd.DataFrame(keys)


def train_sequence_lite(model_type: str = "mlp", window_size: int = 7, epochs: int = 8) -> dict[str, float]:
    ensure_runtime_dirs()
    feature_path = FEATURES_DIR / "daily_feature_table.parquet"
    frame = pd.read_parquet(feature_path) if feature_path.exists() else build_daily_feature_table()
    sequences, targets, keys = build_sequence_arrays(frame, window_size=window_size)

    dataset = SequenceDataset(sequences, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    feature_dim = sequences.shape[-1]
    model: nn.Module
    if model_type == "tcn":
        model = SequenceTCN(feature_dim=feature_dim)
    else:
        model = SequenceMLP(feature_dim=feature_dim, window_size=window_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(sequences, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    prediction_frame = pd.DataFrame(probabilities, columns=TARGET_COLUMNS)
    scores = multi_target_log_loss(pd.DataFrame(targets, columns=TARGET_COLUMNS), prediction_frame, TARGET_COLUMNS)

    report = "\n".join(
        [
            f"# Sequence-lite Report ({model_type})",
            "",
            f"- Window size: {window_size}",
            f"- Epochs: {epochs}",
            f"- Mean log-loss: {scores['mean']:.6f}",
            f"- Std across targets: {scores['std']:.6f}",
            "",
            "This report is diagnostic. Compare it against the calibrated tabular baseline before keeping the model family.",
        ]
    )
    write_markdown(REPORT_MODELS_DIR / f"sequence_lite_{model_type}.md", report)
    return {target: float(scores[target]) for target in TARGET_COLUMNS} | {"mean": float(scores["mean"])}

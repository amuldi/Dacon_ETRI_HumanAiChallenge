"""Target-wise subject prior tuning and submission generation."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline import (
    _build_valid_frame,
    _outer_train_indices,
    build_dummy_predictions,
    load_feature_table,
    load_fold_manifest,
)
from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .io import load_submission_template
from .paths import (
    EXPERIMENTS_DIR,
    OOF_DIR,
    REPORT_EXPERIMENTS_DIR,
    REPORT_OOF_DIR,
    REPORT_SUBMISSIONS_DIR,
    SUBMISSIONS_DIR,
    ensure_runtime_dirs,
)
from .utils import clip_probabilities, multi_target_log_loss, write_json, write_markdown


SMOOTHING_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0]
HALF_LIFE_GRID: list[float | None] = [None, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0]


@dataclass(frozen=True)
class PriorConfig:
    smoothing_m: float
    half_life: float | None

    @property
    def name(self) -> str:
        half_life = "static" if self.half_life is None else f"hl{self.half_life:g}"
        return f"m{self.smoothing_m:g}_{half_life}"


def _candidate_configs() -> list[PriorConfig]:
    return [PriorConfig(smoothing_m=m, half_life=half_life) for m in SMOOTHING_GRID for half_life in HALF_LIFE_GRID]


def _weighted_subject_priors(train_frame: pd.DataFrame, target: str, config: PriorConfig) -> tuple[float, dict[Any, float]]:
    global_prior = float(train_frame[target].mean())
    priors: dict[Any, float] = {}
    for subject_id, group in train_frame.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id", sort=False):
        values = group[target].to_numpy(dtype=float)
        if config.half_life is None:
            weights = np.ones(len(values), dtype=float)
        else:
            ages = np.arange(len(values) - 1, -1, -1, dtype=float)
            weights = np.power(0.5, ages / config.half_life)
        numerator = float(np.sum(values * weights) + config.smoothing_m * global_prior)
        denominator = float(np.sum(weights) + config.smoothing_m)
        priors[subject_id] = numerator / denominator
    return global_prior, priors


def _predict_prior(frame: pd.DataFrame, global_prior: float, subject_priors: dict[Any, float]) -> np.ndarray:
    return clip_probabilities(frame["subject_id"].map(subject_priors).fillna(global_prior).to_numpy(dtype=float))


def _oof_for_config(train: pd.DataFrame, scheme_manifest: pd.DataFrame, split_scheme: str, target: str, config: PriorConfig) -> pd.Series:
    oof = pd.Series(np.nan, index=train.index, dtype=float)
    for fold_id in sorted(scheme_manifest["fold_id"].unique()):
        valid_frame = _build_valid_frame(train, scheme_manifest, int(fold_id))
        outer_train = train.loc[_outer_train_indices(train, valid_frame, split_scheme)].copy()
        if len(outer_train) == 0 or len(valid_frame) == 0:
            continue
        global_prior, subject_priors = _weighted_subject_priors(outer_train, target, config)
        oof.loc[valid_frame.index] = _predict_prior(valid_frame, global_prior, subject_priors)
    return oof


def _target_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float:
    values = clip_probabilities(y_prob.to_numpy(dtype=float))
    targets = y_true.to_numpy(dtype=float)
    return float(-np.mean(targets * np.log(values) + (1.0 - targets) * np.log(1.0 - values)))


def tune_prior_v2(split_scheme: str = "group_time", persist: bool = True) -> dict[str, Any]:
    ensure_runtime_dirs()
    feature_table = load_feature_table()
    manifest = load_fold_manifest()
    train = feature_table[feature_table["split"] == "train"].copy()
    scheme_manifest = manifest[manifest["split_scheme"] == split_scheme].copy()

    best_configs: dict[str, PriorConfig] = {}
    oof = pd.DataFrame(index=train.index, columns=TARGET_COLUMNS, dtype=float)
    config_scores: dict[str, list[dict[str, Any]]] = {}

    for target in TARGET_COLUMNS:
        target_results: list[dict[str, Any]] = []
        best_score = float("inf")
        best_config: PriorConfig | None = None
        best_oof: pd.Series | None = None
        for config in _candidate_configs():
            target_oof = _oof_for_config(train, scheme_manifest, split_scheme, target, config)
            scored = target_oof.notna()
            score = _target_log_loss(train.loc[scored, target], target_oof.loc[scored])
            target_results.append({"config": config.name, "score": score, **asdict(config)})
            if score < best_score:
                best_score = score
                best_config = config
                best_oof = target_oof
        if best_config is None or best_oof is None:
            raise RuntimeError(f"No valid prior config for {target}")
        best_configs[target] = best_config
        oof[target] = best_oof
        config_scores[target] = sorted(target_results, key=lambda item: item["score"])[:10]

    scored_mask = oof[TARGET_COLUMNS].notna().all(axis=1)
    scored_train = train.loc[scored_mask].copy()
    scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], oof.loc[scored_mask], TARGET_COLUMNS)
    dummy_scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], build_dummy_predictions(scored_train), TARGET_COLUMNS)

    result = {
        "name": f"prior_v2_{split_scheme}",
        "split_scheme": split_scheme,
        "scores": scores,
        "dummy_scores": dummy_scores,
        "improvement_over_dummy": float(dummy_scores["mean"] - scores["mean"]),
        "best_configs": {target: asdict(config) | {"name": config.name} for target, config in best_configs.items()},
        "top_config_scores": config_scores,
    }

    if not persist:
        return result

    oof_export = scored_train[KEY_COLUMNS + TARGET_COLUMNS].copy()
    oof_export["split_scheme"] = split_scheme
    oof_export["model_family"] = "prior_v2"
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_prior_v2"] = oof.loc[scored_mask, target].values
    oof_path = OOF_DIR / f"oof_predictions_prior_v2_{split_scheme}.parquet"
    oof_export.to_parquet(oof_path, index=False)

    report_lines = [
        f"# Prior V2 Report ({split_scheme})",
        "",
        f"- Mean log-loss: {scores['mean']:.6f}",
        f"- Dummy mean log-loss: {dummy_scores['mean']:.6f}",
        f"- Improvement over dummy: {result['improvement_over_dummy']:.6f}",
        "",
        "## Selected Target Configs",
        "",
    ]
    for target in TARGET_COLUMNS:
        config = best_configs[target]
        half_life = "static" if config.half_life is None else f"{config.half_life:g}"
        report_lines.append(
            f"- `{target}`: score={scores[target]:.6f}, smoothing_m={config.smoothing_m:g}, half_life={half_life}"
        )
    report_lines.extend(["", "## Top Configs", ""])
    for target in TARGET_COLUMNS:
        top_items = ", ".join(f"{item['config']}={item['score']:.4f}" for item in config_scores[target][:5])
        report_lines.append(f"- `{target}`: {top_items}")

    write_markdown(REPORT_OOF_DIR / f"prior_v2_{split_scheme}.md", "\n".join(report_lines))
    write_markdown(REPORT_EXPERIMENTS_DIR / f"prior_v2_{split_scheme}.md", "\n".join(report_lines))
    write_json(EXPERIMENTS_DIR / f"prior_v2_{split_scheme}.json", result)
    result["artifacts"] = {
        "oof_path": str(oof_path),
        "report_path": str(REPORT_OOF_DIR / f"prior_v2_{split_scheme}.md"),
        "experiment_path": str(EXPERIMENTS_DIR / f"prior_v2_{split_scheme}.json"),
    }
    return result


def make_prior_v2_submission(split_scheme: str = "group_time", tag: str = "prior_v2") -> Path:
    ensure_runtime_dirs()
    result = tune_prior_v2(split_scheme=split_scheme, persist=True)
    feature_table = load_feature_table()
    train = feature_table[feature_table["split"] == "train"].copy()
    test = feature_table[feature_table["split"] == "test"].copy()

    predictions = load_submission_template()[KEY_COLUMNS].copy()
    report_lines = [
        f"# Prior V2 Submission Report ({split_scheme})",
        "",
        f"- Source OOF mean log-loss: {result['scores']['mean']:.6f}",
        f"- Train rows: {len(train)}",
        f"- Test rows: {len(test)}",
        "",
        "## Target Configs",
        "",
    ]
    for target in TARGET_COLUMNS:
        config_payload = result["best_configs"][target]
        config = PriorConfig(
            smoothing_m=float(config_payload["smoothing_m"]),
            half_life=None if config_payload["half_life"] is None else float(config_payload["half_life"]),
        )
        global_prior, subject_priors = _weighted_subject_priors(train, target, config)
        predictions[target] = _predict_prior(test, global_prior, subject_priors)
        half_life = "static" if config.half_life is None else f"{config.half_life:g}"
        report_lines.append(
            f"- `{target}`: global_prior={global_prior:.4f}, smoothing_m={config.smoothing_m:g}, half_life={half_life}"
        )

    output_path = SUBMISSIONS_DIR / f"submission_{tag}_prior_v2_{split_scheme}.csv"
    predictions.to_csv(output_path, index=False)
    write_markdown(REPORT_SUBMISSIONS_DIR / f"submission_{tag}_prior_v2_{split_scheme}.md", "\n".join(report_lines))
    return output_path

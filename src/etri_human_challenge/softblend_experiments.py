"""Target-wise public LGB softblend experiment assembly."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .io import load_submission_template, load_train_labels
from .paths import MODELS_DIR, OOF_DIR, ROOT
from .utils import binary_log_loss


CLIP_MIN = 0.02
CLIP_MAX = 0.98
N_FOLDS = 5
SEEDS = [42, 1234, 9999, 7, 314, 2025, 777, 555]
EXPECTED_TRAIN_ROWS = 450
EXPECTED_TEST_ROWS = 250
CONFIG_PATH = ROOT / "configs" / "public_lgb_softblend_experiments.json"
LOGS_DIR = ROOT / "logs"
SUBMISSIONS_DIR = ROOT / "submissions"
EXPERIMENT_LOG_PATH = LOGS_DIR / "experiments.csv"

SOURCE_RUNS = {
    "public_core": "public_lgb_public_core",
    "public_hist411": "public_lgb_public_hist411",
    "public_hist365": "public_lgb_public_hist365",
}

EXPERIMENT_LOG_COLUMNS = [
    "timestamp",
    "experiment_name",
    "validation_scheme",
    "seeds",
    "total_oof_logloss",
    "target_logloss_Q1",
    "target_logloss_Q2",
    "target_logloss_Q3",
    "target_logloss_S1",
    "target_logloss_S2",
    "target_logloss_S3",
    "target_logloss_S4",
    "submission_file",
    "feature_view_by_target",
    "notes",
]


@dataclass(frozen=True)
class SoftblendConfig:
    name: str
    q2_hist_weight: float
    q3_hist_weight: float
    s4_hist_weight: float
    submission_file: str
    notes: str


@dataclass(frozen=True)
class SourcePredictions:
    oof: dict[str, pd.DataFrame]
    test: dict[str, pd.DataFrame]
    labels: pd.DataFrame
    train_keys: pd.DataFrame
    sample_submission: pd.DataFrame
    validation_scheme: str


@dataclass(frozen=True)
class ExperimentResult:
    config: SoftblendConfig
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    scores: dict[str, float]
    feature_view_by_target: dict[str, str]
    submission_path: Path
    validation_scheme: str


def load_experiment_configs(path: Path = CONFIG_PATH) -> dict[str, SoftblendConfig]:
    payload = json.loads(path.read_text())
    configs: dict[str, SoftblendConfig] = {}
    for name, item in payload.items():
        configs[name] = SoftblendConfig(
            name=name,
            q2_hist_weight=float(item["q2_hist_weight"]),
            q3_hist_weight=float(item["q3_hist_weight"]),
            s4_hist_weight=float(item["s4_hist_weight"]),
            submission_file=str(item["submission_file"]),
            notes=str(item["notes"]),
        )
    return configs


def _prediction_columns(frame: pd.DataFrame, view: str) -> pd.DataFrame:
    columns = {f"{target}_public_lgb": target for target in TARGET_COLUMNS}
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"{view} OOF prediction columns missing: {missing}")
    return frame[list(columns)].rename(columns=columns)


def _assert_targets_exist(frame: pd.DataFrame, frame_name: str) -> None:
    missing = [target for target in TARGET_COLUMNS if target not in frame.columns]
    if missing:
        raise KeyError(f"{frame_name} target columns missing: {missing}")


def _assert_shape(frame: pd.DataFrame, rows: int, frame_name: str) -> None:
    if frame.shape != (rows, len(TARGET_COLUMNS)):
        raise ValueError(f"{frame_name} shape invalid: {frame.shape}, expected {(rows, len(TARGET_COLUMNS))}")
    if list(frame.columns) != TARGET_COLUMNS:
        raise ValueError(f"{frame_name} columns invalid: {frame.columns.tolist()}")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{frame_name} contains NaN or infinite predictions")


def _assert_clipped(frame: pd.DataFrame, frame_name: str, *, clip_min: float, clip_max: float) -> None:
    values = frame.to_numpy(dtype=float)
    if values.min() < clip_min - 1e-12 or values.max() > clip_max + 1e-12:
        raise ValueError(
            f"{frame_name} predictions outside [{clip_min}, {clip_max}]: "
            f"min={values.min():.6f}, max={values.max():.6f}"
        )


def _assert_key_alignment(frames: dict[str, pd.DataFrame]) -> None:
    reference_name = next(iter(frames))
    reference = frames[reference_name][KEY_COLUMNS + TARGET_COLUMNS].reset_index(drop=True)
    for name, frame in frames.items():
        candidate = frame[KEY_COLUMNS + TARGET_COLUMNS].reset_index(drop=True)
        if not reference.equals(candidate):
            raise ValueError(f"OOF key/label alignment mismatch: {reference_name} vs {name}")


def _detect_validation_scheme(oof_frames: dict[str, pd.DataFrame]) -> str:
    values: set[str] = set()
    for frame in oof_frames.values():
        if "split_scheme" in frame.columns:
            values.update(str(item) for item in frame["split_scheme"].dropna().unique())
    if not values:
        return "unknown"
    if len(values) > 1:
        return "+".join(sorted(values))
    return next(iter(values))


def load_source_predictions() -> SourcePredictions:
    train = load_train_labels()
    sample = load_submission_template()
    _assert_targets_exist(train, "train")
    _assert_targets_exist(sample, "sample_submission")
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"train row count mismatch: {len(train)} != {EXPECTED_TRAIN_ROWS}")
    if len(sample) != EXPECTED_TEST_ROWS:
        raise ValueError(f"test row count mismatch: {len(sample)} != {EXPECTED_TEST_ROWS}")

    raw_oof: dict[str, pd.DataFrame] = {}
    oof: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    for view, run_name in SOURCE_RUNS.items():
        oof_path = OOF_DIR / f"oof_predictions_{run_name}.parquet"
        test_path = MODELS_DIR / f"test_predictions_{run_name}.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF prediction not found: {oof_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"test prediction not found: {test_path}")

        raw = pd.read_parquet(oof_path)
        _assert_targets_exist(raw, f"{view} OOF")
        raw_oof[view] = raw
        oof[view] = _prediction_columns(raw, view)

        test_frame = pd.read_csv(test_path)
        _assert_targets_exist(test_frame, f"{view} test")
        test[view] = test_frame[TARGET_COLUMNS].copy()
        _assert_shape(oof[view], EXPECTED_TRAIN_ROWS, f"{view} OOF predictions")
        _assert_shape(test[view], EXPECTED_TEST_ROWS, f"{view} test predictions")

    _assert_key_alignment(raw_oof)
    labels = raw_oof["public_core"][TARGET_COLUMNS].copy()
    train_keys = raw_oof["public_core"][KEY_COLUMNS].copy()
    return SourcePredictions(
        oof=oof,
        test=test,
        labels=labels,
        train_keys=train_keys,
        sample_submission=sample,
        validation_scheme=_detect_validation_scheme(raw_oof),
    )


def _clip_predictions(frame: pd.DataFrame, *, clip_min: float, clip_max: float) -> pd.DataFrame:
    return frame.clip(lower=clip_min, upper=clip_max).astype(float)


def _blend(left: pd.Series, right: pd.Series, right_weight: float) -> pd.Series:
    return (1.0 - right_weight) * left.astype(float) + right_weight * right.astype(float)


def feature_view_by_target(config: SoftblendConfig) -> dict[str, str]:
    return {
        "Q1": "public_core",
        "Q2": f"blend(public_core={1.0 - config.q2_hist_weight:.2f},public_hist411={config.q2_hist_weight:.2f})"
        if config.q2_hist_weight < 1.0
        else "public_hist411",
        "Q3": f"blend(public_core={1.0 - config.q3_hist_weight:.2f},public_hist365={config.q3_hist_weight:.2f})"
        if config.q3_hist_weight < 1.0
        else "public_hist365",
        "S1": "public_core",
        "S2": "public_core",
        "S3": "public_core",
        "S4": "public_hist411" if config.s4_hist_weight == 1.0 else f"blend(public_core={1.0 - config.s4_hist_weight:.2f},public_hist411={config.s4_hist_weight:.2f})",
    }


def assemble_predictions(
    config: SoftblendConfig,
    sources: SourcePredictions,
    *,
    split: str,
    clip_min: float = CLIP_MIN,
    clip_max: float = CLIP_MAX,
) -> pd.DataFrame:
    if split not in {"oof", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    source = sources.oof if split == "oof" else sources.test
    assembled = pd.DataFrame(index=source["public_core"].index, columns=TARGET_COLUMNS, dtype=float)
    assembled["Q1"] = source["public_core"]["Q1"]
    assembled["Q2"] = _blend(source["public_core"]["Q2"], source["public_hist411"]["Q2"], config.q2_hist_weight)
    assembled["Q3"] = _blend(source["public_core"]["Q3"], source["public_hist365"]["Q3"], config.q3_hist_weight)
    assembled["S1"] = source["public_core"]["S1"]
    assembled["S2"] = source["public_core"]["S2"]
    assembled["S3"] = source["public_core"]["S3"]
    assembled["S4"] = _blend(source["public_core"]["S4"], source["public_hist411"]["S4"], config.s4_hist_weight)
    clipped = _clip_predictions(assembled, clip_min=clip_min, clip_max=clip_max)
    rows = EXPECTED_TRAIN_ROWS if split == "oof" else EXPECTED_TEST_ROWS
    _assert_shape(clipped, rows, f"{config.name} {split} predictions")
    _assert_clipped(clipped, f"{config.name} {split}", clip_min=clip_min, clip_max=clip_max)
    return clipped


def score_oof(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    scores = {
        target: binary_log_loss(labels[target].to_numpy(dtype=float), predictions[target].to_numpy(dtype=float))
        for target in TARGET_COLUMNS
    }
    scores["mean"] = float(np.mean([scores[target] for target in TARGET_COLUMNS]))
    return scores


def resolve_submission_path(config: SoftblendConfig) -> Path:
    path = Path(config.submission_file)
    if not path.is_absolute():
        path = ROOT / path
    return path


def write_submission(
    config: SoftblendConfig,
    sources: SourcePredictions,
    test_predictions: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> Path:
    output_path = resolve_submission_path(config)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing submission: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = sources.sample_submission[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = test_predictions[target].to_numpy(dtype=float)
    expected_columns = sources.sample_submission[KEY_COLUMNS + TARGET_COLUMNS].columns.tolist()
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"submission columns mismatch: {submission.columns.tolist()} != {expected_columns}")
    if len(submission) != EXPECTED_TEST_ROWS:
        raise ValueError(f"submission row count mismatch: {len(submission)} != {EXPECTED_TEST_ROWS}")
    _assert_clipped(submission[TARGET_COLUMNS], config.name, clip_min=CLIP_MIN, clip_max=CLIP_MAX)
    submission.to_csv(output_path, index=False)
    return output_path


def validate_existing_submission(
    path: Path,
    sources: SourcePredictions,
    expected_predictions: pd.DataFrame,
    *,
    name: str,
) -> None:
    submission = pd.read_csv(path)
    expected_columns = sources.sample_submission[KEY_COLUMNS + TARGET_COLUMNS].columns.tolist()
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"{name} submission columns mismatch: {submission.columns.tolist()} != {expected_columns}")
    if len(submission) != EXPECTED_TEST_ROWS:
        raise ValueError(f"{name} submission row count mismatch: {len(submission)} != {EXPECTED_TEST_ROWS}")
    sample_keys = sources.sample_submission[KEY_COLUMNS].astype(str).reset_index(drop=True)
    submission_keys = submission[KEY_COLUMNS].astype(str).reset_index(drop=True)
    if not sample_keys.equals(submission_keys):
        raise ValueError(f"{name} submission key columns do not match sample_submission")
    _assert_clipped(submission[TARGET_COLUMNS], name, clip_min=CLIP_MIN, clip_max=CLIP_MAX)
    max_abs_diff = float(
        np.max(np.abs(submission[TARGET_COLUMNS].to_numpy(dtype=float) - expected_predictions.to_numpy(dtype=float)))
    )
    if max_abs_diff > 1e-10:
        raise ValueError(f"{name} existing submission differs from assembled predictions: max_abs_diff={max_abs_diff:.12f}")


def run_experiment(
    config: SoftblendConfig,
    sources: SourcePredictions,
    *,
    write_new_submission: bool,
    overwrite: bool = False,
) -> ExperimentResult:
    oof_predictions = assemble_predictions(config, sources, split="oof")
    test_predictions = assemble_predictions(config, sources, split="test")
    scores = score_oof(sources.labels, oof_predictions)
    output_path = resolve_submission_path(config)
    if write_new_submission:
        output_path = write_submission(config, sources, test_predictions, overwrite=overwrite)
    elif not output_path.exists():
        raise FileNotFoundError(f"Configured submission file does not exist: {output_path}")
    else:
        validate_existing_submission(output_path, sources, test_predictions, name=config.name)
    return ExperimentResult(
        config=config,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        scores=scores,
        feature_view_by_target=feature_view_by_target(config),
        submission_path=output_path,
        validation_scheme=sources.validation_scheme,
    )


def append_experiment_log(result: ExperimentResult, *, timestamp: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "timestamp": timestamp,
        "experiment_name": result.config.name,
        "validation_scheme": result.validation_scheme,
        "seeds": json.dumps(SEEDS),
        "total_oof_logloss": result.scores["mean"],
        "submission_file": str(result.submission_path.relative_to(ROOT)),
        "feature_view_by_target": json.dumps(result.feature_view_by_target, ensure_ascii=False, sort_keys=True),
        "notes": result.config.notes,
    }
    for target in TARGET_COLUMNS:
        row[f"target_logloss_{target}"] = result.scores[target]
    frame = pd.DataFrame([row], columns=EXPERIMENT_LOG_COLUMNS)
    frame.to_csv(EXPERIMENT_LOG_PATH, mode="a", header=not EXPERIMENT_LOG_PATH.exists(), index=False)


def _fold_iter_for_target(train_keys: pd.DataFrame, labels: pd.DataFrame, target: str, seed: int, validation_scheme: str):
    if validation_scheme == "public_stratified":
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed))
        yield from splitter.split(np.zeros((len(labels), 1)), labels[target].astype(int).to_numpy())
        return
    if validation_scheme == "subject_holdout":
        from .proper_cv import subject_stratified_holdout_iter

        frame = pd.concat([train_keys.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
        yield from subject_stratified_holdout_iter(frame, n_folds=N_FOLDS, random_state=int(seed))
        return
    if validation_scheme == "group_time":
        from .group_time_cv import group_time_split_iter

        frame = train_keys.reset_index(drop=True).copy()
        yield from group_time_split_iter(frame, n_folds=N_FOLDS, valid_fraction=1.0 / N_FOLDS)
        return
    raise ValueError(f"Unsupported stability validation_scheme: {validation_scheme}")


def write_stability_report(result: ExperimentResult, sources: SourcePredictions) -> Path:
    stability_dir = LOGS_DIR / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        for seed in SEEDS:
            for fold, (_, valid_idx) in enumerate(
                _fold_iter_for_target(sources.train_keys, sources.labels, target, seed, result.validation_scheme)
            ):
                y_true = sources.labels[target].iloc[valid_idx].to_numpy(dtype=float)
                y_pred = result.oof_predictions[target].iloc[valid_idx].to_numpy(dtype=float)
                rows.append(
                    {
                        "target": target,
                        "fold": int(fold),
                        "seed": int(seed),
                        "logloss": binary_log_loss(y_true, y_pred),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["mean"] = frame.groupby("target")["logloss"].transform("mean")
    frame["std"] = frame.groupby("target")["logloss"].transform(lambda values: float(values.std(ddof=0)))
    output_path = stability_dir / f"stability_{result.config.name}.csv"
    frame[["target", "fold", "seed", "logloss", "mean", "std"]].to_csv(output_path, index=False)
    return output_path


def run_configured_experiments(
    *,
    experiment_names: list[str],
    write_submission_names: set[str],
    overwrite: bool = False,
    timestamp: str | None = None,
) -> list[ExperimentResult]:
    timestamp = timestamp or pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")
    configs = load_experiment_configs()
    unknown = sorted(set(experiment_names) - set(configs))
    if unknown:
        raise KeyError(f"Unknown experiment configs: {unknown}")
    if not overwrite:
        existing = [
            str(resolve_submission_path(configs[name]).relative_to(ROOT))
            for name in experiment_names
            if name in write_submission_names and resolve_submission_path(configs[name]).exists()
        ]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing submissions: {existing}")

    sources = load_source_predictions()
    results: list[ExperimentResult] = []
    for name in experiment_names:
        result = run_experiment(
            configs[name],
            sources,
            write_new_submission=name in write_submission_names,
            overwrite=overwrite,
        )
        append_experiment_log(result, timestamp=timestamp)
        write_stability_report(result, sources)
        results.append(result)
    return results

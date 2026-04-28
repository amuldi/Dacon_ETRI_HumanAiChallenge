"""OOF-first baseline training."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    HAS_CATBOOST = False

from .calibration import IdentityCalibrator, PlattCalibrator
from .constants import KEY_COLUMNS, RANDOM_STATE, TARGET_COLUMNS
from .contracts import ExperimentCard, save_experiment_card
from .io import load_submission_template
from .paths import (
    EXPERIMENTS_DIR,
    FEATURES_DIR,
    FOLDS_DIR,
    OOF_DIR,
    REPORT_EXPERIMENTS_DIR,
    REPORT_OOF_DIR,
    REPORT_SUBMISSIONS_DIR,
    SUBMISSIONS_DIR,
    ensure_runtime_dirs,
)
from .utils import binary_log_loss, clip_probabilities, multi_target_log_loss, write_markdown


DEFAULT_MODEL_FAMILY = "hgb_prior"
SUPPORTED_MODEL_FAMILIES = {"hgb_prior", "hgb_select_resid", "catboost"}
SUBJECT_PRIOR_SMOOTHING = 5.0
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
MIN_BLEND_GAIN = 0.30
FEATURE_SELECTION_TOP_K = 48
RESIDUAL_FEATURE_SELECTION_TOP_K = 8
RESIDUAL_TARGET_CLIP = 0.05
PRIOR_FEATURE_NAME = "subject_prior_feature"
FORCED_FEATURE_COLUMNS = {"d_dayofweek", "d_is_weekend", "d_month"}
NON_FEATURE_COLUMNS = set(KEY_COLUMNS + TARGET_COLUMNS + ["split"])


@dataclass
class FoldResult:
    fold_id: int
    raw_scores: dict[str, Any]
    calibrated_scores: dict[str, Any]
    blended_scores: dict[str, Any]
    blend_alphas: dict[str, float]
    selection_sizes: dict[str, int]


@dataclass
class TargetBundle:
    model: Any
    calibrator: Any
    blend_alpha: float
    global_prior: float
    subject_priors: dict[Any, float]
    model_columns: list[str]
    model_family: str


def load_feature_table(path: Path | None = None) -> pd.DataFrame:
    feature_path = path or (FEATURES_DIR / "daily_feature_table.parquet")
    return pd.read_parquet(feature_path)


def load_fold_manifest(path: Path | None = None) -> pd.DataFrame:
    manifest_path = path or (FOLDS_DIR / "fold_manifest.parquet")
    return pd.read_parquet(manifest_path)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_dummy_predictions(train: pd.DataFrame) -> pd.DataFrame:
    priors = {target: float(train[target].mean()) for target in TARGET_COLUMNS}
    return pd.DataFrame({target: np.full(len(train), prior) for target, prior in priors.items()}, index=train.index)


def _validate_model_family(model_family: str) -> str:
    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_family: {model_family}")
    if model_family == "catboost" and not HAS_CATBOOST:
        raise RuntimeError("catboost model_family was requested but CatBoost is not available.")
    return model_family


def _make_classifier_model(model_family: str) -> Any:
    if model_family == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            depth=6,
            learning_rate=0.05,
            iterations=350,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=3,
        max_iter=150,
        random_state=RANDOM_STATE,
    )


def _make_residual_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_depth=2,
        max_iter=30,
        random_state=RANDOM_STATE,
    )


def _model_feature_columns(feature_columns: list[str], model_family: str) -> list[str]:
    return feature_columns + ["subject_id"] if model_family == "catboost" else feature_columns


def _fit_model(model: Any, features: pd.DataFrame, target: pd.Series, model_family: str) -> Any:
    if model_family == "catboost":
        model.fit(features, target, cat_features=["subject_id"])
        return model
    model.fit(features, target)
    return model


def _fit_residual_model(model: HistGradientBoostingRegressor, features: pd.DataFrame, target: np.ndarray) -> HistGradientBoostingRegressor:
    model.fit(features, target)
    return model


def _absolute_correlation(values: pd.Series, target: np.ndarray) -> float:
    mask = values.notna().to_numpy()
    if mask.sum() < 3:
        return 0.0
    feature_values = values.to_numpy(dtype=float)[mask]
    target_values = target[mask]
    feature_std = float(np.std(feature_values))
    target_std = float(np.std(target_values))
    if feature_std == 0.0 or target_std == 0.0:
        return 0.0
    correlation = np.corrcoef(feature_values, target_values)[0, 1]
    if not np.isfinite(correlation):
        return 0.0
    return float(abs(correlation))


def _leave_one_out_prior_values(
    train_frame: pd.DataFrame,
    target: str,
    global_prior: float,
    smoothing_m: float = SUBJECT_PRIOR_SMOOTHING,
) -> np.ndarray:
    summary = train_frame.groupby("subject_id")[target].agg(["sum", "count"])
    subject_sum = train_frame["subject_id"].map(summary["sum"]).to_numpy(dtype=float)
    subject_count = train_frame["subject_id"].map(summary["count"]).to_numpy(dtype=float)
    targets = train_frame[target].to_numpy(dtype=float)
    numerators = subject_sum - targets + smoothing_m * global_prior
    denominators = np.maximum(subject_count - 1.0 + smoothing_m, smoothing_m)
    return clip_probabilities(numerators / denominators)


def _augment_with_subject_prior(frame: pd.DataFrame, subject_priors: dict[Any, float], global_prior: float) -> pd.DataFrame:
    augmented = frame.copy()
    augmented[PRIOR_FEATURE_NAME] = _subject_prior_values(augmented["subject_id"], subject_priors, global_prior)
    return augmented


def _select_target_features(
    fit_frame: pd.DataFrame,
    feature_columns: list[str],
    score_target: np.ndarray,
    model_family: str,
) -> list[str]:
    if model_family != "hgb_select_resid":
        return _model_feature_columns(feature_columns, model_family)

    scored_columns: list[tuple[float, str]] = []
    for column in feature_columns:
        score = _absolute_correlation(fit_frame[column], score_target)
        if score > 0.0:
            scored_columns.append((score, column))
    scored_columns.sort(key=lambda item: (-item[0], item[1]))

    top_k = RESIDUAL_FEATURE_SELECTION_TOP_K if model_family == "hgb_select_resid" else FEATURE_SELECTION_TOP_K
    selected = [column for _, column in scored_columns[:top_k]]
    for column in sorted(FORCED_FEATURE_COLUMNS):
        if column in fit_frame.columns:
            selected.append(column)
    selected.append(PRIOR_FEATURE_NAME)
    return list(dict.fromkeys(selected))


def _choose_calibration_indices(train_frame: pd.DataFrame, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    if scheme == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        fit_idx, calib_idx = next(splitter.split(train_frame, groups=train_frame["subject_id"]))
        return train_frame.index.to_numpy()[fit_idx], train_frame.index.to_numpy()[calib_idx]

    ordered = train_frame.sort_values(["subject_id", "lifelog_date"]).copy()
    ordered["rank"] = ordered.groupby("subject_id").cumcount()
    ordered["size"] = ordered.groupby("subject_id")["subject_id"].transform("size")
    cutoff = (ordered["size"] * 0.8).astype(int).clip(lower=1)
    fit_mask = ordered["rank"] < cutoff
    fit_idx = ordered.index[fit_mask].to_numpy()
    calib_idx = ordered.index[~fit_mask].to_numpy()
    if len(calib_idx) == 0:
        fit_idx = ordered.index[:-1].to_numpy()
        calib_idx = ordered.index[-1:].to_numpy()
    return fit_idx, calib_idx


def _outer_train_indices(train: pd.DataFrame, valid_frame: pd.DataFrame, split_scheme: str) -> pd.Index:
    if split_scheme == "group_time":
        valid_starts = valid_frame.groupby("subject_id")["sleep_date"].min().to_dict()
        train_mask = train.apply(
            lambda row: row["subject_id"] in valid_starts and row["sleep_date"] < valid_starts[row["subject_id"]],
            axis=1,
        )
        return train.index[train_mask & ~train.index.isin(valid_frame.index)]
    return train.index.difference(valid_frame.index)


def _predict_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    if isinstance(probabilities, list):
        probabilities = probabilities[0]
    return clip_probabilities(np.asarray(probabilities)[:, 1])


def _compute_subject_prior_table(
    train_frame: pd.DataFrame,
    target: str,
    smoothing_m: float = SUBJECT_PRIOR_SMOOTHING,
) -> tuple[float, dict[Any, float]]:
    global_prior = float(train_frame[target].mean())
    if len(train_frame) == 0:
        return global_prior, {}

    summary = train_frame.groupby("subject_id")[target].agg(["sum", "count"])
    smoothed = (summary["sum"] + smoothing_m * global_prior) / (summary["count"] + smoothing_m)
    return global_prior, {key: float(value) for key, value in smoothed.to_dict().items()}


def _subject_prior_values(subject_ids: pd.Series, subject_priors: dict[Any, float], global_prior: float) -> np.ndarray:
    values = subject_ids.map(subject_priors)
    return clip_probabilities(values.fillna(global_prior).to_numpy(dtype=float))


def _tune_blend_alpha(
    probabilities: np.ndarray,
    prior_values: np.ndarray,
    targets: np.ndarray,
    min_gain: float,
) -> tuple[float, np.ndarray]:
    prior_loss = binary_log_loss(targets, prior_values)
    best_alpha = 1.0
    best_loss = None
    best_values = clip_probabilities(probabilities)
    for alpha in ALPHA_GRID:
        blended = clip_probabilities(alpha * probabilities + (1.0 - alpha) * prior_values)
        loss = binary_log_loss(targets, blended)
        if best_loss is None or loss < best_loss:
            best_loss = float(loss)
            best_alpha = float(alpha)
            best_values = blended
    if best_loss is None or best_loss > prior_loss - min_gain:
        return 0.0, clip_probabilities(prior_values)
    return best_alpha, best_values


def _blend_min_gain(model_family: str) -> float:
    return MIN_BLEND_GAIN


def _fit_target_bundle(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    split_scheme: str,
    model_family: str,
    smoothing_m: float = SUBJECT_PRIOR_SMOOTHING,
) -> TargetBundle:
    fit_idx, calib_idx = _choose_calibration_indices(train_frame, split_scheme)
    fit_frame = train_frame.loc[fit_idx].copy()
    calib_frame = train_frame.loc[calib_idx].copy()

    calibrator: Any = IdentityCalibrator()
    blend_alpha = 1.0
    blend_min_gain = _blend_min_gain(model_family)
    fit_global_prior, fit_subject_priors = _compute_subject_prior_table(fit_frame, target, smoothing_m)

    if model_family == "hgb_select_resid":
        fit_frame_aug = fit_frame.copy()
        fit_frame_aug[PRIOR_FEATURE_NAME] = _leave_one_out_prior_values(
            fit_frame,
            target,
            fit_global_prior,
            smoothing_m,
        )
        residual_target = fit_frame[target].to_numpy(dtype=float) - fit_frame_aug[PRIOR_FEATURE_NAME].to_numpy(dtype=float)
        residual_target = np.clip(residual_target, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
        model_columns = _select_target_features(fit_frame_aug, feature_columns + [PRIOR_FEATURE_NAME], residual_target, model_family)
        calibration_model = _fit_residual_model(
            _make_residual_model(),
            fit_frame_aug[model_columns],
            residual_target,
        )

        if len(calib_frame) > 0:
            calib_frame_aug = _augment_with_subject_prior(calib_frame, fit_subject_priors, fit_global_prior)
            calib_prior = calib_frame_aug[PRIOR_FEATURE_NAME].to_numpy(dtype=float)
            calib_prob = clip_probabilities(calib_prior + calibration_model.predict(calib_frame_aug[model_columns]))
            calibrator = PlattCalibrator().fit(calib_prob, calib_frame[target].to_numpy())
            calibrated_calib = calibrator.predict(calib_prob)
            blend_alpha, _ = _tune_blend_alpha(
                calibrated_calib,
                calib_prior,
                calib_frame[target].to_numpy(dtype=float),
                min_gain=blend_min_gain,
            )

        global_prior, subject_priors = _compute_subject_prior_table(train_frame, target, smoothing_m)
        train_frame_aug = train_frame.copy()
        train_frame_aug[PRIOR_FEATURE_NAME] = _leave_one_out_prior_values(
            train_frame,
            target,
            global_prior,
            smoothing_m,
        )
        final_residual_target = train_frame[target].to_numpy(dtype=float) - train_frame_aug[PRIOR_FEATURE_NAME].to_numpy(dtype=float)
        final_residual_target = np.clip(final_residual_target, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
        final_model = _fit_residual_model(
            _make_residual_model(),
            train_frame_aug[model_columns],
            final_residual_target,
        )
        return TargetBundle(
            model=final_model,
            calibrator=calibrator,
            blend_alpha=blend_alpha,
            global_prior=global_prior,
            subject_priors=subject_priors,
            model_columns=model_columns,
            model_family=model_family,
        )

    model_columns = _select_target_features(fit_frame, feature_columns, fit_frame[target].to_numpy(dtype=float), model_family)
    calibration_model = _fit_model(
        _make_classifier_model(model_family),
        fit_frame[model_columns],
        fit_frame[target],
        model_family,
    )

    if len(calib_frame) > 0:
        calib_prob = _predict_probability(calibration_model, calib_frame[model_columns])
        calibrator = PlattCalibrator().fit(calib_prob, calib_frame[target].to_numpy())
        calibrated_calib = calibrator.predict(calib_prob)
        calib_prior = _subject_prior_values(calib_frame["subject_id"], fit_subject_priors, fit_global_prior)
        blend_alpha, _ = _tune_blend_alpha(
            calibrated_calib,
            calib_prior,
            calib_frame[target].to_numpy(dtype=float),
            min_gain=blend_min_gain,
        )

    final_model = _fit_model(
        _make_classifier_model(model_family),
        train_frame[model_columns],
        train_frame[target],
        model_family,
    )
    global_prior, subject_priors = _compute_subject_prior_table(train_frame, target, smoothing_m)
    return TargetBundle(
        model=final_model,
        calibrator=calibrator,
        blend_alpha=blend_alpha,
        global_prior=global_prior,
        subject_priors=subject_priors,
        model_columns=model_columns,
        model_family=model_family,
    )


def _predict_target_bundle(bundle: TargetBundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_prior = _subject_prior_values(frame["subject_id"], bundle.subject_priors, bundle.global_prior)
    if bundle.model_family == "hgb_select_resid":
        frame_aug = frame.copy()
        frame_aug[PRIOR_FEATURE_NAME] = subject_prior
        raw = clip_probabilities(subject_prior + bundle.model.predict(frame_aug[bundle.model_columns]))
    else:
        raw = _predict_probability(bundle.model, frame[bundle.model_columns])
    calibrated = bundle.calibrator.predict(raw)
    blended = clip_probabilities(bundle.blend_alpha * calibrated + (1.0 - bundle.blend_alpha) * subject_prior)
    return raw, calibrated, blended


def _build_valid_frame(train: pd.DataFrame, scheme_manifest: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    valid_keys = scheme_manifest[scheme_manifest["fold_id"] == fold_id][KEY_COLUMNS]
    valid_rows = train.reset_index().merge(valid_keys.assign(_valid=1), on=KEY_COLUMNS, how="left")
    valid_idx = valid_rows.loc[valid_rows["_valid"].fillna(0).eq(1), "index"].to_numpy()
    return train.loc[valid_idx].copy()


def _score_stage_export(
    scored_train: pd.DataFrame,
    oof_raw: pd.DataFrame,
    oof_cal: pd.DataFrame,
    oof_blend: pd.DataFrame,
    split_scheme: str,
    model_family: str,
) -> pd.DataFrame:
    export = scored_train[KEY_COLUMNS + TARGET_COLUMNS].copy()
    export["split_scheme"] = split_scheme
    export["model_family"] = model_family
    export["is_primary_scored"] = 1
    for target in TARGET_COLUMNS:
        export[f"{target}_raw"] = oof_raw.loc[scored_train.index, target].values
        export[f"{target}_cal"] = oof_cal.loc[scored_train.index, target].values
        export[f"{target}_blend"] = oof_blend.loc[scored_train.index, target].values
    return export


def _artifact_stem(model_family: str, split_scheme: str) -> str:
    return f"baseline_{model_family}_{split_scheme}"


def evaluate_baseline(
    split_scheme: str = "group",
    model_family: str = DEFAULT_MODEL_FAMILY,
    persist: bool = False,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    model_family = _validate_model_family(model_family)

    feature_table = load_feature_table()
    manifest = load_fold_manifest()
    train = feature_table[feature_table["split"] == "train"].copy()
    feature_columns = get_feature_columns(train)

    scheme_manifest = manifest[manifest["split_scheme"] == split_scheme].copy()
    oof_raw = pd.DataFrame(index=train.index, columns=TARGET_COLUMNS, dtype=float)
    oof_cal = pd.DataFrame(index=train.index, columns=TARGET_COLUMNS, dtype=float)
    oof_blend = pd.DataFrame(index=train.index, columns=TARGET_COLUMNS, dtype=float)
    fold_results: list[FoldResult] = []

    for fold_id in sorted(scheme_manifest["fold_id"].unique()):
        valid_frame = _build_valid_frame(train, scheme_manifest, fold_id)
        outer_train = train.loc[_outer_train_indices(train, valid_frame, split_scheme)].copy()
        if len(outer_train) == 0 or len(valid_frame) == 0:
            continue

        raw_fold = pd.DataFrame(index=valid_frame.index, columns=TARGET_COLUMNS, dtype=float)
        cal_fold = pd.DataFrame(index=valid_frame.index, columns=TARGET_COLUMNS, dtype=float)
        blend_fold = pd.DataFrame(index=valid_frame.index, columns=TARGET_COLUMNS, dtype=float)
        blend_alphas: dict[str, float] = {}
        selection_sizes: dict[str, int] = {}

        for target in TARGET_COLUMNS:
            bundle = _fit_target_bundle(
                outer_train,
                feature_columns,
                target,
                split_scheme,
                model_family=model_family,
            )
            raw, calibrated, blended = _predict_target_bundle(bundle, valid_frame)
            raw_fold[target] = raw
            cal_fold[target] = calibrated
            blend_fold[target] = blended
            blend_alphas[target] = float(bundle.blend_alpha)
            selection_sizes[target] = len(bundle.model_columns)

        oof_raw.loc[valid_frame.index, TARGET_COLUMNS] = raw_fold[TARGET_COLUMNS]
        oof_cal.loc[valid_frame.index, TARGET_COLUMNS] = cal_fold[TARGET_COLUMNS]
        oof_blend.loc[valid_frame.index, TARGET_COLUMNS] = blend_fold[TARGET_COLUMNS]
        fold_results.append(
            FoldResult(
                fold_id=int(fold_id),
                raw_scores=multi_target_log_loss(valid_frame[TARGET_COLUMNS], raw_fold, TARGET_COLUMNS),
                calibrated_scores=multi_target_log_loss(valid_frame[TARGET_COLUMNS], cal_fold, TARGET_COLUMNS),
                blended_scores=multi_target_log_loss(valid_frame[TARGET_COLUMNS], blend_fold, TARGET_COLUMNS),
                blend_alphas=blend_alphas,
                selection_sizes=selection_sizes,
            )
        )

    scored_mask = oof_blend[TARGET_COLUMNS].notna().all(axis=1)
    scored_train = train.loc[scored_mask].copy()
    raw_scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], oof_raw.loc[scored_mask], TARGET_COLUMNS)
    calibrated_scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], oof_cal.loc[scored_mask], TARGET_COLUMNS)
    blended_scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], oof_blend.loc[scored_mask], TARGET_COLUMNS)
    dummy_scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], build_dummy_predictions(scored_train), TARGET_COLUMNS)

    result = {
        "model_family": model_family,
        "feature_count": len(feature_columns),
        "raw_scores": raw_scores,
        "calibrated_scores": calibrated_scores,
        "blended_scores": blended_scores,
        "dummy_scores": dummy_scores,
        "fold_results": [result.__dict__ for result in fold_results],
    }

    if not persist:
        return result

    stem = _artifact_stem(model_family, split_scheme)
    oof_export = _score_stage_export(scored_train, oof_raw, oof_cal, oof_blend, split_scheme, model_family)
    oof_path = OOF_DIR / f"oof_predictions_{model_family}_{split_scheme}.parquet"
    oof_export.to_parquet(oof_path, index=False)

    report_lines = [
        f"# Baseline Report ({model_family}, {split_scheme})",
        "",
        f"- Model family: {model_family}",
        f"- Feature count: {len(feature_columns)}",
        f"- Prior type: subject_smoothed",
        f"- Prior smoothing m: {SUBJECT_PRIOR_SMOOTHING:.1f}",
        f"- Alpha selection: fold-internal calibration split, grid={len(ALPHA_GRID)}",
        f"- Minimum blend gain over pure prior: {_blend_min_gain(model_family):.3f}",
        f"- Dummy mean log-loss: {dummy_scores['mean']:.6f}",
        f"- Raw mean log-loss: {raw_scores['mean']:.6f}",
        f"- Calibrated mean log-loss: {calibrated_scores['mean']:.6f}",
        f"- Blended mean log-loss: {blended_scores['mean']:.6f}",
        "",
        "## Fold Scores",
        "",
    ]
    for fold_result in fold_results:
        alpha_summary = ", ".join(
            f"{target}={fold_result.blend_alphas[target]:.2f}" for target in TARGET_COLUMNS
        )
        selection_summary = ", ".join(
            f"{target}={fold_result.selection_sizes[target]}" for target in TARGET_COLUMNS
        )
        report_lines.append(
            f"- Fold {fold_result.fold_id}: raw={fold_result.raw_scores['mean']:.6f}, calibrated={fold_result.calibrated_scores['mean']:.6f}, blended={fold_result.blended_scores['mean']:.6f}"
        )
        report_lines.append(f"  alpha: {alpha_summary}")
        report_lines.append(f"  selected: {selection_summary}")

    report_text = "\n".join(report_lines)
    write_markdown(REPORT_OOF_DIR / f"{stem}.md", report_text)
    write_markdown(REPORT_EXPERIMENTS_DIR / f"{stem}.md", report_text)

    card = ExperimentCard(
        name=stem,
        model_family=model_family,
        feature_view="daily_feature_table",
        split_scheme=split_scheme,
        mean_log_loss=blended_scores["mean"],
        std_log_loss=blended_scores["std"],
        target_scores={target: float(blended_scores[target]) for target in TARGET_COLUMNS},
        calibration="per-target platt scaling with fold-internal alpha tuning against smoothed subject prior",
        improvement_over_dummy=float(dummy_scores["mean"] - blended_scores["mean"]),
        accepted=bool(blended_scores["mean"] <= calibrated_scores["mean"] and blended_scores["mean"] < dummy_scores["mean"]),
        paper_relevance="Captures whether subject baseline plus day-level modality summaries explain future states under strict time-aware validation.",
        score_breakdown={
            "dummy": {target: float(dummy_scores[target]) for target in TARGET_COLUMNS} | {"mean": float(dummy_scores["mean"]), "std": float(dummy_scores["std"])},
            "raw": {target: float(raw_scores[target]) for target in TARGET_COLUMNS} | {"mean": float(raw_scores["mean"]), "std": float(raw_scores["std"])},
            "calibrated": {target: float(calibrated_scores[target]) for target in TARGET_COLUMNS} | {"mean": float(calibrated_scores["mean"]), "std": float(calibrated_scores["std"])},
            "blended": {target: float(blended_scores[target]) for target in TARGET_COLUMNS} | {"mean": float(blended_scores["mean"]), "std": float(blended_scores["std"])},
        },
        notes=[
            "prior_type=subject_smoothed",
            f"smoothing_m={SUBJECT_PRIOR_SMOOTHING:.1f}",
            "alpha_selection=fold_internal",
            f"min_blend_gain={_blend_min_gain(model_family):.3f}",
            f"residual_target_clip={RESIDUAL_TARGET_CLIP:.2f}" if model_family == "hgb_select_resid" else "residual_target_clip=none",
            "official_score_stage=blended",
            "Sequence-lite should only continue if it beats the blended OOF score.",
        ],
    )
    experiment_path = EXPERIMENTS_DIR / f"{stem}.json"
    save_experiment_card(experiment_path, card)

    result["artifacts"] = {
        "oof_path": str(oof_path),
        "report_path": str(REPORT_OOF_DIR / f"{stem}.md"),
        "experiment_path": str(experiment_path),
    }
    return result


def train_baseline(
    split_scheme: str = "group",
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> dict[str, Any]:
    return evaluate_baseline(split_scheme=split_scheme, model_family=model_family, persist=True)


def evaluate_subject_prior(
    split_scheme: str = "group_time",
    smoothing_m: float = SUBJECT_PRIOR_SMOOTHING,
) -> dict[str, Any]:
    feature_table = load_feature_table()
    manifest = load_fold_manifest()
    train = feature_table[feature_table["split"] == "train"].copy()
    scheme_manifest = manifest[manifest["split_scheme"] == split_scheme].copy()

    oof = pd.DataFrame(index=train.index, columns=TARGET_COLUMNS, dtype=float)
    for fold_id in sorted(scheme_manifest["fold_id"].unique()):
        valid_frame = _build_valid_frame(train, scheme_manifest, fold_id)
        outer_train = train.loc[_outer_train_indices(train, valid_frame, split_scheme)].copy()
        if len(outer_train) == 0 or len(valid_frame) == 0:
            continue

        for target in TARGET_COLUMNS:
            global_prior, subject_priors = _compute_subject_prior_table(outer_train, target, smoothing_m)
            oof.loc[valid_frame.index, target] = _subject_prior_values(
                valid_frame["subject_id"],
                subject_priors,
                global_prior,
            )

    scored_mask = oof[TARGET_COLUMNS].notna().all(axis=1)
    scored_train = train.loc[scored_mask].copy()
    subject_scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], oof.loc[scored_mask], TARGET_COLUMNS)
    dummy_scores = multi_target_log_loss(scored_train[TARGET_COLUMNS], build_dummy_predictions(scored_train), TARGET_COLUMNS)
    return {
        "scores": subject_scores,
        "dummy_scores": dummy_scores,
        "improvement_over_dummy": float(dummy_scores["mean"] - subject_scores["mean"]),
    }


def make_submission(
    split_scheme: str = "group_time",
    tag: str = "baseline",
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> Path:
    ensure_runtime_dirs()
    model_family = _validate_model_family(model_family)

    feature_table = load_feature_table()
    train = feature_table[feature_table["split"] == "train"].copy()
    test = feature_table[feature_table["split"] == "test"].copy()
    feature_columns = get_feature_columns(train)

    predictions = load_submission_template()[KEY_COLUMNS].copy()
    report_lines = [
        f"# Submission Report ({model_family}, {split_scheme})",
        "",
        f"- Model family: {model_family}",
        f"- Feature count: {len(feature_columns)}",
        f"- Train rows: {len(train)}",
        f"- Test rows: {len(test)}",
        f"- Prior smoothing m: {SUBJECT_PRIOR_SMOOTHING:.1f}",
        "",
        "## Target Bundles",
        "",
    ]

    for target in TARGET_COLUMNS:
        bundle = _fit_target_bundle(
            train,
            feature_columns,
            target,
            split_scheme,
            model_family=model_family,
        )
        _, _, blended = _predict_target_bundle(bundle, test)
        predictions[target] = blended
        seen_subject_rate = float(test["subject_id"].isin(bundle.subject_priors).mean())
        report_lines.append(
            f"- `{target}`: global_prior={bundle.global_prior:.4f}, blend_alpha={bundle.blend_alpha:.2f}, selected_features={len(bundle.model_columns)}, seen_subject_rate={seen_subject_rate:.3f}"
        )

    output_path = SUBMISSIONS_DIR / f"submission_{tag}_{model_family}_{split_scheme}.csv"
    predictions.to_csv(output_path, index=False)
    write_markdown(
        REPORT_SUBMISSIONS_DIR / f"submission_{tag}_{model_family}_{split_scheme}.md",
        "\n".join(report_lines),
    )
    return output_path

"""Notebook-inspired public leaderboard LightGBM pipeline."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    lgb = None
    HAS_LIGHTGBM = False

from .constants import KEY_COLUMNS, TARGET_COLUMNS
from .features import HISTORY_ANCHORS
from .io import load_modality_frame, load_submission_template
from .paths import (
    FEATURES_DIR,
    MODELS_DIR,
    OOF_DIR,
    RAW_MODALITY_DIR,
    REPORT_FEATURES_DIR,
    REPORT_OOF_DIR,
    REPORT_SUBMISSIONS_DIR,
    SUBMISSIONS_DIR,
    ensure_runtime_dirs,
)
from .utils import clip_probabilities, multi_target_log_loss, write_json, write_markdown


PUBLIC_LGB_DEFAULT_SEEDS = [42, 1234, 9999, 7, 314, 2025, 777, 555]
PUBLIC_LGB_FEATURE_VIEWS = {
    "public_core",
    "public_full",
    "public_notebook",
    "public_hist365",
    "public_hist411",
}
PUBLIC_LGB_TARGETWISE_PRESETS = {
    "histmix_guarded_v1": {
        "Q2": "public_hist411",
        "Q3": "public_hist365",
        "S4": "public_hist411",
    },
    "histmix_guarded_v1_tuned": {
        "Q2": "public_hist411",
        "Q3": "public_hist365",
        "S4": "public_hist411",
    },
    "histmix_q2s4_v1": {
        "Q2": "public_hist411",
        "S4": "public_hist411",
    },
    "histmix_q2q3_v1": {
        "Q2": "public_hist411",
        "Q3": "public_hist365",
    },
    "histmix_q3s4_v1": {
        "Q3": "public_hist365",
        "S4": "public_hist411",
    },
    "histmix_aggressive_v1": {
        "Q2": "public_hist411",
        "Q3": "public_hist365",
        "S1": "public_hist411",
        "S4": "public_hist411",
    },
}
TARGET_ENCODING_WINDOWS = [3, 7, 14, 21]
PUBLIC_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.3,
    "reg_lambda": 2.0,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
}


def _feature_table_path() -> Path:
    return FEATURES_DIR / "public_lgb_feature_table.parquet"


def _run_name_for_feature_view(feature_view: str) -> str:
    return f"public_lgb_{feature_view}"


def _run_name_for_preset(preset_name: str) -> str:
    return f"public_lgb_targetwise_{preset_name}"


def _test_prediction_path_for_run(run_name: str) -> Path:
    return MODELS_DIR / f"test_predictions_{run_name}.csv"


def _summary_path_for_run(run_name: str) -> Path:
    return FEATURES_DIR / f"{run_name}_summary.json"


def _test_prediction_path(feature_view: str) -> Path:
    return _test_prediction_path_for_run(_run_name_for_feature_view(feature_view))


def _summary_path(feature_view: str) -> Path:
    return _summary_path_for_run(_run_name_for_feature_view(feature_view))


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q90": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_median": float(np.nanmedian(values)),
        f"{prefix}_q10": float(np.nanpercentile(values, 10)),
        f"{prefix}_q90": float(np.nanpercentile(values, 90)),
    }


def _sleep_window(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output = output[output["timestamp"].dt.hour < 9].copy()
    output["sleep_date"] = output["timestamp"].dt.normalize()
    return output


def _extract_sleep_hr_features() -> pd.DataFrame:
    frame = _sleep_window(load_modality_frame(RAW_MODALITY_DIR / "ch2025_wHr.parquet"))
    records: list[dict[str, object]] = []
    for (subject_id, sleep_date), group in frame.groupby(["subject_id", "sleep_date"], sort=False):
        samples: list[np.ndarray] = []
        hour_counts = group["timestamp"].dt.hour.nunique()
        for payload in group["heart_rate"]:
            try:
                values = np.asarray(payload, dtype=float).ravel()
            except Exception:
                continue
            values = values[np.isfinite(values) & (values > 0)]
            if values.size:
                samples.append(values)
        flat = np.concatenate(samples) if samples else np.array([], dtype=float)
        diffs = np.diff(flat) if flat.size > 1 else np.array([], dtype=float)
        row = {
            "subject_id": subject_id,
            "sleep_date": sleep_date,
            "s_hr_sample_cnt": int(flat.size),
            "s_hr_hour_coverage": float(hour_counts / 9.0) if hour_counts else 0.0,
            "s_hr_rmssd": float(np.sqrt(np.mean(np.square(diffs)))) if diffs.size else np.nan,
            "s_hr_spike_share": float(np.mean(np.abs(diffs) >= 10.0)) if diffs.size else np.nan,
        }
        row.update(_safe_stats(flat, "s_hr"))
        records.append(row)
    return pd.DataFrame(records)


def _extract_sleep_pedo_features() -> pd.DataFrame:
    frame = _sleep_window(load_modality_frame(RAW_MODALITY_DIR / "ch2025_wPedo.parquet"))
    if frame.empty:
        return pd.DataFrame(columns=["subject_id", "sleep_date"])
    grouped = frame.groupby(["subject_id", "sleep_date"], sort=False)
    output = grouped.agg(
        s_pedo_row_cnt=("step", "size"),
        s_pedo_step_sum=("step", "sum"),
        s_pedo_distance_sum=("distance", "sum"),
        s_pedo_calories_sum=("burned_calories", "sum"),
        s_pedo_speed_mean=("speed", "mean"),
        s_pedo_speed_p90=("speed", lambda x: float(np.nanpercentile(x, 90)) if len(x) else np.nan),
        s_pedo_active_min=("step", lambda x: int(np.sum(np.asarray(x) > 0))),
        s_pedo_zero_step_share=("step", lambda x: float(np.mean(np.asarray(x) <= 0))),
    )
    return output.reset_index()


def _extract_sleep_activity_features() -> pd.DataFrame:
    frame = _sleep_window(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mActivity.parquet"))
    records: list[dict[str, object]] = []
    active_codes = np.array([3, 7, 8], dtype=int)
    tracked_codes = [0, 3, 4, 7, 8]
    for (subject_id, sleep_date), group in frame.groupby(["subject_id", "sleep_date"], sort=False):
        values = group["m_activity"].to_numpy(dtype=int)
        row: dict[str, object] = {
            "subject_id": subject_id,
            "sleep_date": sleep_date,
            "s_act_row_cnt": int(values.size),
            "s_act_active_ratio": float(np.mean(np.isin(values, active_codes))) if values.size else np.nan,
            "s_act_still_ratio": float(np.mean(values == 0)) if values.size else np.nan,
        }
        for code in tracked_codes:
            row[f"s_act_code{code}_share"] = float(np.mean(values == code)) if values.size else np.nan
        records.append(row)
    return pd.DataFrame(records)


def _extract_sleep_screen_features() -> pd.DataFrame:
    frame = _sleep_window(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mScreenStatus.parquet"))
    records: list[dict[str, object]] = []
    for (subject_id, sleep_date), group in frame.groupby(["subject_id", "sleep_date"], sort=False):
        ordered = group.sort_values("timestamp")
        active = (ordered["m_screen_use"].to_numpy(dtype=float) > 0).astype(int)
        transitions = np.diff(np.r_[0, active]) if active.size else np.array([], dtype=int)
        row = {
            "subject_id": subject_id,
            "sleep_date": sleep_date,
            "s_screen_row_cnt": int(active.size),
            "s_screen_on_min": int(active.sum()),
            "s_screen_on_share": float(active.mean()) if active.size else np.nan,
            "s_screen_session_cnt": int(np.sum(transitions == 1)) if transitions.size else 0,
            "s_screen_longest_on_min": int(_longest_run(active)),
        }
        records.append(row)
    return pd.DataFrame(records)


def _extract_sleep_light_features() -> pd.DataFrame:
    frame = _sleep_window(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mLight.parquet"))
    records: list[dict[str, object]] = []
    for (subject_id, sleep_date), group in frame.groupby(["subject_id", "sleep_date"], sort=False):
        values = group["m_light"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        log_values = np.log1p(np.clip(values, 0.0, None)) if values.size else np.array([], dtype=float)
        row = {
            "subject_id": subject_id,
            "sleep_date": sleep_date,
            "s_light_row_cnt": int(values.size),
            "s_light_zero_share": float(np.mean(values <= 0)) if values.size else np.nan,
        }
        row.update(_safe_stats(log_values, "s_light_log"))
        records.append(row)
    return pd.DataFrame(records)


def _longest_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def build_sleep_feature_table(base_frame: pd.DataFrame) -> pd.DataFrame:
    keys = base_frame[["subject_id", "sleep_date"]].drop_duplicates().copy()
    for feature_frame in [
        _extract_sleep_hr_features(),
        _extract_sleep_pedo_features(),
        _extract_sleep_activity_features(),
        _extract_sleep_screen_features(),
        _extract_sleep_light_features(),
    ]:
        keys = keys.merge(feature_frame, on=["subject_id", "sleep_date"], how="left")
    return keys


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["dow"] = output["lifelog_date"].dt.dayofweek
    output["month"] = output["lifelog_date"].dt.month
    output["week"] = output["lifelog_date"].dt.isocalendar().week.astype(int)
    output["is_weekend"] = (output["dow"] >= 5).astype(int)
    output["subject_num"] = (
        output["subject_id"].astype(str).str.extract(r"(\d+)").iloc[:, 0].fillna(-1).astype(int)
    )
    return output


def _add_subject_zscores(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    numeric_cols = output.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set(TARGET_COLUMNS + ["dow", "month", "week", "is_weekend", "subject_num"])
    zscore_payload: dict[str, pd.Series] = {}
    zscore_cols: list[str] = []

    # train 행만으로 subject별 통계 계산 (T2: 누수 제거)
    train_mask = output["split"] == "train"

    for col in numeric_cols:
        if col in excluded or "__" in col:
            continue

        # train 행에서만 평균/표준편차 계산
        train_stats = (
            output.loc[train_mask]
            .groupby("subject_id")[col]
            .agg(mu="mean", sig="std")
        )

        # train 파라미터를 train+test 전체에 적용
        joined = output[["subject_id"]].join(train_stats, on="subject_id")
        mu  = joined["mu"]
        sig = joined["sig"].replace(0.0, np.nan)

        z_col = f"{col}__subj_z"
        zscore_payload[z_col] = (output[col] - mu) / sig
        zscore_cols.append(z_col)

    if zscore_payload:
        output = pd.concat(
            [output, pd.DataFrame(zscore_payload, index=output.index)], axis=1
        )
    return output, zscore_cols


def _add_target_encodings(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    ordered = output.sort_values(["subject_id", "lifelog_date"]).copy()
    encoding_cols: list[str] = []
    for target in TARGET_COLUMNS:
        lag_col = f"{target}_te_lag1"
        ordered[lag_col] = ordered.groupby("subject_id")[target].shift(1)
        encoding_cols.append(lag_col)
        for window in TARGET_ENCODING_WINDOWS:
            col = f"{target}_te_{window}"
            ordered[col] = ordered.groupby("subject_id")[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            encoding_cols.append(col)
    output[encoding_cols] = ordered[encoding_cols]
    return output, encoding_cols


def build_public_lgb_feature_table(persist: bool = True, suffix: str = "") -> pd.DataFrame:
    ensure_runtime_dirs()
    base = pd.read_parquet(FEATURES_DIR / "daily_feature_table.parquet").copy()
    base = base.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    sleep_features = build_sleep_feature_table(base)
    feature_table = base.merge(sleep_features, on=["subject_id", "sleep_date"], how="left")
    feature_table = _add_time_features(feature_table)
    feature_table, zscore_cols = _add_subject_zscores(feature_table)
    feature_table, te_cols = _add_target_encodings(feature_table)

    if persist:
        fname = f"public_lgb_feature_table{suffix}.parquet"
        path = FEATURES_DIR / fname
        feature_table.to_parquet(path, index=False)
        raw_feature_count = len(
            [
                col
                for col in feature_table.columns
                if col not in KEY_COLUMNS + TARGET_COLUMNS + ["split"] and "__subj_z" not in col and "_te_" not in col
            ]
        )
        report = "\n".join(
            [
                "# Public LGB Feature Table",
                "",
                f"- Rows: {len(feature_table)}",
                f"- Total columns: {len(feature_table.columns)}",
                f"- Raw feature columns: {raw_feature_count}",
                f"- Added sleep columns: {len([c for c in sleep_features.columns if c not in ['subject_id', 'sleep_date']])}",
                f"- Added subject z-score columns: {len(zscore_cols)}",
                f"- Added target encoding columns: {len(te_cols)}",
                "- Note: subject z-score is transductive and uses train+test grouped by subject_id.",
            ]
        )
        write_markdown(REPORT_FEATURES_DIR / "public_lgb_feature_table.md", report)
    return feature_table


def load_public_lgb_feature_table(rebuild: bool = False) -> pd.DataFrame:
    path = _feature_table_path()
    if rebuild or not path.exists():
        return build_public_lgb_feature_table(persist=True)
    return pd.read_parquet(path)


def _ordered_numeric_columns(frame: pd.DataFrame) -> list[str]:
    exclude = set(KEY_COLUMNS + TARGET_COLUMNS + ["split"])
    return [col for col in frame.columns if col not in exclude and pd.api.types.is_numeric_dtype(frame[col])]


def _ordered_subset(numeric_cols: list[str], selected: set[str]) -> list[str]:
    return [col for col in numeric_cols if col in selected]


def _build_feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = _ordered_numeric_columns(frame)
    time_cols = [col for col in ["dow", "month", "week", "is_weekend", "subject_num"] if col in numeric_cols]
    target_encoding_cols = [col for col in numeric_cols if "_te_" in col]
    raw_day_core = [
        col
        for col in numeric_cols
        if "__" not in col
        and (
            col.startswith("tb_")
            or (
                col.startswith("d_")
                and not col.startswith(("d_behavior", "d_modal"))
            )
        )
    ]
    sleep_raw_cols = [col for col in numeric_cols if col.startswith("s_") and "__" not in col]
    day_zscore_cols = [
        col
        for col in numeric_cols
        if col.endswith("__subj_z")
        and not col.startswith("s_")
        and (
            col[:-8].startswith("tb_")
            or (
                col[:-8].startswith("d_")
                and not col[:-8].startswith(("d_behavior", "d_modal"))
            )
        )
    ]
    anchor_zscore_cols = [
        col
        for col in numeric_cols
        if col.endswith("__subj_z") and col[:-8] in HISTORY_ANCHORS
    ]
    anchor_lag_roll_cols = [
        col
        for col in numeric_cols
        if any(col.startswith(f"{anchor}__") for anchor in HISTORY_ANCHORS)
        and any(token in col for token in ["__lag1", "__roll_mean_3", "__roll_mean_7", "__roll_mean_14"])
    ]
    anchor_simple_history_cols = [
        col
        for col in numeric_cols
        if any(col.startswith(f"{anchor}__") for anchor in HISTORY_ANCHORS)
        and any(
            token in col
            for token in [
                "__lag1",
                "__roll_mean_3",
                "__roll_mean_7",
                "__roll_mean_14",
                "__roll_std_7",
                "__roll_std_14",
            ]
        )
    ]
    return {
        "numeric_cols": numeric_cols,
        "time_cols": time_cols,
        "target_encoding_cols": target_encoding_cols,
        "raw_day_core": raw_day_core,
        "sleep_raw_cols": sleep_raw_cols,
        "day_zscore_cols": day_zscore_cols,
        "anchor_zscore_cols": anchor_zscore_cols,
        "anchor_lag_roll_cols": anchor_lag_roll_cols,
        "anchor_simple_history_cols": anchor_simple_history_cols,
    }


def get_public_lgb_feature_columns(frame: pd.DataFrame, feature_view: str = "public_core") -> list[str]:
    if feature_view not in PUBLIC_LGB_FEATURE_VIEWS:
        raise ValueError(f"Unsupported feature_view: {feature_view}")
    groups = _build_feature_groups(frame)
    numeric_cols = groups["numeric_cols"]
    if feature_view == "public_full":
        return numeric_cols
    if feature_view == "public_notebook":
        selected = (
            set(groups["raw_day_core"])
            | set(groups["sleep_raw_cols"])
            | set(groups["day_zscore_cols"])
            | set(groups["target_encoding_cols"])
            | set(groups["time_cols"])
        )
        return _ordered_subset(numeric_cols, selected)
    if feature_view == "public_hist365":
        selected = (
            set(groups["raw_day_core"])
            | set(groups["sleep_raw_cols"])
            | set(groups["anchor_lag_roll_cols"])
            | set(groups["anchor_zscore_cols"])
            | set(groups["target_encoding_cols"])
            | set(groups["time_cols"])
        )
        return _ordered_subset(numeric_cols, selected)
    if feature_view == "public_hist411":
        selected = (
            set(groups["raw_day_core"])
            | set(groups["sleep_raw_cols"])
            | set(groups["anchor_simple_history_cols"])
            | set(groups["anchor_zscore_cols"])
            | set(groups["target_encoding_cols"])
            | set(groups["time_cols"])
        )
        return _ordered_subset(numeric_cols, selected)
    selected: list[str] = []
    for col in numeric_cols:
        if "__" in col and not col.endswith("__subj_z"):
            continue
        selected.append(col)
    return selected


def resolve_target_feature_views(
    *,
    default_feature_view: str = "public_core",
    preset_name: str | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    if default_feature_view not in PUBLIC_LGB_FEATURE_VIEWS:
        raise ValueError(f"Unsupported default_feature_view: {default_feature_view}")
    mapping = {target: default_feature_view for target in TARGET_COLUMNS}
    if preset_name:
        if preset_name not in PUBLIC_LGB_TARGETWISE_PRESETS:
            raise ValueError(f"Unsupported targetwise preset: {preset_name}")
        mapping.update(PUBLIC_LGB_TARGETWISE_PRESETS[preset_name])
    if overrides:
        mapping.update(overrides)
    invalid = {target: view for target, view in mapping.items() if view not in PUBLIC_LGB_FEATURE_VIEWS}
    if invalid:
        raise ValueError(f"Unsupported feature views in target mapping: {invalid}")
    return mapping


def resolve_seed_list(seed_values: list[int] | None = None) -> list[int]:
    if seed_values:
        return [int(seed) for seed in seed_values]
    return PUBLIC_LGB_DEFAULT_SEEDS.copy()


def _train_public_lgb_with_target_views(
    *,
    run_name: str,
    target_feature_views: dict[str, str],
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    persist: bool = True,
    cv_scheme: str = "public_stratified",
    use_target_params: bool = False,
) -> dict[str, Any]:
    if not HAS_LIGHTGBM:
        raise RuntimeError("lightgbm is not installed. Install it into .vendor before training public_lgb.")

    ensure_runtime_dirs()
    frame = load_public_lgb_feature_table(rebuild=rebuild_features)
    train = frame[frame["split"] == "train"].reset_index(drop=True).copy()
    test = frame[frame["split"] == "test"].reset_index(drop=True).copy()
    seed_list = resolve_seed_list(seeds)

    distinct_views = sorted(set(target_feature_views.values()))
    feature_cols_by_view = {
        feature_view: get_public_lgb_feature_columns(frame, feature_view=feature_view) for feature_view in distinct_views
    }
    train_payload_by_view = {
        feature_view: train[feature_cols].copy() for feature_view, feature_cols in feature_cols_by_view.items()
    }
    test_payload_by_view = {
        feature_view: test[feature_cols].copy() for feature_view, feature_cols in feature_cols_by_view.items()
    }

    oof_preds = np.zeros((len(train), len(TARGET_COLUMNS)), dtype=float)
    test_preds = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
    seed_scores: dict[str, list[dict[str, Any]]] = {}

    for target_idx, target in enumerate(TARGET_COLUMNS):
        feature_view = target_feature_views[target]
        X_train = train_payload_by_view[feature_view]
        X_test = test_payload_by_view[feature_view]
        y = train[target].astype(int).to_numpy()
        target_oof = np.zeros(len(train), dtype=float)
        target_test = np.zeros(len(test), dtype=float)
        target_seed_scores: list[dict[str, Any]] = []
        for seed in seed_list:
            if cv_scheme == "subject_holdout":
                from .proper_cv import subject_stratified_holdout_iter
                fold_splits = list(
                    subject_stratified_holdout_iter(train, n_folds=n_folds, random_state=int(seed))
                )
            else:
                splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
                fold_splits = list(splitter.split(X_train, y))

            seed_oof = np.zeros(len(train), dtype=float)
            seed_test = np.zeros(len(test), dtype=float)

            if use_target_params:
                from .lgb_target_params import TARGET_LGB_PARAMS
                base_params = TARGET_LGB_PARAMS.get(target, PUBLIC_LGB_PARAMS)
                params = {**base_params, "random_state": int(seed)}
            else:
                params = {**PUBLIC_LGB_PARAMS, "random_state": int(seed)}

            for train_idx, valid_idx in fold_splits:
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train.iloc[train_idx],
                    y[train_idx],
                    eval_set=[(X_train.iloc[valid_idx], y[valid_idx])],
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
                )
                seed_oof[valid_idx] = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
                seed_test += model.predict_proba(X_test)[:, 1] / n_folds
            target_seed_scores.append({"seed": int(seed), "log_loss": float(log_loss(y, clip_probabilities(seed_oof)))})
            target_oof += seed_oof
            target_test += seed_test
        oof_preds[:, target_idx] = target_oof / len(seed_list)
        test_preds[:, target_idx] = target_test / len(seed_list)
        seed_scores[target] = target_seed_scores

    oof_frame = pd.DataFrame(oof_preds, columns=TARGET_COLUMNS)
    score_payload = multi_target_log_loss(train[TARGET_COLUMNS], oof_frame, TARGET_COLUMNS)
    feature_counts_by_view = {feature_view: len(cols) for feature_view, cols in feature_cols_by_view.items()}
    result = {
        "name": run_name,
        "target_feature_views": target_feature_views,
        "n_features_by_view": feature_counts_by_view,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_folds": int(n_folds),
        "seeds": [int(seed) for seed in seed_list],
        "scores": score_payload,
        "seed_scores": seed_scores,
    }
    if len(distinct_views) == 1:
        result["feature_view"] = distinct_views[0]
        result["n_features"] = feature_counts_by_view[distinct_views[0]]

    if not persist:
        result["train_predictions"] = oof_frame
        result["test_predictions"] = pd.DataFrame(test_preds, columns=TARGET_COLUMNS)
        return result

    oof_export = train[KEY_COLUMNS + TARGET_COLUMNS].copy()
    oof_export["split_scheme"] = cv_scheme
    oof_export["model_family"] = run_name
    for target in TARGET_COLUMNS:
        oof_export[f"{target}_public_lgb"] = oof_frame[target].values
    oof_path = OOF_DIR / f"oof_predictions_{run_name}.parquet"
    oof_export.to_parquet(oof_path, index=False)

    test_pred_path = _test_prediction_path_for_run(run_name)
    pd.DataFrame(test_preds, columns=TARGET_COLUMNS).to_csv(test_pred_path, index=False)

    report_lines = [
        f"# Public LGB Report ({run_name})",
        "",
        f"- Train rows: {len(train)}",
        f"- Test rows: {len(test)}",
        f"- Folds: {n_folds}",
        f"- Seeds: {seed_list}",
        f"- Mean OOF log-loss: {score_payload['mean']:.6f}",
        "",
        "## Target Views",
        "",
    ]
    for target in TARGET_COLUMNS:
        report_lines.append(f"- `{target}`: `{target_feature_views[target]}`")
    report_lines.extend(["", "## Feature Counts By View", ""])
    for feature_view in distinct_views:
        report_lines.append(f"- `{feature_view}`: {feature_counts_by_view[feature_view]}")
    report_lines.extend(["", "## Target Scores", ""])
    for target in TARGET_COLUMNS:
        report_lines.append(f"- `{target}`: {score_payload[target]:.6f}")
    report_path = REPORT_OOF_DIR / f"{run_name}.md"
    write_markdown(report_path, "\n".join(report_lines))
    summary_path = _summary_path_for_run(run_name)
    write_json(
        summary_path,
        {
            "run_name": run_name,
            "feature_view": result.get("feature_view"),
            "n_features": result.get("n_features"),
            "target_feature_views": target_feature_views,
            "n_features_by_view": feature_counts_by_view,
            "seeds": seed_list,
            "n_folds": int(n_folds),
            "scores": score_payload,
        },
    )

    result["artifacts"] = {
        "oof_path": str(oof_path),
        "test_prediction_path": str(test_pred_path),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }
    return result


def train_public_lgb(
    feature_view: str = "public_core",
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    persist: bool = True,
    cv_scheme: str = "public_stratified",
    use_target_params: bool = False,
) -> dict[str, Any]:
    return _train_public_lgb_with_target_views(
        run_name=_run_name_for_feature_view(feature_view),
        target_feature_views=resolve_target_feature_views(default_feature_view=feature_view),
        n_folds=n_folds,
        seeds=seeds,
        rebuild_features=rebuild_features,
        persist=persist,
        cv_scheme=cv_scheme,
        use_target_params=use_target_params,
    )


def train_public_lgb_targetwise(
    *,
    preset_name: str,
    default_feature_view: str = "public_core",
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    persist: bool = True,
    cv_scheme: str = "public_stratified",
    use_target_params: bool = False,
) -> dict[str, Any]:
    return _train_public_lgb_with_target_views(
        run_name=_run_name_for_preset(preset_name),
        target_feature_views=resolve_target_feature_views(
            default_feature_view=default_feature_view,
            preset_name=preset_name,
        ),
        n_folds=n_folds,
        seeds=seeds,
        rebuild_features=rebuild_features,
        persist=persist,
        cv_scheme=cv_scheme,
        use_target_params=use_target_params,
    )


def make_public_lgb_submission(
    tag: str = "public_lgb_v1",
    feature_view: str = "public_core",
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    clip_min: float = 0.02,
    clip_max: float = 0.98,
) -> Path:
    ensure_runtime_dirs()
    cached_test_path = _test_prediction_path(feature_view)
    cached_summary_path = _summary_path(feature_view)
    requested_seeds = resolve_seed_list(seeds)
    can_reuse_cache = False
    if cached_test_path.exists() and cached_summary_path.exists() and not rebuild_features:
        cached_result = json.loads(cached_summary_path.read_text())
        can_reuse_cache = (
            cached_result.get("feature_view") == feature_view
            and int(cached_result.get("n_folds", -1)) == int(n_folds)
            and [int(seed) for seed in cached_result.get("seeds", [])] == requested_seeds
        )
    if can_reuse_cache:
        result = cached_result
        test_preds = pd.read_csv(cached_test_path)
    else:
        result = train_public_lgb(
            feature_view=feature_view,
            n_folds=n_folds,
            seeds=requested_seeds,
            rebuild_features=rebuild_features,
            persist=True,
        )
        test_preds = pd.read_csv(cached_test_path)
    submission = load_submission_template()[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = np.clip(test_preds[target].to_numpy(dtype=float), clip_min, clip_max)
    output_path = SUBMISSIONS_DIR / f"submission_{tag}_public_lgb_{feature_view}.csv"
    submission.to_csv(output_path, index=False)

    report_lines = [
        f"# Public LGB Submission Report ({feature_view})",
        "",
        f"- Source OOF mean log-loss: {result['scores']['mean']:.6f}",
        f"- Feature count: {result['n_features']}",
        f"- Folds: {n_folds}",
        f"- Seeds: {result['seeds']}",
        f"- Clip range: [{clip_min:.2f}, {clip_max:.2f}]",
        "",
        "## Target Scores",
        "",
    ]
    for target in TARGET_COLUMNS:
        report_lines.append(f"- `{target}`: {result['scores'][target]:.6f}")
    write_markdown(REPORT_SUBMISSIONS_DIR / f"submission_{tag}_public_lgb_{feature_view}.md", "\n".join(report_lines))
    return output_path


def make_public_lgb_targetwise_submission(
    *,
    tag: str = "public_lgb_v3",
    preset_name: str,
    default_feature_view: str = "public_core",
    n_folds: int = 5,
    seeds: list[int] | None = None,
    rebuild_features: bool = False,
    clip_min: float = 0.02,
    clip_max: float = 0.98,
    cv_scheme: str = "public_stratified",
    use_target_params: bool = False,
) -> Path:
    ensure_runtime_dirs()
    run_name = _run_name_for_preset(preset_name)
    cached_test_path = _test_prediction_path_for_run(run_name)
    cached_summary_path = _summary_path_for_run(run_name)
    requested_seeds = resolve_seed_list(seeds)
    target_feature_views = resolve_target_feature_views(
        default_feature_view=default_feature_view,
        preset_name=preset_name,
    )
    can_reuse_cache = False
    if cached_test_path.exists() and cached_summary_path.exists() and not rebuild_features:
        cached_result = json.loads(cached_summary_path.read_text())
        can_reuse_cache = (
            cached_result.get("run_name") == run_name
            and int(cached_result.get("n_folds", -1)) == int(n_folds)
            and [int(seed) for seed in cached_result.get("seeds", [])] == requested_seeds
            and cached_result.get("target_feature_views") == target_feature_views
        )
    if can_reuse_cache:
        result = cached_result
        test_preds = pd.read_csv(cached_test_path)
    else:
        result = train_public_lgb_targetwise(
            preset_name=preset_name,
            default_feature_view=default_feature_view,
            n_folds=n_folds,
            seeds=requested_seeds,
            rebuild_features=rebuild_features,
            persist=True,
            cv_scheme=cv_scheme,
            use_target_params=use_target_params,
        )
        test_preds = pd.read_csv(cached_test_path)

    submission = load_submission_template()[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = np.clip(test_preds[target].to_numpy(dtype=float), clip_min, clip_max)
    output_path = SUBMISSIONS_DIR / f"submission_{tag}_{run_name}.csv"
    submission.to_csv(output_path, index=False)

    report_lines = [
        f"# Public LGB Targetwise Submission Report ({preset_name})",
        "",
        f"- Source OOF mean log-loss: {result['scores']['mean']:.6f}",
        f"- Default feature view: `{default_feature_view}`",
        f"- Folds: {n_folds}",
        f"- Seeds: {result['seeds']}",
        f"- Clip range: [{clip_min:.2f}, {clip_max:.2f}]",
        "",
        "## Target Views",
        "",
    ]
    for target in TARGET_COLUMNS:
        report_lines.append(f"- `{target}`: `{target_feature_views[target]}`")
    report_lines.extend(["", "## Feature Counts By View", ""])
    for feature_view, feature_count in result["n_features_by_view"].items():
        report_lines.append(f"- `{feature_view}`: {feature_count}")
    report_lines.extend(["", "## Target Scores", ""])
    for target in TARGET_COLUMNS:
        report_lines.append(f"- `{target}`: {result['scores'][target]:.6f}")
    write_markdown(REPORT_SUBMISSIONS_DIR / f"submission_{tag}_{run_name}.md", "\n".join(report_lines))
    return output_path

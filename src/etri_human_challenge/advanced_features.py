"""Advanced temporal and behavioral feature engineering — Phase 2.

Adds three categories of new features on top of the public_lgb feature table:

  1. Second-order temporal dynamics
     - Acceleration (delta-of-delta), rolling skew/kurtosis, weekly autocorr,
       trend sign consistency.
  2. Cross-feature interactions
     - Cardiac efficiency, GPS×WiFi social score, night disruption composite,
       ambience-mobility, screen session density.
  3. Behavioral consistency & anomaly
     - Per-anchor habit score (1/CV), personal z-score vs expanding baseline,
       global anomaly score (mean abs personal z across key anchors).

Design contract:
  - All new columns have no trailing "__" unless they inherit parent names.
  - Shift(1) is applied before any rolling op — NO future leakage.
  - NaN is preferred over zero for missing values.
  - All operations are per-subject (groupby subject_id).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import HISTORY_ANCHORS, EPSILON, _safe_series_divide
from .paths import FEATURES_DIR, ensure_runtime_dirs
from .utils import write_json


# ─────────────────────────────────────────────
# Numpy-only skew / kurtosis (no scipy needed)
# ─────────────────────────────────────────────

def _np_skew(arr: np.ndarray) -> float:
    """Population skewness via numpy."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return np.nan
    mu = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 3))


def _np_kurt(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher) via numpy."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return np.nan
    mu = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 4) - 3.0)


def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=3).apply(_np_skew, raw=True)


def _rolling_kurt(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=4).apply(_np_kurt, raw=True)


# ─────────────────────────────────────────────
# Feature building helpers
# ─────────────────────────────────────────────

def _col(frame: pd.DataFrame, name: str, fill: float = 0.0) -> pd.Series:
    if name in frame.columns:
        return frame[name].fillna(fill).astype(float)
    return pd.Series(fill, index=frame.index, dtype=float)


# ─────────────────────────────────────────────
# 1. Second-order temporal dynamics
# ─────────────────────────────────────────────

def add_second_order_temporal(frame: pd.DataFrame) -> pd.DataFrame:
    """
    For each HISTORY_ANCHOR present in frame, add:
    - __accel_3_7: (current - roll_mean_3) - (current - roll_mean_7) ≡ roll_mean_7 - roll_mean_3
      Captures short-term acceleration relative to medium-term trend.
    - __roll_skew_14: skewness of last 14 values (distribution shape).
    - __roll_kurt_14: excess kurtosis of last 14 values (tail heaviness).
    - __weekly_autocorr: 14-day rolling corr between value and its 7-day lag
      (quantifies weekly habit regularity).
    - __trend_sign_7: rolling mean of sign(diff) over 7 days shifted by 1
      (+1 = sustained increase, -1 = sustained decrease, ~0 = oscillating).

    WHY this matters:
    - Acceleration captures abrupt behavioral shifts the slope alone misses.
    - Skew/kurtosis detect asymmetric deterioration patterns.
    - Weekly autocorr distinguishes habitual from chaotic subjects.
    """
    frame = frame.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    grouped = frame.groupby("subject_id", sort=False)
    additions: dict[str, pd.Series] = {}

    active_anchors = [col for col in HISTORY_ANCHORS if col in frame.columns]

    for col in active_anchors:
        # Acceleration: difference between short-term and medium-term deviation
        roll3_col = f"{col}__roll_mean_3"
        roll7_col = f"{col}__roll_mean_7"
        if roll3_col in frame.columns and roll7_col in frame.columns:
            additions[f"{col}__accel_3_7"] = frame[roll7_col] - frame[roll3_col]
        elif roll7_col in frame.columns:
            roll3 = grouped[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            additions[f"{col}__accel_3_7"] = frame[roll7_col] - roll3

        # Rolling skew/kurtosis (14-day)
        additions[f"{col}__roll_skew_14"] = grouped[col].transform(
            lambda s, fn=_rolling_skew: fn(s, 14)
        )
        additions[f"{col}__roll_kurt_14"] = grouped[col].transform(
            lambda s, fn=_rolling_kurt: fn(s, 14)
        )

        # Weekly autocorrelation (14-day rolling corr with 7-day lag)
        def _weekly_autocorr_transform(s: pd.Series) -> pd.Series:
            lag7 = s.shift(7)
            return s.shift(1).rolling(14, min_periods=7).corr(lag7.shift(1))

        additions[f"{col}__weekly_autocorr"] = grouped[col].transform(
            _weekly_autocorr_transform
        )

        # Trend sign consistency: rolling mean of sign(diff) over 7 days
        def _trend_sign(s: pd.Series) -> pd.Series:
            return np.sign(s.diff()).shift(1).rolling(7, min_periods=3).mean()

        additions[f"{col}__trend_sign_7"] = grouped[col].transform(_trend_sign)

    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)
    return frame.copy()


# ─────────────────────────────────────────────
# 2. Cross-feature interactions
# ─────────────────────────────────────────────

def add_cross_feature_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """
    New cross-modality interactions not present in the existing pipeline:

    x_hr_per_step_log       : HR / log(steps+1)  — cardiac efficiency
    x_gps_wifi_social       : log(GPS extent) × log(WiFi unique)  — social mobility
    x_ble_wifi_indoor       : log(BLE unique) × log(WiFi unique)  — indoor proximity
    x_night_disruption      : night screen × log(night steps+1)  — fragmented rest
    x_activity_entropy_hr   : activity entropy × resting HR  — physiological stress
    x_screen_session_density: sessions / (screen_on_min+1)  — phone pick-up rate
    x_speech_mobility       : speech share × log(GPS speed+1)  — commute/social

    WHY: Second-order interactions capture behavioral states that single-modal
    features cannot represent (e.g. a person walking at night while on phone).
    """
    additions: dict[str, pd.Series] = {}

    hr_mean      = _col(frame, "d_whr_mean", fill=np.nan)
    step_sum     = _col(frame, "d_wpedo_step_sum", fill=0.0)
    gps_extent   = _col(frame, "d_mgps_extent", fill=0.0).clip(lower=0.0)
    wifi_unique  = _col(frame, "d_mwifi_unique_bssid_cnt", fill=0.0)
    ble_unique   = _col(frame, "d_mble_unique_addr_cnt", fill=0.0)
    act_entropy  = _col(frame, "d_mact_entropy", fill=np.nan)
    hr_rest      = _col(frame, "d_whr_rest_q10", fill=np.nan)
    screen_on    = _col(frame, "d_mscreen_on_min", fill=0.0)
    sessions     = _col(frame, "d_mscreen_session_cnt", fill=0.0)
    speech       = _col(frame, "d_mamb_speech_top1_share", fill=0.0)
    gps_speed    = _col(frame, "d_mgps_speed_mean", fill=0.0)

    # Night time composites
    screen_night = (
        _col(frame, "tb_overnight_mscreen_on_share", 0.0)
        + _col(frame, "tb_evening_mscreen_on_share", 0.0)
    )
    steps_night = (
        _col(frame, "tb_overnight_wpedo_step_sum", 0.0)
        + _col(frame, "tb_evening_wpedo_step_sum", 0.0)
    )

    step_log = np.log1p(step_sum)
    additions["x_hr_per_step_log"]        = _safe_series_divide(hr_mean, step_log + 1.0)
    additions["x_gps_wifi_social"]        = np.log1p(gps_extent) * np.log1p(wifi_unique)
    additions["x_ble_wifi_indoor"]        = np.log1p(ble_unique) * np.log1p(wifi_unique)
    additions["x_night_disruption"]       = screen_night * np.log1p(steps_night)
    additions["x_activity_entropy_hr"]    = act_entropy * hr_rest
    additions["x_screen_session_density"] = _safe_series_divide(sessions, screen_on + 1.0)
    additions["x_speech_mobility"]        = speech * np.log1p(gps_speed)

    # Resting HR deviation from daytime HR (sleep quality proxy)
    hr_day = (
        _col(frame, "tb_morning_whr_mean", fill=np.nan)
        + _col(frame, "tb_afternoon_whr_mean", fill=np.nan)
    ) / 2.0
    additions["x_hr_rest_day_gap"] = hr_rest - hr_day

    frame = pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)
    return frame.copy()


# ─────────────────────────────────────────────
# 3. Behavioral consistency & anomaly
# ─────────────────────────────────────────────

_KEY_ANCHORS_FOR_CONSISTENCY = [
    "d_mscreen_on_min",
    "d_wpedo_step_sum",
    "d_whr_mean",
    "d_mac_charge_min",
    "d_mgps_extent",
]


def add_behavioral_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Per-subject behavioral regularity metrics:

    {col}__habit_score_7  : 1 / (1 + CV_7)  — higher = more regular over last 7 days
    {col}__personal_zscore: (value - expanding_mean) / expanding_std  — deviation
                            from this subject's own historical baseline.
    x_global_anomaly_score: mean(|personal_zscore|) across key anchors  — overall
                            behavioral novelty (anomaly detection proxy).

    WHY: On a 450-row dataset subjects differ hugely in baseline behavior. A
    personal z-score captures "is this unusual FOR THIS PERSON" — much more
    predictive than population-level z-scores for health labels.
    """
    frame = frame.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    grouped = frame.groupby("subject_id", sort=False)
    additions: dict[str, pd.Series] = {}

    personal_z_cols: list[str] = []

    active_anchors = [col for col in _KEY_ANCHORS_FOR_CONSISTENCY if col in frame.columns]

    for col in active_anchors:
        # Habit score: inverse of coefficient of variation over last 7 days
        roll_mean_7 = grouped[col].transform(
            lambda s: s.shift(1).rolling(7, min_periods=2).mean()
        )
        roll_std_7 = grouped[col].transform(
            lambda s: s.shift(1).rolling(7, min_periods=2).std()
        )
        cv_7 = _safe_series_divide(roll_std_7, roll_mean_7.abs() + EPSILON)
        habit_col = f"{col}__habit_score_7"
        additions[habit_col] = 1.0 / (1.0 + cv_7)

        # Personal z-score: deviation from expanding (leave-one-out) personal mean
        exp_mean = grouped[col].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean()
        )
        exp_std = grouped[col].transform(
            lambda s: s.shift(1).expanding(min_periods=3).std()
        )
        pz_col = f"{col}__personal_zscore"
        additions[pz_col] = _safe_series_divide(frame[col] - exp_mean, exp_std + EPSILON)
        personal_z_cols.append(pz_col)

    # Global anomaly score: mean absolute personal z-score across anchors
    if personal_z_cols:
        z_df = pd.DataFrame({c: additions[c] for c in personal_z_cols}, index=frame.index)
        additions["x_global_anomaly_score"] = z_df.abs().mean(axis=1)

    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)
    return frame.copy()


# ─────────────────────────────────────────────
# 4. Subject-level rank features
# ─────────────────────────────────────────────

def add_subject_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile rank of each value within its subject's own history.
    Rank [0,1] is more comparable across subjects than raw values.

    Only applied to key anchors to avoid dimensionality explosion.

    WHY: Subjects have vastly different behavioral scales. Rank is
    invariant to monotone transforms and captures relative position
    within personal history (e.g. "this is a high-activity day for
    this specific person").
    """
    frame = frame.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    grouped = frame.groupby("subject_id", sort=False)
    additions: dict[str, pd.Series] = {}

    active_anchors = [col for col in _KEY_ANCHORS_FOR_CONSISTENCY if col in frame.columns]

    for col in active_anchors:
        def _expanding_rank(s: pd.Series) -> pd.Series:
            """Rank of current value within expanding past window (no future leak)."""
            result = pd.Series(np.nan, index=s.index, dtype=float)
            for i in range(1, len(s)):
                past = s.iloc[:i].dropna()
                if len(past) < 2:
                    continue
                current = s.iloc[i]
                if pd.isna(current):
                    continue
                result.iloc[i] = float((past < current).mean())
            return result

        additions[f"{col}__expanding_rank"] = grouped[col].transform(_expanding_rank)

    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)
    return frame.copy()


# ─────────────────────────────────────────────
# 5. Main pipeline
# ─────────────────────────────────────────────

def build_advanced_feature_table(
    base_frame: pd.DataFrame,
    *,
    include_second_order: bool = True,
    include_cross_features: bool = True,
    include_consistency: bool = True,
    include_rank: bool = True,
    persist: bool = True,
) -> pd.DataFrame:
    """
    Extend the public_lgb feature table with all advanced features.

    Parameters
    ----------
    base_frame:
        Output of ``build_public_lgb_feature_table()``.
    include_second_order:
        Add acceleration, skew, kurtosis, weekly autocorr features.
    include_cross_features:
        Add cross-modality interaction features.
    include_consistency:
        Add habit score, personal z-score, global anomaly score.
    include_rank:
        Add expanding rank features (slightly expensive — skip if slow).
    persist:
        Save to artifacts/features/advanced_feature_table.parquet.
    """
    frame = base_frame.copy()

    if include_second_order:
        frame = add_second_order_temporal(frame)

    if include_cross_features:
        frame = add_cross_feature_interactions(frame)

    if include_consistency:
        frame = add_behavioral_consistency(frame)

    if include_rank:
        frame = add_subject_rank_features(frame)

    if persist:
        ensure_runtime_dirs()
        path = FEATURES_DIR / "advanced_feature_table.parquet"
        frame.to_parquet(path, index=False)

        base_cols = set(base_frame.columns)
        new_cols = [c for c in frame.columns if c not in base_cols]
        meta: dict[str, Any] = {
            "total_columns": int(frame.shape[1]),
            "new_columns": len(new_cols),
            "new_column_names": new_cols[:50],  # sample
            "rows": int(len(frame)),
        }
        write_json(FEATURES_DIR / "advanced_feature_table_meta.json", meta)

    return frame


def load_advanced_feature_table(rebuild: bool = False) -> pd.DataFrame:
    path = FEATURES_DIR / "advanced_feature_table.parquet"
    if rebuild or not path.exists():
        from .public_lgb import load_public_lgb_feature_table
        base = load_public_lgb_feature_table(rebuild=False)
        return build_advanced_feature_table(base, persist=True)
    return pd.read_parquet(path)

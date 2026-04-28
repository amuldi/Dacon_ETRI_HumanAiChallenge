"""Day-level handcrafted feature generation for the ETRI challenge."""

from __future__ import annotations

from . import bootstrap as _bootstrap  # noqa: F401

from functools import reduce
from typing import Any

import numpy as np
import pandas as pd

from .constants import TIME_BUCKETS, WINDOWS
from .io import build_key_frame, load_modality_frame
from .paths import FEATURES_DIR, RAW_MODALITY_DIR, REPORT_FEATURES_DIR, ensure_runtime_dirs
from .utils import hour_to_bucket, safe_float, write_markdown


BUCKET_NAMES = list(TIME_BUCKETS.keys())
ACTIVITY_CODES = [0, 1, 3, 4, 7, 8]
MODALITY_PREFIXES = ["mac", "mscreen", "mact", "mlight", "wlight", "wpedo", "whr", "mgps", "mwifi", "mble", "muse", "mamb"]
HISTORY_ANCHORS = [
    "d_mscreen_on_min",
    "d_mac_charge_min",
    "d_wpedo_step_sum",
    "d_wpedo_active_min",
    "d_whr_mean",
    "d_whr_rest_q10",
    "d_mgps_speed_mean",
    "d_mgps_extent",
    "d_mwifi_unique_bssid_cnt",
    "d_mble_unique_addr_cnt",
    "d_muse_unique_app_cnt",
    "d_mamb_speech_top1_share",
    "d_mlight_log_mean",
    "d_wlight_log_mean",
    "d_modal_observed_cnt",
    "d_modal_mean_coverage",
    "d_behavior_screen_night_day_gap",
    "d_behavior_charge_night_day_gap",
    "d_behavior_steps_night_day_ratio",
    "x_screen_charge_share",
    "x_steps_extent_log",
    "x_stationary_screen",
    "x_usage_screen_density",
]
ROLL_WINDOWS = [window for window in WINDOWS if window > 1]
EPSILON = 1e-3


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["lifelog_date"] = prepared["timestamp"].dt.normalize()
    prepared["hour_bucket"] = prepared["timestamp"].dt.hour.map(hour_to_bucket)
    return prepared.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)


def _safe_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array[~np.isnan(array)]


def _safe_mean(values: Any) -> float:
    array = _safe_array(values)
    return float(array.mean()) if len(array) else np.nan


def _safe_std(values: Any) -> float:
    array = _safe_array(values)
    return float(array.std()) if len(array) else np.nan


def _safe_min(values: Any) -> float:
    array = _safe_array(values)
    return float(array.min()) if len(array) else np.nan


def _safe_max(values: Any) -> float:
    array = _safe_array(values)
    return float(array.max()) if len(array) else np.nan


def _safe_sum(values: Any) -> float:
    array = _safe_array(values)
    return float(array.sum()) if len(array) else np.nan


def _safe_quantile(values: Any, q: float) -> float:
    array = _safe_array(values)
    return float(np.quantile(array, q)) if len(array) else np.nan


def _entropy_from_probs(values: Any) -> float:
    array = _safe_array(values)
    if len(array) == 0:
        return np.nan
    total = array.sum()
    if total <= 0:
        return 0.0
    probs = array / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _binary_run_stats(mask: Any) -> tuple[float, float]:
    array = np.asarray(mask, dtype=int)
    if len(array) == 0:
        return np.nan, np.nan
    starts = (array == 1) & np.concatenate([[True], array[:-1] == 0])
    episode_cnt = float(starts.sum())
    longest = 0
    current = 0
    for value in array:
        if value == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return episode_cnt, float(longest)


def _transition_count(values: Any) -> float:
    array = np.asarray(values)
    if len(array) <= 1:
        return 0.0
    return float(np.sum(array[1:] != array[:-1]))


def _share(mask: Any) -> float:
    array = np.asarray(mask, dtype=float)
    return float(array.mean()) if len(array) else np.nan


def _safe_log_mean(values: Any) -> float:
    array = _safe_array(values)
    return float(np.log1p(np.clip(array, a_min=0.0, a_max=None)).mean()) if len(array) else np.nan


def _safe_log_p90(values: Any) -> float:
    array = _safe_array(values)
    return float(np.quantile(np.log1p(np.clip(array, a_min=0.0, a_max=None)), 0.9)) if len(array) else np.nan


def _series_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(0.0).astype(float)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _safe_series_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    if np.isscalar(denominator):
        denominator_series = pd.Series(float(denominator), index=numerator.index, dtype=float)
    else:
        denominator_series = denominator.astype(float)
    denominator_series = denominator_series.replace(0.0, np.nan)
    return numerator.astype(float) / denominator_series


def _coefficient_of_variation(mean_value: pd.Series, std_value: pd.Series) -> pd.Series:
    denominator = mean_value.abs().replace(0.0, np.nan)
    return std_value / denominator


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    def _slope(window_values: np.ndarray) -> float:
        array = np.asarray(window_values, dtype=float)
        array = array[~np.isnan(array)]
        if len(array) < 2:
            return np.nan
        x = np.arange(len(array), dtype=float)
        x = x - x.mean()
        y = array - array.mean()
        denominator = np.sum(x * x)
        if denominator == 0:
            return np.nan
        return float(np.sum(x * y) / denominator)

    return values.shift(1).rolling(window, min_periods=2).apply(_slope, raw=True)


def _base_record(subject_id: str, lifelog_date: pd.Timestamp, prefix: str, group: pd.DataFrame) -> dict[str, float | str | pd.Timestamp]:
    observed_hours = float(group["timestamp"].dt.hour.nunique())
    bucket_present = float(group["hour_bucket"].nunique())
    return {
        "subject_id": subject_id,
        "lifelog_date": lifelog_date,
        f"d_{prefix}_row_cnt": float(len(group)),
        f"d_{prefix}_coverage_ratio": observed_hours / 24.0,
        f"d_{prefix}_bucket_present_cnt": bucket_present,
    }


def _add_bucket_feature(record: dict[str, Any], group: pd.DataFrame, values: Any, feature_base: str, reducer) -> None:
    value_array = np.asarray(values)
    for bucket in BUCKET_NAMES:
        mask = group["hour_bucket"].eq(bucket).to_numpy()
        bucket_values = value_array[mask]
        record[f"tb_{bucket}_{feature_base}"] = reducer(bucket_values)


def _dense_calendar(base: pd.DataFrame, key_frame: pd.DataFrame) -> pd.DataFrame:
    all_keys = pd.concat(
        [
            base[["subject_id", "lifelog_date"]],
            key_frame[["subject_id", "lifelog_date"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    bounds = all_keys.groupby("subject_id")["lifelog_date"].agg(["min", "max"]).reset_index()
    records: list[dict[str, Any]] = []
    for row in bounds.itertuples(index=False):
        for current_date in pd.date_range(row.min, row.max, freq="D"):
            records.append({"subject_id": row.subject_id, "lifelog_date": current_date.normalize()})
    return pd.DataFrame(records)


def _summarize_m_ac_status() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mACStatus.parquet"))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        charging = group["m_charging"].fillna(0).to_numpy(dtype=float)
        active = (charging > 0).astype(float)
        record = _base_record(subject_id, lifelog_date, "mac", group)
        record["d_mac_charge_min"] = float(active.sum())
        record["d_mac_charge_share"] = _share(active)
        record["d_mac_charge_episode_cnt"], record["d_mac_longest_charge_min"] = _binary_run_stats(active)
        _add_bucket_feature(record, group, active, "mac_charge_share", _share)
        records.append(record)
    return pd.DataFrame(records)


def _summarize_m_screen_status() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mScreenStatus.parquet"))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        screen = group["m_screen_use"].fillna(0).to_numpy(dtype=float)
        active = (screen > 0).astype(float)
        record = _base_record(subject_id, lifelog_date, "mscreen", group)
        record["d_mscreen_on_min"] = float(active.sum())
        record["d_mscreen_on_share"] = _share(active)
        record["d_mscreen_session_cnt"], record["d_mscreen_longest_session_min"] = _binary_run_stats(active)
        _add_bucket_feature(record, group, active, "mscreen_on_share", _share)
        records.append(record)
    return pd.DataFrame(records)


def _summarize_m_activity() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mActivity.parquet"))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        activity = group["m_activity"].fillna(-1).to_numpy(dtype=int)
        record = _base_record(subject_id, lifelog_date, "mact", group)
        shares = []
        for code in ACTIVITY_CODES:
            share = _share(activity == code)
            record[f"d_mact_code{code}_share"] = share
            if not np.isnan(share):
                shares.append(share)
        record["d_mact_entropy"] = _entropy_from_probs(np.asarray(shares, dtype=float))
        record["d_mact_transition_cnt"] = _transition_count(activity)
        records.append(record)
    return pd.DataFrame(records)


def _summarize_light(path_name: str, value_column: str, prefix: str, include_bucket: bool) -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / path_name))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        values = group[value_column].fillna(0).to_numpy(dtype=float)
        record = _base_record(subject_id, lifelog_date, prefix, group)
        record[f"d_{prefix}_log_mean"] = _safe_log_mean(values)
        record[f"d_{prefix}_log_p90"] = _safe_log_p90(values)
        record[f"d_{prefix}_std"] = _safe_std(values)
        record[f"d_{prefix}_zero_share"] = _share(values <= 0)
        if include_bucket:
            bucket_logs = np.log1p(np.clip(values, a_min=0.0, a_max=None))
            _add_bucket_feature(record, group, bucket_logs, f"{prefix}_log_mean", _safe_mean)
        records.append(record)
    return pd.DataFrame(records)


def _summarize_w_pedo() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_wPedo.parquet"))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        steps = group["step"].fillna(0).to_numpy(dtype=float)
        distance = group["distance"].fillna(0).to_numpy(dtype=float)
        speed = group["speed"].fillna(0).to_numpy(dtype=float)
        calories = group["burned_calories"].fillna(0).to_numpy(dtype=float)
        active = ((steps > 0) | (distance > 0) | (speed > 0)).astype(float)

        record = _base_record(subject_id, lifelog_date, "wpedo", group)
        record["d_wpedo_step_sum"] = float(steps.sum())
        record["d_wpedo_active_min"] = float(active.sum())
        record["d_wpedo_distance_sum"] = float(distance.sum())
        record["d_wpedo_speed_mean_active"] = _safe_mean(speed[active > 0])
        record["d_wpedo_speed_p90"] = _safe_quantile(speed, 0.9)
        record["d_wpedo_calories_sum"] = float(calories.sum())
        _add_bucket_feature(record, group, steps, "wpedo_step_sum", _safe_sum)
        records.append(record)
    return pd.DataFrame(records)


def _heart_rate_row_summary(value: Any) -> dict[str, float]:
    if value is None:
        return {"hr_mean": np.nan, "hr_min": np.nan, "hr_max": np.nan, "hr_std": np.nan, "hr_count": 0.0}
    array = np.asarray(value, dtype=float)
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return {"hr_mean": np.nan, "hr_min": np.nan, "hr_max": np.nan, "hr_std": np.nan, "hr_count": 0.0}
    return {
        "hr_mean": float(array.mean()),
        "hr_min": float(array.min()),
        "hr_max": float(array.max()),
        "hr_std": float(array.std()),
        "hr_count": float(len(array)),
    }


def _summarize_w_hr() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_wHr.parquet"))
    row_stats = pd.DataFrame([_heart_rate_row_summary(value) for value in frame["heart_rate"]])
    frame = pd.concat([frame.reset_index(drop=True), row_stats], axis=1)

    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        hr_mean = group["hr_mean"].to_numpy(dtype=float)
        hr_min = group["hr_min"].to_numpy(dtype=float)
        hr_max = group["hr_max"].to_numpy(dtype=float)
        hr_count = group["hr_count"].to_numpy(dtype=float)
        record = _base_record(subject_id, lifelog_date, "whr", group)
        record["d_whr_mean"] = _safe_mean(hr_mean)
        record["d_whr_std"] = _safe_std(hr_mean)
        record["d_whr_min"] = _safe_min(hr_min)
        record["d_whr_max"] = _safe_max(hr_max)
        record["d_whr_rest_q10"] = _safe_quantile(hr_mean, 0.1)
        record["d_whr_valid_sample_cnt"] = float(hr_count.sum())
        _add_bucket_feature(record, group, hr_mean, "whr_mean", _safe_mean)
        records.append(record)
    return pd.DataFrame(records)


def _gps_row_summary(value: Any) -> dict[str, float]:
    if value is None:
        return {
            "point_cnt": 0.0,
            "speed_mean": np.nan,
            "speed_max": np.nan,
            "lat_mean": np.nan,
            "lon_mean": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "lon_min": np.nan,
            "lon_max": np.nan,
        }
    points = list(value)
    if len(points) == 0:
        return {
            "point_cnt": 0.0,
            "speed_mean": np.nan,
            "speed_max": np.nan,
            "lat_mean": np.nan,
            "lon_mean": np.nan,
            "lat_min": np.nan,
            "lat_max": np.nan,
            "lon_min": np.nan,
            "lon_max": np.nan,
        }
    speed = np.asarray([safe_float(item.get("speed")) for item in points if safe_float(item.get("speed")) is not None], dtype=float)
    lat = np.asarray([safe_float(item.get("latitude")) for item in points if safe_float(item.get("latitude")) is not None], dtype=float)
    lon = np.asarray([safe_float(item.get("longitude")) for item in points if safe_float(item.get("longitude")) is not None], dtype=float)
    return {
        "point_cnt": float(len(points)),
        "speed_mean": _safe_mean(speed),
        "speed_max": _safe_max(speed),
        "lat_mean": _safe_mean(lat),
        "lon_mean": _safe_mean(lon),
        "lat_min": _safe_min(lat),
        "lat_max": _safe_max(lat),
        "lon_min": _safe_min(lon),
        "lon_max": _safe_max(lon),
    }


def _summarize_m_gps() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mGps.parquet"))
    row_stats = pd.DataFrame([_gps_row_summary(value) for value in frame["m_gps"]])
    frame = pd.concat([frame.reset_index(drop=True), row_stats], axis=1)

    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        speed_mean = group["speed_mean"].to_numpy(dtype=float)
        speed_max = group["speed_max"].to_numpy(dtype=float)
        lat_min = group["lat_min"].to_numpy(dtype=float)
        lat_max = group["lat_max"].to_numpy(dtype=float)
        lon_min = group["lon_min"].to_numpy(dtype=float)
        lon_max = group["lon_max"].to_numpy(dtype=float)

        extent = np.sqrt(np.square(np.nanmax(lat_max) - np.nanmin(lat_min)) + np.square(np.nanmax(lon_max) - np.nanmin(lon_min)))
        if not np.isfinite(extent):
            extent = np.nan

        record = _base_record(subject_id, lifelog_date, "mgps", group)
        record["d_mgps_speed_mean"] = _safe_mean(speed_mean)
        record["d_mgps_speed_p90"] = _safe_quantile(speed_mean, 0.9)
        record["d_mgps_speed_max"] = _safe_max(speed_max)
        record["d_mgps_stationary_share"] = _share(speed_mean < 0.5)
        record["d_mgps_point_cnt"] = float(np.nansum(group["point_cnt"].to_numpy(dtype=float)))
        record["d_mgps_extent"] = float(extent) if np.isfinite(extent) else np.nan
        _add_bucket_feature(record, group, speed_mean, "mgps_speed_mean", _safe_mean)
        records.append(record)
    return pd.DataFrame(records)


def _summarize_m_wifi() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mWifi.parquet"))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        ap_counts: list[float] = []
        row_rssi_mean: list[float] = []
        row_rssi_max: list[float] = []
        unique_bssid: set[str] = set()

        for value in group["m_wifi"]:
            items = list(value) if value is not None else []
            ap_counts.append(float(len(items)))
            rssis: list[float] = []
            for item in items:
                bssid = str(item.get("bssid", "")).strip()
                if bssid:
                    unique_bssid.add(bssid)
                rssi = safe_float(item.get("rssi"))
                if rssi is not None:
                    rssis.append(rssi)
            row_rssi_mean.append(_safe_mean(rssis))
            row_rssi_max.append(_safe_max(rssis))

        record = _base_record(subject_id, lifelog_date, "mwifi", group)
        record["d_mwifi_scan_cnt"] = float(len(group))
        record["d_mwifi_ap_cnt_mean"] = _safe_mean(ap_counts)
        record["d_mwifi_unique_bssid_cnt"] = float(len(unique_bssid))
        record["d_mwifi_rssi_mean"] = _safe_mean(row_rssi_mean)
        record["d_mwifi_max_rssi_mean"] = _safe_mean(row_rssi_max)
        records.append(record)
    return pd.DataFrame(records)


def _summarize_m_ble() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mBle.parquet"))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        device_counts: list[float] = []
        empty_scans: list[float] = []
        row_rssi_mean: list[float] = []
        row_rssi_max: list[float] = []
        unique_address: set[str] = set()
        unique_class: set[str] = set()

        for value in group["m_ble"]:
            items = list(value) if value is not None else []
            device_counts.append(float(len(items)))
            empty_scans.append(float(len(items) == 0))
            rssis: list[float] = []
            for item in items:
                address = str(item.get("address", "")).strip()
                if address:
                    unique_address.add(address)
                device_class = str(item.get("device_class", "")).strip()
                if device_class:
                    unique_class.add(device_class)
                rssi = safe_float(item.get("rssi"))
                if rssi is not None:
                    rssis.append(rssi)
            row_rssi_mean.append(_safe_mean(rssis))
            row_rssi_max.append(_safe_max(rssis))

        record = _base_record(subject_id, lifelog_date, "mble", group)
        record["d_mble_scan_cnt"] = float(len(group))
        record["d_mble_empty_scan_share"] = _safe_mean(empty_scans)
        record["d_mble_dev_cnt_mean"] = _safe_mean(device_counts)
        record["d_mble_unique_addr_cnt"] = float(len(unique_address))
        record["d_mble_class_cnt"] = float(len(unique_class))
        record["d_mble_rssi_mean"] = _safe_mean(row_rssi_mean)
        record["d_mble_max_rssi_mean"] = _safe_mean(row_rssi_max)
        records.append(record)
    return pd.DataFrame(records)


def _summarize_m_usage_stats() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mUsageStats.parquet"))
    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        app_counts: list[float] = []
        row_log_sum: list[float] = []
        row_log_max: list[float] = []
        dominant_apps: list[str] = []
        unique_apps: set[str] = set()

        for value in group["m_usage_stats"]:
            items = list(value) if value is not None else []
            app_counts.append(float(len(items)))
            totals: list[float] = []
            best_app = ""
            best_time = -1.0
            for item in items:
                app_name = str(item.get("app_name", "")).replace("\xa0", " ").strip()
                if app_name:
                    unique_apps.add(app_name)
                total_time = safe_float(item.get("total_time"))
                if total_time is not None:
                    totals.append(total_time)
                    if total_time > best_time:
                        best_time = total_time
                        best_app = app_name
            dominant_apps.append(best_app)
            row_log_sum.append(float(np.log1p(np.sum(totals))) if totals else np.nan)
            row_log_max.append(float(np.log1p(np.max(totals))) if totals else np.nan)

        dominant_share = np.nan
        if dominant_apps:
            app_counts_by_name = pd.Series(dominant_apps).value_counts()
            dominant_share = float(app_counts_by_name.iloc[0] / len(dominant_apps))

        record = _base_record(subject_id, lifelog_date, "muse", group)
        record["d_muse_snap_cnt"] = float(len(group))
        record["d_muse_app_cnt_mean"] = _safe_mean(app_counts)
        record["d_muse_unique_app_cnt"] = float(len(unique_apps))
        record["d_muse_top1_app_share"] = dominant_share
        record["d_muse_sum_log_total_time_mean"] = _safe_mean(row_log_sum)
        record["d_muse_sum_log_total_time_p90"] = _safe_quantile(row_log_sum, 0.9)
        record["d_muse_max_log_total_time_mean"] = _safe_mean(row_log_max)
        records.append(record)
    return pd.DataFrame(records)


def _ambience_row_summary(value: Any) -> dict[str, Any]:
    items = list(value) if value is not None else []
    if len(items) == 0:
        return {
            "top1_label": "",
            "top1_score": np.nan,
            "entropy": np.nan,
        }
    labels: list[str] = []
    scores: list[float] = []
    for item in items:
        item_list = list(item)
        if len(item_list) < 2:
            continue
        labels.append(str(item_list[0]))
        score = safe_float(item_list[1])
        if score is not None:
            scores.append(score)
    top1_label = labels[0] if labels else ""
    top1_score = scores[0] if scores else np.nan
    return {
        "top1_label": top1_label,
        "top1_score": top1_score,
        "entropy": _entropy_from_probs(scores),
    }


def _label_flag(label: str, keywords: list[str]) -> float:
    lowered = label.lower()
    return float(any(keyword in lowered for keyword in keywords))


def _summarize_m_ambience() -> pd.DataFrame:
    frame = _prepare_frame(load_modality_frame(RAW_MODALITY_DIR / "ch2025_mAmbience.parquet"))
    row_stats = pd.DataFrame([_ambience_row_summary(value) for value in frame["m_ambience"]])
    frame = pd.concat([frame.reset_index(drop=True), row_stats], axis=1)

    records: list[dict[str, Any]] = []
    for (subject_id, lifelog_date), group in frame.groupby(["subject_id", "lifelog_date"], sort=False):
        top_labels = group["top1_label"].fillna("").astype(str)
        record = _base_record(subject_id, lifelog_date, "mamb", group)
        record["d_mamb_top1_conf_mean"] = _safe_mean(group["top1_score"].to_numpy(dtype=float))
        record["d_mamb_entropy_mean"] = _safe_mean(group["entropy"].to_numpy(dtype=float))
        record["d_mamb_unique_top1_cnt"] = float(top_labels.nunique())
        record["d_mamb_speech_top1_share"] = _safe_mean(top_labels.map(lambda x: _label_flag(x, ["speech", "conversation"])))
        record["d_mamb_music_top1_share"] = _safe_mean(top_labels.map(lambda x: _label_flag(x, ["music"])))
        record["d_mamb_vehicle_top1_share"] = _safe_mean(top_labels.map(lambda x: _label_flag(x, ["vehicle", "engine", "car", "truck"])))
        record["d_mamb_silence_top1_share"] = _safe_mean(top_labels.map(lambda x: _label_flag(x, ["silence"])))
        record["d_mamb_indoor_top1_share"] = _safe_mean(top_labels.map(lambda x: _label_flag(x, ["inside"])))
        record["d_mamb_outdoor_top1_share"] = _safe_mean(top_labels.map(lambda x: _label_flag(x, ["outside"])))
        records.append(record)
    return pd.DataFrame(records)


def build_base_daily_table() -> pd.DataFrame:
    key_frame = build_key_frame()
    modality_tables = [
        _summarize_m_ac_status(),
        _summarize_m_screen_status(),
        _summarize_m_activity(),
        _summarize_light("ch2025_mLight.parquet", "m_light", "mlight", include_bucket=True),
        _summarize_light("ch2025_wLight.parquet", "w_light", "wlight", include_bucket=False),
        _summarize_w_pedo(),
        _summarize_w_hr(),
        _summarize_m_gps(),
        _summarize_m_wifi(),
        _summarize_m_ble(),
        _summarize_m_usage_stats(),
        _summarize_m_ambience(),
    ]
    merged = reduce(
        lambda left, right: left.merge(right, on=["subject_id", "lifelog_date"], how="outer"),
        modality_tables,
    )
    calendar = _dense_calendar(merged, key_frame)
    base = calendar.merge(merged, on=["subject_id", "lifelog_date"], how="left")
    base = base.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)

    lifelog_dt = pd.to_datetime(base["lifelog_date"])
    base = base.assign(
        d_dayofweek=lifelog_dt.dt.weekday.astype(float),
        d_is_weekend=lifelog_dt.dt.weekday.isin([5, 6]).astype(float),
        d_month=lifelog_dt.dt.month.astype(float),
    )

    derived: dict[str, pd.Series] = {}
    for prefix in MODALITY_PREFIXES:
        row_col = f"d_{prefix}_row_cnt"
        cov_col = f"d_{prefix}_coverage_ratio"
        bucket_col = f"d_{prefix}_bucket_present_cnt"
        derived[f"miss_{prefix}_any"] = base[row_col].isna().astype(float)
        base[row_col] = base[row_col].fillna(0.0)
        base[cov_col] = base[cov_col].fillna(0.0)
        base[bucket_col] = base[bucket_col].fillna(0.0)
        derived[f"d_{prefix}_bucket_util_share"] = base[bucket_col] / float(len(BUCKET_NAMES))
        derived[f"d_{prefix}_row_per_bucket"] = base[row_col] / (base[bucket_col] + 1.0)
        derived[f"d_{prefix}_row_per_hour"] = base[row_col] / ((base[cov_col] * 24.0) + 1.0)

    base = pd.concat([base, pd.DataFrame(derived)], axis=1)

    observed_cols = [1.0 - base[f"miss_{prefix}_any"] for prefix in MODALITY_PREFIXES]
    coverage_cols = [base[f"d_{prefix}_coverage_ratio"] for prefix in MODALITY_PREFIXES]
    row_cols = [base[f"d_{prefix}_row_cnt"] for prefix in MODALITY_PREFIXES]
    bucket_util_cols = [base[f"d_{prefix}_bucket_util_share"] for prefix in MODALITY_PREFIXES]
    derived = {
        "d_modal_observed_cnt": pd.concat(observed_cols, axis=1).sum(axis=1),
        "d_modal_total_row_cnt": pd.concat(row_cols, axis=1).sum(axis=1),
        "d_modal_mean_coverage": pd.concat(coverage_cols, axis=1).mean(axis=1),
        "d_modal_mean_bucket_util": pd.concat(bucket_util_cols, axis=1).mean(axis=1),
    }
    derived["d_modal_missing_cnt"] = float(len(MODALITY_PREFIXES)) - derived["d_modal_observed_cnt"]

    screen_night = _series_or_zero(base, "tb_overnight_mscreen_on_share") + _series_or_zero(base, "tb_evening_mscreen_on_share")
    screen_day = _series_or_zero(base, "tb_morning_mscreen_on_share") + _series_or_zero(base, "tb_afternoon_mscreen_on_share")
    charge_night = _series_or_zero(base, "tb_overnight_mac_charge_share") + _series_or_zero(base, "tb_evening_mac_charge_share")
    charge_day = _series_or_zero(base, "tb_morning_mac_charge_share") + _series_or_zero(base, "tb_afternoon_mac_charge_share")
    light_night = _series_or_zero(base, "tb_overnight_mlight_log_mean") + _series_or_zero(base, "tb_evening_mlight_log_mean")
    light_day = _series_or_zero(base, "tb_morning_mlight_log_mean") + _series_or_zero(base, "tb_afternoon_mlight_log_mean")
    steps_night = _series_or_zero(base, "tb_overnight_wpedo_step_sum") + _series_or_zero(base, "tb_evening_wpedo_step_sum")
    steps_day = _series_or_zero(base, "tb_morning_wpedo_step_sum") + _series_or_zero(base, "tb_afternoon_wpedo_step_sum")
    gps_night = _series_or_zero(base, "tb_overnight_mgps_speed_mean") + _series_or_zero(base, "tb_evening_mgps_speed_mean")
    gps_day = _series_or_zero(base, "tb_morning_mgps_speed_mean") + _series_or_zero(base, "tb_afternoon_mgps_speed_mean")

    screen_on = _series_or_zero(base, "d_mscreen_on_min")
    charge_min = _series_or_zero(base, "d_mac_charge_min")
    step_sum = _series_or_zero(base, "d_wpedo_step_sum")
    gps_extent = _series_or_zero(base, "d_mgps_extent").clip(lower=0.0)
    hr_mean = _series_or_zero(base, "d_whr_mean")
    stationary_share = _series_or_zero(base, "d_mgps_stationary_share")
    app_cnt = _series_or_zero(base, "d_muse_unique_app_cnt")
    usage_time = _series_or_zero(base, "d_muse_sum_log_total_time_mean")
    speech_share = _series_or_zero(base, "d_mamb_speech_top1_share")

    derived["d_behavior_screen_night_day_gap"] = screen_night - screen_day
    derived["d_behavior_screen_night_day_ratio"] = _safe_series_divide(screen_night + EPSILON, screen_day + EPSILON)
    derived["d_behavior_charge_night_day_gap"] = charge_night - charge_day
    derived["d_behavior_charge_night_day_ratio"] = _safe_series_divide(charge_night + EPSILON, charge_day + EPSILON)
    derived["d_behavior_light_night_day_gap"] = light_night - light_day
    derived["d_behavior_steps_night_day_ratio"] = _safe_series_divide(steps_night + 1.0, steps_day + 1.0)
    derived["d_behavior_gps_night_day_gap"] = gps_night - gps_day
    derived["d_behavior_gps_night_day_ratio"] = _safe_series_divide(gps_night + EPSILON, gps_day + EPSILON)

    step_log = np.log1p(step_sum)
    extent_log = np.log1p(gps_extent)
    derived["x_screen_charge_share"] = _series_or_zero(base, "d_mscreen_on_share") * _series_or_zero(base, "d_mac_charge_share")
    derived["x_screen_charge_gap"] = screen_on - charge_min
    derived["x_screen_per_step"] = _safe_series_divide(screen_on, step_log + 1.0)
    derived["x_charge_per_screen"] = _safe_series_divide(charge_min, screen_on + 1.0)
    derived["x_steps_extent_log"] = step_log * extent_log
    derived["x_steps_hr"] = step_log * hr_mean
    derived["x_stationary_screen"] = stationary_share * _series_or_zero(base, "d_mscreen_on_share")
    derived["x_usage_screen_density"] = _safe_series_divide(app_cnt, screen_on + 1.0)
    derived["x_usage_speech"] = usage_time * speech_share
    derived["x_night_screen_charge"] = screen_night * charge_night
    derived["x_night_screen_lowlight"] = _safe_series_divide(screen_night, light_night + 1.0)
    base = pd.concat([base, pd.DataFrame(derived)], axis=1)
    return base.copy()


def _days_since_observed(series: pd.Series) -> pd.Series:
    values = series.fillna(0).to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    last_seen: int | None = None
    for index, value in enumerate(values):
        if value > 0:
            output[index] = 0.0
            last_seen = index
        elif last_seen is not None:
            output[index] = float(index - last_seen)
    return pd.Series(output, index=series.index)


def add_temporal_context(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)
    grouped = frame.groupby("subject_id", sort=False)
    additions: dict[str, pd.Series] = {}

    for column in [anchor for anchor in HISTORY_ANCHORS if anchor in frame.columns]:
        shifted = grouped[column].shift(1)
        additions[f"{column}__lag1"] = shifted
        additions[f"{column}__expanding_mean"] = grouped[column].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        for window in ROLL_WINDOWS:
            additions[f"{column}__roll_mean_{window}"] = grouped[column].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
        additions[f"{column}__roll_std_7"] = grouped[column].transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
        additions[f"{column}__roll_std_14"] = grouped[column].transform(lambda s: s.shift(1).rolling(14, min_periods=2).std())
        additions[f"{column}__delta_3"] = frame[column] - additions[f"{column}__roll_mean_3"]
        additions[f"{column}__delta_7"] = frame[column] - additions[f"{column}__roll_mean_7"]
        additions[f"{column}__delta_14"] = frame[column] - additions[f"{column}__roll_mean_14"]
        additions[f"{column}__delta_expanding"] = frame[column] - additions[f"{column}__expanding_mean"]
        additions[f"{column}__zscore_7"] = additions[f"{column}__delta_7"] / (additions[f"{column}__roll_std_7"] + EPSILON)
        additions[f"{column}__zscore_14"] = additions[f"{column}__delta_14"] / (additions[f"{column}__roll_std_14"] + EPSILON)
        additions[f"{column}__ratio_3"] = _safe_series_divide(frame[column] + EPSILON, additions[f"{column}__roll_mean_3"] + EPSILON)
        additions[f"{column}__ratio_14"] = _safe_series_divide(frame[column] + EPSILON, additions[f"{column}__roll_mean_14"] + EPSILON)
        additions[f"{column}__recent_vs_history_gap"] = additions[f"{column}__roll_mean_3"] - additions[f"{column}__roll_mean_14"]
        additions[f"{column}__recent_vs_history_ratio"] = _safe_series_divide(
            additions[f"{column}__roll_mean_3"] + EPSILON,
            additions[f"{column}__roll_mean_14"] + EPSILON,
        )
        additions[f"{column}__slope_7"] = grouped[column].transform(lambda s: _rolling_slope(s, 7))
        additions[f"{column}__slope_14"] = grouped[column].transform(lambda s: _rolling_slope(s, 14))
        additions[f"{column}__cv_14"] = _coefficient_of_variation(additions[f"{column}__roll_mean_14"], additions[f"{column}__roll_std_14"])

    for prefix in MODALITY_PREFIXES:
        row_col = f"d_{prefix}_row_cnt"
        if row_col in frame.columns:
            additions[f"miss_days_since_last_{prefix}"] = grouped[row_col].transform(_days_since_observed)
            additions[f"{row_col}__obs_share_7"] = grouped[row_col].transform(
                lambda s: s.shift(1).gt(0).astype(float).rolling(7, min_periods=1).mean()
            )
            additions[f"{row_col}__obs_share_14"] = grouped[row_col].transform(
                lambda s: s.shift(1).gt(0).astype(float).rolling(14, min_periods=1).mean()
            )
            additions[f"{row_col}__roll_mean_7"] = grouped[row_col].transform(
                lambda s: s.shift(1).rolling(7, min_periods=1).mean()
            )
            additions[f"{row_col}__roll_mean_14"] = grouped[row_col].transform(
                lambda s: s.shift(1).rolling(14, min_periods=1).mean()
            )
            additions[f"{row_col}__delta_7"] = frame[row_col] - additions[f"{row_col}__roll_mean_7"]
        cov_col = f"d_{prefix}_coverage_ratio"
        if cov_col in frame.columns:
            additions[f"{cov_col}__roll_mean_7"] = grouped[cov_col].transform(
                lambda s: s.shift(1).rolling(7, min_periods=1).mean()
            )
            additions[f"{cov_col}__roll_mean_14"] = grouped[cov_col].transform(
                lambda s: s.shift(1).rolling(14, min_periods=1).mean()
            )
            additions[f"{cov_col}__delta_7"] = frame[cov_col] - additions[f"{cov_col}__roll_mean_7"]
        bucket_col = f"d_{prefix}_bucket_present_cnt"
        if bucket_col in frame.columns:
            additions[f"{bucket_col}__roll_mean_7"] = grouped[bucket_col].transform(
                lambda s: s.shift(1).rolling(7, min_periods=1).mean()
            )

    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], axis=1)
    return frame.copy()


def build_daily_feature_table() -> pd.DataFrame:
    key_frame = build_key_frame()
    base = build_base_daily_table()
    enriched = add_temporal_context(base)
    feature_table = key_frame.merge(enriched, on=["subject_id", "lifelog_date"], how="left")
    feature_table = feature_table.sort_values(["subject_id", "sleep_date"]).reset_index(drop=True)
    return feature_table


def render_feature_report(frame: pd.DataFrame) -> str:
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    lines = [
        "# Feature Report",
        "",
        f"- Rows: {len(frame)}",
        f"- Columns: {frame.shape[1]}",
        f"- Numeric columns: {len(numeric_cols)}",
        f"- Train rows: {(frame['split'] == 'train').sum()}",
        f"- Test rows: {(frame['split'] == 'test').sum()}",
        "",
        "## Core Interface",
        "",
        "- Keys: `subject_id`, `sleep_date`, `lifelog_date`",
        "- Targets are present for train rows and null for test rows",
        "- All derived features are numeric or missingness flags",
        "",
        "## Example Columns",
        "",
    ]
    for column in numeric_cols[:60]:
        lines.append(f"- `{column}`")
    return "\n".join(lines)


def run_feature_build() -> pd.DataFrame:
    ensure_runtime_dirs()
    feature_table = build_daily_feature_table()
    feature_table.to_parquet(FEATURES_DIR / "daily_feature_table.parquet", index=False)
    write_markdown(REPORT_FEATURES_DIR / "daily_feature_table.md", render_feature_report(feature_table))
    return feature_table

"""Project-wide constants."""

from __future__ import annotations

TARGET_COLUMNS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"]
META_COLUMNS = ["split"]

TIME_BUCKETS = {
    "overnight": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}

MODALITY_FILES = {
    "m_ac_status": "ch2025_mACStatus.parquet",
    "m_activity": "ch2025_mActivity.parquet",
    "m_ambience": "ch2025_mAmbience.parquet",
    "m_ble": "ch2025_mBle.parquet",
    "m_gps": "ch2025_mGps.parquet",
    "m_light": "ch2025_mLight.parquet",
    "m_screen_status": "ch2025_mScreenStatus.parquet",
    "m_usage_stats": "ch2025_mUsageStats.parquet",
    "m_wifi": "ch2025_mWifi.parquet",
    "w_hr": "ch2025_wHr.parquet",
    "w_light": "ch2025_wLight.parquet",
    "w_pedo": "ch2025_wPedo.parquet",
}

SCALAR_MODALITY_COLUMNS = {
    "m_ac_status": ["m_charging"],
    "m_activity": ["m_activity"],
    "m_light": ["m_light"],
    "m_screen_status": ["m_screen_use"],
    "w_light": ["w_light"],
    "w_pedo": [
        "step",
        "step_frequency",
        "running_step",
        "walking_step",
        "distance",
        "speed",
        "burned_calories",
    ],
}

WINDOWS = [1, 3, 7, 14]
RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
EPSILON = 1e-6


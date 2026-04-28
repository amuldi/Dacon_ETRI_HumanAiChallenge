"""타겟별 LightGBM 하이퍼파라미터 설정.

설계 원칙:
  - 어려운 타겟(S4, Q1): num_leaves 줄이고, reg_lambda 올리고,
    feature_fraction 줄여 앙상블 다양성 확보
  - 쉬운 타겟(S1): num_leaves 올리고, reg 줄여 충분한 용량 허용
  - n_estimators: 항상 넉넉히 설정, early_stopping이 최적점 탐색
"""
from .public_lgb import PUBLIC_LGB_PARAMS

TARGET_LGB_PARAMS: dict[str, dict] = {
    # Q1: 균형 데이터 (0.496), 어려움 — 피처 다양성으로 과적합 방지
    "Q1": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":        31,
        "learning_rate":     0.015,
        "feature_fraction":  0.5,
        "bagging_fraction":  0.7,
        "reg_alpha":         0.5,
        "reg_lambda":        3.0,
        "min_child_samples": 25,
        "n_estimators":      3000,
    },

    # Q2: hist411 사용, 현재 파라미터가 거의 최적
    "Q2": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":       31,
        "learning_rate":    0.02,
        "feature_fraction": 0.6,
        "reg_alpha":        0.3,
        "reg_lambda":       2.0,
        "n_estimators":     2500,
    },

    # Q3: hist365 사용, Q2와 동일 설정
    "Q3": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":       31,
        "learning_rate":    0.02,
        "feature_fraction": 0.6,
        "reg_alpha":        0.3,
        "reg_lambda":       2.0,
        "n_estimators":     2500,
    },

    # S1: 가장 쉬움 (0.524) — 용량 확장
    "S1": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":       63,
        "learning_rate":    0.025,
        "feature_fraction": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_estimators":     2000,
    },

    # S2, S3: 표준 파라미터, 트리 수만 증가
    "S2": {**PUBLIC_LGB_PARAMS, "n_estimators": 2000},
    "S3": {**PUBLIC_LGB_PARAMS, "n_estimators": 2000},

    # S4: 가장 어려움 (0.611), subject 분산 큼 — 최대 정규화
    "S4": {
        **PUBLIC_LGB_PARAMS,
        "num_leaves":        15,
        "learning_rate":     0.01,
        "feature_fraction":  0.5,
        "bagging_fraction":  0.6,
        "reg_alpha":         1.0,
        "reg_lambda":        5.0,
        "min_child_samples": 30,
        "n_estimators":      4000,
    },
}

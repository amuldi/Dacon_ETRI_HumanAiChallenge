from __future__ import annotations

import json

from etri_human_challenge import bootstrap as _bootstrap  # noqa: F401
from etri_human_challenge.baseline import HAS_CATBOOST, evaluate_baseline, evaluate_subject_prior


def _score_summary(scores: dict[str, float], dummy_scores: dict[str, float]) -> dict[str, float]:
    return {
        "mean_log_loss": float(scores["mean"]),
        "std_log_loss": float(scores["std"]),
        "improvement_over_dummy": float(dummy_scores["mean"] - scores["mean"]),
    }


def main() -> None:
    subject_prior = evaluate_subject_prior(split_scheme="group_time")
    hgb = evaluate_baseline(split_scheme="group_time", model_family="hgb_prior", persist=False)
    hgb_select_resid = evaluate_baseline(split_scheme="group_time", model_family="hgb_select_resid", persist=False)

    results = {
        "dummy": {
            "mean_log_loss": float(hgb["dummy_scores"]["mean"]),
            "std_log_loss": float(hgb["dummy_scores"]["std"]),
            "improvement_over_dummy": 0.0,
        },
        "subject_prior": _score_summary(subject_prior["scores"], subject_prior["dummy_scores"]),
        "hgb_raw": _score_summary(hgb["raw_scores"], hgb["dummy_scores"]),
        "hgb_calibrated": _score_summary(hgb["calibrated_scores"], hgb["dummy_scores"]),
        "hgb_prior_blend": _score_summary(hgb["blended_scores"], hgb["dummy_scores"]),
        "hgb_select_resid_raw": _score_summary(hgb_select_resid["raw_scores"], hgb_select_resid["dummy_scores"]),
        "hgb_select_resid_calibrated": _score_summary(hgb_select_resid["calibrated_scores"], hgb_select_resid["dummy_scores"]),
        "hgb_select_resid_blend": _score_summary(hgb_select_resid["blended_scores"], hgb_select_resid["dummy_scores"]),
    }

    if HAS_CATBOOST:
        catboost = evaluate_baseline(split_scheme="group_time", model_family="catboost", persist=False)
        results["catboost_raw"] = _score_summary(catboost["raw_scores"], catboost["dummy_scores"])
        results["catboost_calibrated"] = _score_summary(catboost["calibrated_scores"], catboost["dummy_scores"])
        results["catboost_prior_blend"] = _score_summary(catboost["blended_scores"], catboost["dummy_scores"])

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

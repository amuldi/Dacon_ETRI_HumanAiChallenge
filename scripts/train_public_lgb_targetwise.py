from __future__ import annotations

import argparse
import json

from etri_human_challenge.public_lgb import (
    PUBLIC_LGB_FEATURE_VIEWS,
    PUBLIC_LGB_TARGETWISE_PRESETS,
    train_public_lgb_targetwise,
)


def _parse_seeds(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True, choices=sorted(PUBLIC_LGB_TARGETWISE_PRESETS))
    parser.add_argument("--default-feature-view", default="public_core", choices=sorted(PUBLIC_LGB_FEATURE_VIEWS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    result = train_public_lgb_targetwise(
        preset_name=args.preset,
        default_feature_view=args.default_feature_view,
        n_folds=args.n_folds,
        seeds=_parse_seeds(args.seeds),
        rebuild_features=args.rebuild_features,
        persist=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "preset": args.preset,
                "default_feature_view": args.default_feature_view,
                "mean_log_loss": result["scores"]["mean"],
                "n_features_by_view": result["n_features_by_view"],
                "artifacts": result["artifacts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

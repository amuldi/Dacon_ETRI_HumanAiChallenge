from __future__ import annotations

import argparse
import json

from etri_human_challenge.public_lgb import (
    PUBLIC_LGB_FEATURE_VIEWS,
    make_public_lgb_submission,
)


def _parse_seeds(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="public_lgb_v1")
    parser.add_argument("--feature-view", default="public_core", choices=sorted(PUBLIC_LGB_FEATURE_VIEWS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--clip-min", type=float, default=0.02)
    parser.add_argument("--clip-max", type=float, default=0.98)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    output_path = make_public_lgb_submission(
        tag=args.tag,
        feature_view=args.feature_view,
        n_folds=args.n_folds,
        seeds=_parse_seeds(args.seeds),
        rebuild_features=args.rebuild_features,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "feature_view": args.feature_view,
                "submission_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

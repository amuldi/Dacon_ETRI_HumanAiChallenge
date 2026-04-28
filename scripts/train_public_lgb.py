from __future__ import annotations

import argparse
import json

from etri_human_challenge.public_lgb import (
    PUBLIC_LGB_FEATURE_VIEWS,
    train_public_lgb,
)


def _parse_seeds(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-view", default="public_core", choices=sorted(PUBLIC_LGB_FEATURE_VIEWS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    result = train_public_lgb(
        feature_view=args.feature_view,
        n_folds=args.n_folds,
        seeds=_parse_seeds(args.seeds),
        rebuild_features=args.rebuild_features,
        persist=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "feature_view": args.feature_view,
                "mean_log_loss": result["scores"]["mean"],
                "n_features": result["n_features"],
                "artifacts": result["artifacts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

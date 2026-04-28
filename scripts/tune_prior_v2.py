from __future__ import annotations

import argparse
import json

from etri_human_challenge.prior_v2 import tune_prior_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-scheme", default="group_time", choices=["group", "group_time"])
    args = parser.parse_args()

    result = tune_prior_v2(split_scheme=args.split_scheme, persist=True)
    print(
        json.dumps(
            {
                "status": "ok",
                "split_scheme": args.split_scheme,
                "mean_log_loss": result["scores"]["mean"],
                "best_configs": result["best_configs"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from etri_human_challenge.prior_v2 import make_prior_v2_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-scheme", default="group_time", choices=["group", "group_time"])
    parser.add_argument("--tag", default="prior_v2")
    args = parser.parse_args()

    output_path = make_prior_v2_submission(split_scheme=args.split_scheme, tag=args.tag)
    print(
        json.dumps(
            {
                "status": "ok",
                "split_scheme": args.split_scheme,
                "submission_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

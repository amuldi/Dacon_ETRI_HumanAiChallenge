from __future__ import annotations

import argparse
import json

from etri_human_challenge.baseline import DEFAULT_MODEL_FAMILY, SUPPORTED_MODEL_FAMILIES, make_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-scheme", default="group_time", choices=["group", "group_time"])
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY, choices=sorted(SUPPORTED_MODEL_FAMILIES))
    args = parser.parse_args()

    output_path = make_submission(
        split_scheme=args.split_scheme,
        tag=args.tag,
        model_family=args.model_family,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "split_scheme": args.split_scheme,
                "model_family": args.model_family,
                "submission_path": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

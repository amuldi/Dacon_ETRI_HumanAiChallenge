from __future__ import annotations

import argparse
import json
from pathlib import Path

from etri_human_challenge.submission_mix import mix_submission_files


def _parse_target_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in values:
        target, raw_weight = item.split("=", 1)
        weights[target.strip()] = float(raw_weight)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target-weight", action="append", default=[])
    args = parser.parse_args()

    output_path = mix_submission_files(
        args.left,
        args.right,
        tag=args.tag,
        target_weights=_parse_target_weights(args.target_weight),
    )
    print(json.dumps({"status": "ok", "submission_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

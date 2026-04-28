#!/usr/bin/env python3
"""Generate requested public LGB softblend submissions and logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / ".vendor"))

from etri_human_challenge.softblend_experiments import run_configured_experiments


DEFAULT_EXPERIMENTS = [
    "histmix_guarded_v1_reproduce",
    "softblend_w090",
    "softblend_w085",
    "softblend_w095",
]
DEFAULT_WRITE_SUBMISSIONS = {
    "softblend_w090",
    "softblend_w085",
    "softblend_w095",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--write-submissions", default=",".join(sorted(DEFAULT_WRITE_SUBMISSIONS)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiment_names = [item.strip() for item in args.experiments.split(",") if item.strip()]
    write_submission_names = {item.strip() for item in args.write_submissions.split(",") if item.strip()}

    results = run_configured_experiments(
        experiment_names=experiment_names,
        write_submission_names=write_submission_names,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "experiments": [
                    {
                        "name": result.config.name,
                        "validation_scheme": result.validation_scheme,
                        "oof_mean": result.scores["mean"],
                        "submission_file": str(result.submission_path),
                    }
                    for result in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

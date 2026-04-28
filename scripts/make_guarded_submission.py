from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from etri_human_challenge.constants import KEY_COLUMNS, TARGET_COLUMNS
from etri_human_challenge.io import load_submission_template
from etri_human_challenge.paths import REPORT_SUBMISSIONS_DIR, SUBMISSIONS_DIR, ensure_runtime_dirs
from etri_human_challenge.utils import write_markdown


DEFAULT_V1_PATH = SUBMISSIONS_DIR / "submission_hgb_prior_v1_hgb_prior_group_time.csv"
DEFAULT_V2_PATH = SUBMISSIONS_DIR / "submission_prior_v2_prior_v2_group_time.csv"


def _parse_targets(value: str) -> list[str]:
    targets = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(targets) - set(TARGET_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    return targets


def make_guarded_submission(
    *,
    tag: str = "v3_s2_s4_guarded",
    v1_path: Path = DEFAULT_V1_PATH,
    v2_path: Path = DEFAULT_V2_PATH,
    v2_targets: list[str] | None = None,
    blend_weight: float | None = None,
) -> Path:
    ensure_runtime_dirs()
    v2_targets = v2_targets or ["S2", "S4"]
    if blend_weight is not None and not 0.0 <= blend_weight <= 2.0:
        raise ValueError("--blend-weight must be between 0 and 2.")

    template = load_submission_template()
    v1 = pd.read_csv(v1_path)
    v2 = pd.read_csv(v2_path)
    expected_columns = KEY_COLUMNS + TARGET_COLUMNS
    if list(v1.columns) != expected_columns:
        raise ValueError(f"v1 submission columns do not match sample: {v1_path}")
    if list(v2.columns) != expected_columns:
        raise ValueError(f"v2 submission columns do not match sample: {v2_path}")
    if len(v1) != len(template) or len(v2) != len(template):
        raise ValueError("Submission row counts do not match the sample submission.")

    guarded = v1.copy()
    if blend_weight is None:
        for target in v2_targets:
            guarded[target] = v2[target]
        strategy = f"target_swap:{','.join(v2_targets)}"
    else:
        for target in v2_targets:
            guarded[target] = (1.0 - blend_weight) * v1[target] + blend_weight * v2[target]
        blend_kind = "target_blend" if blend_weight <= 1.0 else "target_extrapolate"
        strategy = f"{blend_kind}:{','.join(v2_targets)}:v2_weight={blend_weight:g}"

    output_path = SUBMISSIONS_DIR / f"submission_{tag}_public_guarded_group_time.csv"
    guarded.to_csv(output_path, index=False)

    deltas = {}
    for target in TARGET_COLUMNS:
        diff = (guarded[target] - v1[target]).abs()
        deltas[target] = {
            "mean_abs_delta_vs_v1": float(diff.mean()),
            "max_abs_delta_vs_v1": float(diff.max()),
        }

    report_lines = [
        "# Public Guarded Submission Report",
        "",
        f"- Strategy: `{strategy}`",
        f"- Output: `{output_path}`",
        f"- Source v1: `{v1_path}`",
        f"- Source v2: `{v2_path}`",
        "- Public anchor: v1=0.6195173919, v2=0.6203498892",
        "- Rationale: keep the public-best v1 Q-target priors; only carry lower-drift S-target changes from v2.",
        "",
        "## Delta vs v1",
        "",
    ]
    for target in TARGET_COLUMNS:
        payload = deltas[target]
        report_lines.append(
            f"- `{target}`: mean_abs_delta={payload['mean_abs_delta_vs_v1']:.6f}, "
            f"max_abs_delta={payload['max_abs_delta_vs_v1']:.6f}"
        )
    write_markdown(REPORT_SUBMISSIONS_DIR / f"submission_{tag}_public_guarded_group_time.md", "\n".join(report_lines))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v3_s2_s4_guarded")
    parser.add_argument("--v1-path", type=Path, default=DEFAULT_V1_PATH)
    parser.add_argument("--v2-path", type=Path, default=DEFAULT_V2_PATH)
    parser.add_argument("--v2-targets", default="S2,S4")
    parser.add_argument("--blend-weight", type=float, default=None)
    args = parser.parse_args()

    output_path = make_guarded_submission(
        tag=args.tag,
        v1_path=args.v1_path,
        v2_path=args.v2_path,
        v2_targets=_parse_targets(args.v2_targets),
        blend_weight=args.blend_weight,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "submission_path": str(output_path),
                "v2_targets": _parse_targets(args.v2_targets),
                "blend_weight": args.blend_weight,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

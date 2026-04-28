from __future__ import annotations

import json

from etri_human_challenge.folds import run_fold_build


def main() -> None:
    manifest = run_fold_build()
    print(json.dumps({"status": "ok", "rows": int(len(manifest)), "schemes": sorted(manifest["split_scheme"].unique().tolist())}, ensure_ascii=False))


if __name__ == "__main__":
    main()


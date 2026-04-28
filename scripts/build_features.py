from __future__ import annotations

import json

from etri_human_challenge.features import run_feature_build


def main() -> None:
    frame = run_feature_build()
    print(json.dumps({"status": "ok", "rows": int(len(frame)), "columns": int(frame.shape[1])}, ensure_ascii=False))


if __name__ == "__main__":
    main()


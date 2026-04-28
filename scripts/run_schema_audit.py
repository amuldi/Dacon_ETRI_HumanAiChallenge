from __future__ import annotations

import json

from etri_human_challenge.schema_audit import run_schema_audit


def main() -> None:
    contract = run_schema_audit()
    print(json.dumps({"status": "ok", "modalities": len(contract["modalities"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()


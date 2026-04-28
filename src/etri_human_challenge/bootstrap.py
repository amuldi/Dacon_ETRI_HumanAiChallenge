"""Local dependency bootstrap.

This project keeps optional third-party wheels inside `.vendor` so the
scripts can run without mutating the system interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = ROOT / ".vendor"

if VENDOR_DIR.exists():
    vendor_str = str(VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


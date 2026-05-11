"""One-time: push existing docs/predictions.json into the Supabase predictions table.

Idempotent — uses upsert on `id` so re-running is safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sentinel.storage import db

PATH = Path("docs/predictions.json")


def main() -> int:
    """Read existing predictions.json and upsert all rows into Supabase."""
    if not PATH.exists():
        print(f"no file at {PATH}")
        return 0
    preds = json.loads(PATH.read_text(encoding="utf-8"))
    if not preds:
        print("predictions.json is empty")
        return 0
    print(f"migrating {len(preds)} predictions...")
    db.upsert_predictions(preds)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

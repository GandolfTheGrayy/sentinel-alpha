"""Move a generated stub from sentinel/{pillar}/_generated/ into the pillar root.

Usage:
    python scripts/promote.py scout sec_scraper.py
    python scripts/promote.py linguist drift_detector.py

After promotion, you must manually wire the file into sentinel/pipeline.py
imports and run pytest sentinel/tests/test_spine.py.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PILLARS = {"scout", "linguist", "historian", "judge", "tests"}


def main(pillar: str, filename: str) -> int:
    """Move file from _generated/ to spine; print follow-up instructions."""
    if pillar not in PILLARS:
        print(f"unknown pillar: {pillar}. one of: {sorted(PILLARS)}", file=sys.stderr)
        return 2
    src = Path(f"sentinel/{pillar}/_generated/{filename}")
    dst = Path(f"sentinel/{pillar}/{filename}")
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 2
    if dst.exists():
        print(f"refusing to overwrite: {dst}", file=sys.stderr)
        return 2
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    subprocess.run(["git", "add", str(src), str(dst)], check=False)
    print(f"promoted: {src} -> {dst}")
    print()
    print("next steps:")
    print(f"  1. read {dst} and clean it up")
    print(f"  2. import it from sentinel/pipeline.py if it should run daily")
    print(f"  3. add a smoke test in sentinel/tests/test_spine.py")
    print(f"  4. pytest sentinel/tests/test_spine.py")
    print(f"  5. git commit -m 'feat({pillar}): promote {filename} to spine'")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

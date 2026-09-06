"""Run the full CineMatch data pipeline end to end, in order.

Equivalent to running each numbered script yourself; use this for
convenience, or run steps individually while iterating.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_download_movielens.py",
    "02_enrich_tmdb.py",
    "03_clean_and_index.py",
    "04_temporal_split.py",
    "05_eda.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 60}\n{step}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / step)])
        if result.returncode != 0:
            sys.exit(f"\n{step} failed (exit code {result.returncode}); stopping.")


if __name__ == "__main__":
    main()

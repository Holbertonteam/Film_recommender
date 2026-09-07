"""Run the Azerbaijani-films pipeline end to end, in order."""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
STEPS = [
    "az_01_fetch_movies.py",
    "az_02_generate_synthetic.py",
    "az_03_temporal_split.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 60}\n{step}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / step)])
        if result.returncode != 0:
            sys.exit(f"\n{step} failed (exit code {result.returncode}); stopping.")


if __name__ == "__main__":
    main()

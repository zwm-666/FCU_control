"""Compatibility wrapper for scripts.run_split_experiments."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_split_experiments import *  # noqa: F401,F403
from scripts.run_split_experiments import main


if __name__ == "__main__":
    raise SystemExit(main())

"""Compatibility wrapper for scripts.fine_tune_existing_model."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.fine_tune_existing_model import *  # noqa: F401,F403
from scripts.fine_tune_existing_model import main


if __name__ == "__main__":
    main()

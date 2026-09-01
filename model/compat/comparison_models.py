"""Compatibility wrapper for scripts.comparison_models."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.comparison_models import *  # noqa: F401,F403
from scripts.comparison_models import main


if __name__ == "__main__":
    raise SystemExit(main())

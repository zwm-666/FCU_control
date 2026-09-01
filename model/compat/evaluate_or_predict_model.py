"""Compatibility wrapper for scripts.evaluate_or_predict_model."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.evaluate_or_predict_model import *  # noqa: F401,F403
from scripts.evaluate_or_predict_model import main


if __name__ == "__main__":
    main()

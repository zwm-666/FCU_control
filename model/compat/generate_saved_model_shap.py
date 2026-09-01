"""Compatibility wrapper for scripts.generate_saved_model_shap."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.generate_saved_model_shap import *  # noqa: F401,F403
from scripts.generate_saved_model_shap import main


if __name__ == "__main__":
    main()

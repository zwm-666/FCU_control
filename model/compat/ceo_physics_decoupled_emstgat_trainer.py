"""Compatibility wrapper for the physics-decoupled EMSTGAT trainer."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.ceo_physics_decoupled_emstgat_trainer import *  # noqa: F401,F403
from scripts.ceo_physics_decoupled_emstgat_trainer import main


if __name__ == "__main__":
    main()

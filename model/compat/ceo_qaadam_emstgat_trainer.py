"""Compatibility wrapper for core.ceo_qaadam_emstgat_trainer."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ceo_qaadam_emstgat_trainer import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("core.ceo_qaadam_emstgat_trainer", run_name="__main__")

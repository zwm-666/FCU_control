"""Compatibility wrapper for core.model."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.model import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("core.model", run_name="__main__")

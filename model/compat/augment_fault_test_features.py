"""Compatibility wrapper for scripts.augment_fault_test_features."""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.augment_fault_test_features import *  # noqa: F401,F403


if __name__ == "__main__":
    paths = augment_all_fault_files()  # noqa: F405
    for name, path in paths.items():
        print(f"{name}: {path}")

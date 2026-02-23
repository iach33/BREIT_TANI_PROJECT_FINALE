"""Pytest configuration for tani_scoring tests."""

import sys
from pathlib import Path

# Add src/ to path so tani_scoring is importable (Kedro does this at runtime)
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

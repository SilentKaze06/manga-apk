"""Ensure server modules are importable whether pytest runs from repo root or server dir."""
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
ML_DIR = SERVER_DIR / "ml"

for p in (str(SERVER_DIR), str(ML_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

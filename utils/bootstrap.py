"""Make frozen experiments package importable without modifying it."""

from __future__ import annotations

import sys
from pathlib import Path

from config import EXPERIMENTS_ROOT


def ensure_experiments_on_path() -> Path:
    root = EXPERIMENTS_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Frozen experiments repo not found at {root}. "
            "llm_study must sit next to experiments/."
        )
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return root

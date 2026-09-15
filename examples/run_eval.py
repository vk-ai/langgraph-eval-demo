#!/usr/bin/env python3
"""Run the golden-task eval harness and print a markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.runner import main

if __name__ == "__main__":
    raise SystemExit(main())

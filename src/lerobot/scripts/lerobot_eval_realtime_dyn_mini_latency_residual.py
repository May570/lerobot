#!/usr/bin/env python3
"""Wrapper to load dyn-mini registrations before latency-residual realtime eval."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LIBERO_DYN_MINI_ROOT = ROOT / "LIBERO" / "libero_dyn_mini"
LEROBOT_SRC_ROOT = ROOT / "lerobot" / "src"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_DYN_MINI_ROOT / "config"))
sys.path.insert(0, str(LIBERO_DYN_MINI_ROOT / "py"))
sys.path.insert(0, str(LEROBOT_SRC_ROOT))

import libero_dyn_mini_v1  # noqa: F401
from lerobot.scripts.lerobot_eval_realtime3_latency_residual import main


if __name__ == "__main__":
    main()

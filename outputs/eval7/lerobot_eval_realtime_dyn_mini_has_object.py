#!/usr/bin/env python3
"""Workspace-local dyn-mini realtime eval wrapper with has_object-aware policy input support."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_DYN_MINI_ROOT = Path("/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini")

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_DYN_MINI_ROOT / "config"))
sys.path.insert(0, str(LIBERO_DYN_MINI_ROOT / "py"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import libero_dyn_mini_v1  # noqa: F401
from lerobot.scripts.lerobot_eval_realtime3_has_object import main


def _parse_bool_arg(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "`--force_zero_environment_state` expects one of: 1/0, true/false, yes/no, on/off."
    )


def _extract_force_zero_environment_state_flag(argv: list[str]) -> list[str]:
    forwarded: list[str] = [argv[0]]
    force_zero_value: bool | None = None
    for arg in argv[1:]:
        if arg == "--force_zero_environment_state":
            force_zero_value = True
            continue
        if arg.startswith("--force_zero_environment_state="):
            force_zero_value = _parse_bool_arg(arg.split("=", 1)[1])
            continue
        forwarded.append(arg)

    if force_zero_value is not None:
        os.environ["LEROBOT_FORCE_ZERO_ENVIRONMENT_STATE"] = "1" if force_zero_value else "0"

    return forwarded


if __name__ == "__main__":
    sys.argv = _extract_force_zero_environment_state_flag(sys.argv)
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute mean/std stats for Kalman sidecar features.")
    p.add_argument("--kalman-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="default: <kalman-root>/normalization.json")
    p.add_argument(
        "--feature-mode",
        type=str,
        choices=("full10", "posvel6"),
        default="full10",
        help="Feature layout used by policy: full10=[pos,vel,pred_exec,valid], posvel6=[pos,vel].",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output or (args.kalman_root / "normalization.json")

    ep_dirs = sorted(args.kalman_root.glob("episode_*"))
    if not ep_dirs:
        raise FileNotFoundError(f"No episode_* dirs under {args.kalman_root}")

    n = 0
    dim = 10 if args.feature_mode == "full10" else 6
    sum_x = np.zeros((dim,), dtype=np.float64)
    sum_x2 = np.zeros((dim,), dtype=np.float64)

    for ep in ep_dirs:
        pos_p = ep / "kalman_pos.npy"
        vel_p = ep / "kalman_vel.npy"
        if args.feature_mode == "full10":
            pred_p = ep / "kalman_pred_exec.npy"
            valid_p = ep / "kalman_valid.npy"
            if not (pos_p.exists() and vel_p.exists() and pred_p.exists() and valid_p.exists()):
                continue
        else:
            pred_p = None
            valid_p = None
            if not (pos_p.exists() and vel_p.exists()):
                continue

        pos = np.load(pos_p, mmap_mode="r")
        vel = np.load(vel_p, mmap_mode="r")
        if args.feature_mode == "full10":
            pred = np.load(pred_p, mmap_mode="r")
            valid = np.load(valid_p, mmap_mode="r").astype(np.float32)[..., None]
            x = np.concatenate([pos, vel, pred, valid], axis=-1).astype(np.float64)
        else:
            x = np.concatenate([pos, vel], axis=-1).astype(np.float64)
        flat = x.reshape(-1, dim)
        sum_x += flat.sum(axis=0)
        sum_x2 += np.square(flat).sum(axis=0)
        n += flat.shape[0]

    if n == 0:
        raise RuntimeError(f"No valid Kalman arrays found in {args.kalman_root}")

    mean = sum_x / n
    var = np.maximum(sum_x2 / n - np.square(mean), 1e-12)
    std = np.sqrt(var)
    payload = {"count": int(n), "feature_mode": args.feature_mode, "mean": mean.tolist(), "std": std.tolist()}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "count": int(n)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

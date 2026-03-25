#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export mask PNGs from local-flow npy episodes.")
    p.add_argument("--flow-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--episode-start", type=int, default=None)
    p.add_argument("--episode-end", type=int, default=None, help="exclusive")
    p.add_argument("--episode-index", type=int, default=None)
    p.add_argument(
        "--cameras",
        type=str,
        default="image,wrist_image",
        help="comma-separated camera names; corresponding flow key is flow_<camera>",
    )
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def iter_episode_dirs(flow_root: Path, args: argparse.Namespace) -> list[Path]:
    if args.episode_index is not None:
        return [flow_root / f"episode_{args.episode_index:06d}"]
    eps = sorted(flow_root.glob("episode_*"))
    out: list[Path] = []
    for ep in eps:
        idx = int(ep.name.split("_")[-1])
        if args.episode_start is not None and idx < args.episode_start:
            continue
        if args.episode_end is not None and idx >= args.episode_end:
            continue
        out.append(ep)
    return out


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cameras = [x.strip() for x in args.cameras.split(",") if x.strip()]
    if not cameras:
        raise ValueError("--cameras cannot be empty")

    episodes = iter_episode_dirs(args.flow_root, args)
    done = 0
    skipped = 0

    for ep_dir in episodes:
        if not ep_dir.exists():
            continue
        ep_name = ep_dir.name
        dst_ep = args.output_root / ep_name
        done_marker = dst_ep / ".done"
        if args.skip_existing and done_marker.exists():
            skipped += 1
            done += 1
            continue

        meta_path = ep_dir / "arrays.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        for cam in cameras:
            flow_key = f"flow_{cam}"
            if flow_key not in meta:
                continue
            flow = np.load(ep_dir / f"{flow_key}.npy", mmap_mode="r")
            cam_dir = dst_ep / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            # flow shape: [T, H, W, 2]
            mask = np.any(np.abs(flow) > 0, axis=-1).astype(np.uint8) * 255
            for i in range(mask.shape[0]):
                cv2.imwrite(str(cam_dir / f"mask_{i:04d}.png"), mask[i])

        done_marker.write_text("ok\n", encoding="utf-8")
        done += 1
        if done % 10 == 0 or done == len(episodes):
            print(f"[{done}/{len(episodes)}] skipped={skipped}", flush=True)

    print(
        json.dumps(
            {
                "flow_root": str(args.flow_root),
                "output_root": str(args.output_root),
                "episodes_total": len(episodes),
                "episodes_done": done,
                "episodes_skipped": skipped,
                "cameras": cameras,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


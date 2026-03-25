#!/usr/bin/env python3
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def convert_one_episode(src_episode_dir: Path, dst_episode_dir: Path, force: bool = False) -> tuple[str, str]:
    src_npz = src_episode_dir / "flows.npz"
    src_index = src_episode_dir / "index.json"
    dst_done = dst_episode_dir / ".done"

    if not src_npz.exists():
        return (src_episode_dir.name, "missing_npz")
    if dst_done.exists() and not force:
        return (src_episode_dir.name, "skipped")

    dst_episode_dir.mkdir(parents=True, exist_ok=True)
    tmp_done = dst_episode_dir / ".done.tmp"
    if tmp_done.exists():
        tmp_done.unlink()

    with np.load(src_npz, allow_pickle=False) as data:
        metadata = {}
        for key in data.files:
            arr = np.asarray(data[key])
            out_path = dst_episode_dir / f"{key}.npy"
            tmp_path = dst_episode_dir / f".{key}.npy.tmp"
            with tmp_path.open("wb") as f:
                np.save(f, arr, allow_pickle=False)
            tmp_path.replace(out_path)
            metadata[key] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}

    if src_index.exists():
        shutil.copy2(src_index, dst_episode_dir / "index.json")

    with (dst_episode_dir / "arrays.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    tmp_done.write_text("ok\n", encoding="utf-8")
    tmp_done.replace(dst_done)
    return (src_episode_dir.name, "converted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--dst-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src_root
    dst_root = args.dst_root
    dst_root.mkdir(parents=True, exist_ok=True)

    episodes = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    total = len(episodes)
    converted = 0
    skipped = 0
    missing = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(convert_one_episode, ep, dst_root / ep.name, args.force): ep.name
            for ep in episodes
        }
        done = 0
        for fut in as_completed(futs):
            ep_name, status = fut.result()
            done += 1
            if status == "converted":
                converted += 1
            elif status == "skipped":
                skipped += 1
            elif status == "missing_npz":
                missing += 1
            if done % 20 == 0 or done == total:
                print(
                    f"[{done}/{total}] converted={converted} skipped={skipped} missing={missing} "
                    f"last={ep_name}:{status}",
                    flush=True,
                )

    summary = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "total": total,
        "converted": converted,
        "skipped": skipped,
        "missing_npz": missing,
    }
    with (dst_root / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

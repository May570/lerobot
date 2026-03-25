#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "task"


def size_mb(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return int(round(total / (1024 * 1024)))


def read_old_mapping(old_meta: Path):
    tasks = {}
    with (old_meta / "tasks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tasks[int(d["task_index"])] = d["task"]

    name_to_idx = {v: k for k, v in tasks.items()}
    ep_to_task = {}
    task_to_eps = defaultdict(list)
    with (old_meta / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ep = int(d["episode_index"])
            tname = d["tasks"][0]
            tidx = name_to_idx[tname]
            ep_to_task[ep] = tidx
            task_to_eps[tidx].append(ep)

    return tasks, ep_to_task, task_to_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-meta", type=Path, default=Path("/share/project/wujiling/datasets/libero_old/meta"))
    ap.add_argument("--source", type=Path, default=Path("/share/project/wujiling/datasets/libero"))
    ap.add_argument("--out", type=Path, default=Path("/share/project/wujiling/datasets/libero_single"))
    args = ap.parse_args()

    src_info = json.loads((args.source / "meta" / "info.json").read_text())
    if src_info.get("codebase_version") != "v3.0":
        raise ValueError(f"source dataset must be v3.0, got {src_info.get('codebase_version')}")

    task_text, ep_to_task, task_to_eps = read_old_mapping(args.old_meta)
    if not ep_to_task:
        raise ValueError("empty episode mapping from old metadata")

    if args.out.exists():
        for p in args.out.iterdir():
            if p.is_dir():
                # keep directory root but rebuild task dirs
                for sub in p.rglob("*"):
                    pass
        # full reset for deterministic output
        import shutil

        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # Prepare per-task dataset contexts.
    used_names = set()
    ctx = {}
    for tidx in sorted(task_text):
        raw = task_text[tidx]
        name = slugify(raw)
        if name in used_names:
            name = f"{name}_{tidx:02d}"
        used_names.add(name)

        root = args.out / name
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        ctx[tidx] = {
            "task_index_old": tidx,
            "task_text": raw,
            "dir_name": name,
            "root": root,
            "next_episode": 0,
            "next_file": 0,
            "global_index": 0,
            "total_frames": 0,
            "episodes": [],
        }

    data_files = sorted((args.source / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError("no source parquet files found")

    seen_eps = set()

    for fp in data_files:
        df = pd.read_parquet(fp)
        if "episode_index" not in df.columns:
            raise ValueError(f"missing episode_index in {fp}")

        for old_ep, ep_df in df.groupby("episode_index", sort=False):
            old_ep = int(old_ep)
            if old_ep not in ep_to_task:
                continue
            tidx = ep_to_task[old_ep]
            c = ctx[tidx]

            new_ep = c["next_episode"]
            ep_df = ep_df.copy()

            ep_len = len(ep_df)
            ep_df["frame_index"] = list(range(ep_len))
            ep_df["episode_index"] = new_ep
            ep_df["index"] = list(range(c["global_index"], c["global_index"] + ep_len))
            ep_df["task_index"] = 0

            file_num = c["next_file"]
            chunk_idx = file_num // 1000
            file_idx = file_num % 1000
            out_data = c["root"] / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
            out_data.parent.mkdir(parents=True, exist_ok=True)
            ep_df.to_parquet(out_data, index=False)

            c["episodes"].append(
                {
                    "episode_index": new_ep,
                    "data/chunk_index": chunk_idx,
                    "data/file_index": file_idx,
                    "dataset_from_index": c["global_index"],
                    "dataset_to_index": c["global_index"] + ep_len,
                    "tasks": [c["task_text"]],
                    "length": ep_len,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )

            c["global_index"] += ep_len
            c["total_frames"] += ep_len
            c["next_episode"] += 1
            c["next_file"] += 1
            seen_eps.add(old_ep)

    expected_eps = set(ep_to_task.keys())
    if seen_eps != expected_eps:
        miss = sorted(expected_eps - seen_eps)
        extra = sorted(seen_eps - expected_eps)
        raise ValueError(f"episode mismatch. missing={len(miss)} extra={len(extra)}")

    # Write per-task metadata.
    task_map = []
    for tidx in sorted(ctx):
        c = ctx[tidx]
        root = c["root"]

        tasks_df = pd.DataFrame(
            {"task": [c["task_text"]], "task_index": [0]}
        ).set_index("task")
        tasks_df.to_parquet(root / "meta" / "tasks.parquet")

        ep_df = pd.DataFrame(c["episodes"])
        ep_df.to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

        info = dict(src_info)
        info["codebase_version"] = "v3.0"
        info["total_episodes"] = c["next_episode"]
        info["total_frames"] = c["total_frames"]
        info["total_tasks"] = 1
        info["splits"] = {"train": f"0:{c['next_episode']}"}
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        info["video_path"] = src_info.get("video_path")
        info["data_files_size_in_mb"] = size_mb(root / "data")
        info["video_files_size_in_mb"] = 0

        (root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4))

        task_map.append(
            {
                "task_index_old": tidx,
                "task_name": c["task_text"],
                "dir_name": c["dir_name"],
                "episodes": c["next_episode"],
                "frames": c["total_frames"],
                "path": str(root),
            }
        )

    (args.out / "task_map.json").write_text(json.dumps(task_map, ensure_ascii=False, indent=2))

    print(f"done. wrote {len(task_map)} single-task datasets under {args.out}")


if __name__ == "__main__":
    main()

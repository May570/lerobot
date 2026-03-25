#!/usr/bin/env python3
"""Build a single combined LeRobot dataset using task mapping from libero_old metadata.

This script reads task->episode mapping from the legacy v2.1 metadata files
(`tasks.jsonl` and `episodes.jsonl`) and creates one output dataset from an
existing v3 source dataset by selecting episodes for the requested task indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_task_indices(raw: str, max_task_idx: int) -> list[int]:
    raw = raw.strip()
    if raw.lower() == "all":
        return list(range(max_task_idx + 1))

    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx > max_task_idx:
            raise ValueError(f"task index out of range: {idx}, expected 0..{max_task_idx}")
        values.append(idx)

    if not values:
        raise ValueError("no valid task indices provided")

    return sorted(set(values))


def load_task_mapping(old_meta_dir: Path) -> tuple[dict[int, str], dict[int, list[int]]]:
    tasks_path = old_meta_dir / "tasks.jsonl"
    episodes_path = old_meta_dir / "episodes.jsonl"

    if not tasks_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"missing required metadata files in {old_meta_dir}")

    task_idx_to_name: dict[int, str] = {}
    task_name_to_idx: dict[str, int] = {}

    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            idx = int(row["task_index"])
            name = str(row["task"])
            task_idx_to_name[idx] = name
            task_name_to_idx[name] = idx

    task_to_episodes: dict[int, list[int]] = {idx: [] for idx in sorted(task_idx_to_name)}

    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ep_idx = int(row["episode_index"])
            tasks = row.get("tasks", [])
            if not tasks:
                raise ValueError(f"episode {ep_idx} has empty tasks list")
            task_name = tasks[0]
            if task_name not in task_name_to_idx:
                raise ValueError(f"episode {ep_idx} task not found in tasks.jsonl: {task_name}")
            task_idx = task_name_to_idx[task_name]
            task_to_episodes[task_idx].append(ep_idx)

    return task_idx_to_name, task_to_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-meta-dir",
        type=Path,
        default=Path("/share/project/wujiling/datasets/libero_old/meta"),
        help="Path to old libero v2.1 meta directory containing tasks.jsonl/episodes.jsonl",
    )
    parser.add_argument(
        "--source-repo-id",
        type=str,
        default="libero",
        help="Repo id for source v3 dataset",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/share/project/wujiling/datasets/libero"),
        help="Path to source v3 dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/share/project/wujiling/datasets/libero_task_combo_from_old"),
        help="Base output directory. Split API creates dataset under output_root/<split-name>",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="combined",
        help="Split name used by split_dataset (final path: output_root/<split-name>)",
    )
    parser.add_argument(
        "--task-indices",
        type=str,
        default="all",
        help='Comma-separated task indices like "0,1,2" or "all"',
    )
    args = parser.parse_args()

    task_idx_to_name, task_to_episodes = load_task_mapping(args.old_meta_dir)
    max_task_idx = max(task_idx_to_name)
    selected_tasks = parse_task_indices(args.task_indices, max_task_idx)

    selected_episodes = sorted({ep for t in selected_tasks for ep in task_to_episodes[t]})
    if not selected_episodes:
        raise ValueError("selected task indices produced no episodes")

    dataset = LeRobotDataset(repo_id=args.source_repo_id, root=args.source_root)

    invalid_eps = [ep for ep in selected_episodes if ep < 0 or ep >= dataset.meta.total_episodes]
    if invalid_eps:
        raise ValueError(
            f"found {len(invalid_eps)} invalid episode indices for source dataset, "
            f"first few: {invalid_eps[:10]}"
        )

    created = split_dataset(
        dataset=dataset,
        splits={args.split_name: selected_episodes},
        output_dir=args.output_root,
    )
    ds = created[args.split_name]

    print("Built dataset:")
    print(f"  root: {ds.root}")
    print(f"  repo_id: {ds.repo_id}")
    print(f"  selected_tasks: {selected_tasks}")
    print(f"  selected_task_count: {len(selected_tasks)}")
    print(f"  episodes: {ds.meta.total_episodes}")
    print(f"  frames: {ds.meta.total_frames}")


if __name__ == "__main__":
    main()

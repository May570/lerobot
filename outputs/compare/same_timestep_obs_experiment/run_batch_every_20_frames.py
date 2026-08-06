#!/usr/bin/env python3
"""Run same-timestep obs comparison every N frames for one episode.

This script wraps `run_same_timestep_obs_experiment.py` and summarizes the
resulting `diff_summary.json` files into a single CSV/JSON report so we can
inspect how offline-vs-online mismatch evolves over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
RUN_ONCE_SCRIPT = THIS_DIR / "run_same_timestep_obs_experiment.py"
RUNS_ROOT = THIS_DIR / "runs"
DEFAULT_POLICY_PATH = Path(
    "/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/base/checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path(
    "/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/datasets/libero_dyn_mini_balanced500_scripted_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="policy_path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="local/libero_dyn_mini_balanced500_scripted_v2")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-length", type=int, required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        default=str(THIS_DIR / "batch_every_20_frames"),
        help="Directory for run manifest and summary outputs.",
    )
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=None)
    parser.add_argument("--env.episode_start_states_path", dest="env_episode_start_states_path", default=None)
    return parser.parse_args()


def frame_indices(start: int, end: int, stride: int) -> list[int]:
    return list(range(start, end + 1, stride))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(dct: dict[str, Any], *keys: str) -> Any:
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_metrics(diff_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "pre_image_mean_abs_diff": nested_get(diff_summary, "before_policy_pre", "observation.images.image", "mean_abs_diff"),
        "pre_image_max_abs_diff": nested_get(diff_summary, "before_policy_pre", "observation.images.image", "max_abs_diff"),
        "pre_image2_mean_abs_diff": nested_get(diff_summary, "before_policy_pre", "observation.images.image2", "mean_abs_diff"),
        "pre_image2_max_abs_diff": nested_get(diff_summary, "before_policy_pre", "observation.images.image2", "max_abs_diff"),
        "pre_state_mean_abs_diff": nested_get(diff_summary, "before_policy_pre", "observation.state", "mean_abs_diff"),
        "pre_state_max_abs_diff": nested_get(diff_summary, "before_policy_pre", "observation.state", "max_abs_diff"),
        "post_image_mean_abs_diff": nested_get(diff_summary, "after_policy_pre", "observation.images.image", "mean_abs_diff"),
        "post_image_max_abs_diff": nested_get(diff_summary, "after_policy_pre", "observation.images.image", "max_abs_diff"),
        "post_image2_mean_abs_diff": nested_get(diff_summary, "after_policy_pre", "observation.images.image2", "mean_abs_diff"),
        "post_image2_max_abs_diff": nested_get(diff_summary, "after_policy_pre", "observation.images.image2", "max_abs_diff"),
        "post_state_mean_abs_diff": nested_get(diff_summary, "after_policy_pre", "observation.state", "mean_abs_diff"),
        "post_state_max_abs_diff": nested_get(diff_summary, "after_policy_pre", "observation.state", "max_abs_diff"),
        "final_images_mean_abs_diff": nested_get(diff_summary, "final_model_batch", "observation.images", "mean_abs_diff"),
        "final_images_max_abs_diff": nested_get(diff_summary, "final_model_batch", "observation.images", "max_abs_diff"),
        "final_state_mean_abs_diff": nested_get(diff_summary, "final_model_batch", "observation.state", "mean_abs_diff"),
        "final_state_max_abs_diff": nested_get(diff_summary, "final_model_batch", "observation.state", "max_abs_diff"),
        "final_ball_pos_mean_abs_diff": nested_get(diff_summary, "final_model_batch", "observation.ball_pos", "mean_abs_diff"),
        "final_ball_pos_max_abs_diff": nested_get(diff_summary, "final_model_batch", "observation.ball_pos", "max_abs_diff"),
    }


def run_one(frame_index: int, args: argparse.Namespace, output_root: Path) -> Path:
    run_name = f"ep{args.episode_index}_f{frame_index}"
    cmd = [
        sys.executable,
        str(RUN_ONCE_SCRIPT),
        "--policy.path",
        str(Path(args.policy_path).expanduser().resolve()),
        "--dataset.root",
        str(Path(args.dataset_root).expanduser().resolve()),
        "--dataset.repo_id",
        str(args.dataset_repo_id),
        "--episode-index",
        str(int(args.episode_index)),
        "--frame-index",
        str(int(frame_index)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--run-name",
        run_name,
    ]
    if args.skip_visualize:
        cmd.append("--skip-visualize")
    if args.env_init_plan_path:
        cmd.extend(["--env.init_plan_path", str(Path(args.env_init_plan_path).expanduser().resolve())])
    if args.env_episode_start_states_path:
        cmd.extend(
            [
                "--env.episode_start_states_path",
                str(Path(args.env_episode_start_states_path).expanduser().resolve()),
            ]
        )

    env = os.environ.copy()
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("HF_HOME", "/tmp/hf")
    env.setdefault("HF_DATASETS_CACHE", "/tmp/hf/datasets")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg")
    subprocess.run(cmd, check=True, env=env)
    return RUNS_ROOT / run_name


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    end_frame = args.end_frame if args.end_frame is not None else (args.episode_length - 1)
    end_frame = min(int(end_frame), int(args.episode_length) - 1)
    frames = frame_indices(int(args.start_frame), int(end_frame), int(args.stride))

    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for frame_index in frames:
        run_dir = run_one(frame_index=frame_index, args=args, output_root=output_root)
        diff_summary = load_json(run_dir / "diff_summary.json")
        metrics = extract_metrics(diff_summary)
        row = {
            "episode_index": int(args.episode_index),
            "frame_index": int(frame_index),
            "run_dir": str(run_dir),
            **metrics,
        }
        rows.append(row)
        run_manifest.append(
            {
                "frame_index": int(frame_index),
                "run_dir": str(run_dir),
                "diff_summary_json": str(run_dir / "diff_summary.json"),
                "metadata_json": str(run_dir / "metadata.json"),
            }
        )

    summary = {
        "experiment": "same_timestep_obs_batch_stride",
        "episode_index": int(args.episode_index),
        "episode_length": int(args.episode_length),
        "stride": int(args.stride),
        "frames": frames,
        "policy_path": str(Path(args.policy_path).expanduser().resolve()),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "rows": rows,
    }

    summary_json = output_root / f"ep{args.episode_index}_stride{args.stride}_summary.json"
    summary_csv = output_root / f"ep{args.episode_index}_stride{args.stride}_summary.csv"
    manifest_json = output_root / f"ep{args.episode_index}_stride{args.stride}_runs.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_json.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(summary_csv, rows)

    print(f"Saved summary JSON to: {summary_json}")
    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Saved run manifest to: {manifest_json}")


if __name__ == "__main__":
    main()

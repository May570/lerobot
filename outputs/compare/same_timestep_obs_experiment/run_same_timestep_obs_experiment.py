#!/usr/bin/env python3
"""Run the same-timestep offline-vs-online observation experiment.

This wrapper reuses the existing compare pipeline, but organizes outputs under
`outputs/compare/same_timestep_obs_experiment/` and applies stricter defaults
for dyn-mini:

1. Pick one dataset episode and one frame index t.
2. Reset the online env to the corresponding initial scene.
3. If t > 0, replay the offline actions up to t.
4. Compare offline and online observations at that same logical timestep.
5. Save pre/post/final comparisons plus image visualizations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPARE_ROOT = REPO_ROOT / "outputs" / "compare"
EXPERIMENT_ROOT = COMPARE_ROOT / "same_timestep_obs_experiment"
CORE_COMPARE_SCRIPT = COMPARE_ROOT / "compare_dyn_mini_model_inputs.py"
VIS_SCRIPT = COMPARE_ROOT / "visualize_compare_dyn_mini_inputs.py"

DEFAULT_POLICY_PATH = Path(
    "/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/base/checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path(
    "/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/datasets/libero_dyn_mini_balanced500_scripted_v2"
)
DEFAULT_INIT_PLAN_PATH = Path(
    "/home/admin123/桌面/wjl/lerobot/outputs/eval9/fixed_starts/rolling_ball_to_bowl.eval_from_dataset_balanced500_scripted_v2_first200.jsonl"
)
DEFAULT_EPISODE_START_CACHE_PATH = Path(
    "/home/admin123/桌面/wjl/lerobot/outputs/eval9/fixed_starts/libero_dyn_mini_dataset_first200_seed1000_b2_ep200.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="policy_path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="local/libero_dyn_mini_balanced500_scripted_v2")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=str(DEFAULT_INIT_PLAN_PATH))
    parser.add_argument(
        "--env.episode_start_states_path",
        dest="env_episode_start_states_path",
        default=str(DEFAULT_EPISODE_START_CACHE_PATH),
    )
    parser.add_argument("--skip-visualize", action="store_true")
    return parser.parse_args()


def sanitize_token(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("._") or "run"


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        run_name = sanitize_token(args.run_name)
    else:
        policy_label = sanitize_token(Path(args.policy_path).parent.name or Path(args.policy_path).name)
        run_name = f"{policy_label}_ep{args.episode_index:04d}_f{args.frame_index:04d}"
    run_dir = EXPERIMENT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_subprocess(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_dir(args)

    env = os.environ.copy()
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("HF_HOME", "/tmp/hf")
    env.setdefault("HF_DATASETS_CACHE", "/tmp/hf/datasets")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg")

    compare_cmd = [
        sys.executable,
        str(CORE_COMPARE_SCRIPT),
        "--policy.path",
        str(Path(args.policy_path).expanduser().resolve()),
        "--dataset.root",
        str(Path(args.dataset_root).expanduser().resolve()),
        "--dataset.repo_id",
        str(args.dataset_repo_id),
        "--episode-index",
        str(int(args.episode_index)),
        "--frame-index",
        str(int(args.frame_index)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--output-dir",
        str(run_dir),
        "--env.init_plan_path",
        str(Path(args.env_init_plan_path).expanduser().resolve()),
        "--env.episode_start_states_path",
        str(Path(args.env_episode_start_states_path).expanduser().resolve()),
    ]
    run_subprocess(compare_cmd, env=env)

    if not args.skip_visualize:
        vis_cmd = [
            sys.executable,
            str(VIS_SCRIPT),
            "--compare-dir",
            str(run_dir),
        ]
        run_subprocess(vis_cmd, env=env)

    manifest = {
        "experiment": "same_timestep_offline_online_obs",
        "run_dir": str(run_dir),
        "policy_path": str(Path(args.policy_path).expanduser().resolve()),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "episode_index": int(args.episode_index),
        "frame_index": int(args.frame_index),
        "seed": int(args.seed),
        "device": str(args.device),
        "env_init_plan_path": str(Path(args.env_init_plan_path).expanduser().resolve()),
        "env_episode_start_states_path": str(Path(args.env_episode_start_states_path).expanduser().resolve()),
        "core_compare_script": str(CORE_COMPARE_SCRIPT),
        "visualize_script": None if args.skip_visualize else str(VIS_SCRIPT),
        "artifacts": {
            "metadata_json": str(run_dir / "metadata.json"),
            "diff_summary_json": str(run_dir / "diff_summary.json"),
            "offline_before_policy_pre_pt": str(run_dir / "offline_before_policy_pre.pt"),
            "online_before_policy_pre_pt": str(run_dir / "online_before_policy_pre.pt"),
            "offline_after_policy_pre_pt": str(run_dir / "offline_after_policy_pre.pt"),
            "online_after_policy_pre_pt": str(run_dir / "online_after_policy_pre.pt"),
            "offline_final_model_batch_pt": str(run_dir / "offline_final_model_batch.pt"),
            "online_final_model_batch_pt": str(run_dir / "online_final_model_batch.pt"),
            "viz_dir": None if args.skip_visualize else str(run_dir / "viz_before_policy_pre"),
        },
    }
    manifest_path = run_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved same-timestep experiment results to: {run_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

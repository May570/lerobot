#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
REPO_SRC = Path(__file__).resolve().parents[2]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


_DYN_MINI_PLAN_CANDIDATES = [
    "rolling_ball_to_bowl.collection_plan_balanced500_v1.jsonl",
    "rolling_ball_to_bowl.collection_plan_diverse_repro_v2.jsonl",
    "rolling_ball_to_bowl.collection_plan_repro_var_v2.jsonl",
    "rolling_ball_to_bowl.eval_plan_balanced_holdout_v1.jsonl",
    "rolling_ball_to_bowl.collection_plan.jsonl",
]


def _candidate_dyn_mini_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []
    env_root = Path(os.environ["LIBERO_DYN_MINI_ROOT"]).expanduser() if "LIBERO_DYN_MINI_ROOT" in os.environ else None
    if env_root is not None:
        candidates.append(env_root)
    candidates.append(repo_root.parent / "LIBERO" / "libero_dyn_mini")
    candidates.append(Path("/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini"))
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            out.append(candidate)
    return out


def _configure_local_dyn_mini() -> None:
    for root in _candidate_dyn_mini_roots():
        py_dir = root / "py"
        config_dir = root / "config"
        if py_dir.is_dir() and str(py_dir) not in sys.path:
            sys.path.insert(0, str(py_dir))
        if config_dir.is_dir() and not os.environ.get("LIBERO_CONFIG_PATH"):
            os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
        try:
            import libero_dyn_mini_v1  # noqa: F401
            return
        except Exception:
            continue


_configure_local_dyn_mini()

from libero.libero import get_libero_path
from lerobot.envs.libero import create_libero_envs, save_episode_start_states
from lerobot.utils.random_utils import set_seed


def _resolve_dyn_mini_default_plan_path() -> Path | None:
    search_dirs: list[Path] = []
    for root in _candidate_dyn_mini_roots():
        candidate = root / "init_files" / "libero_dyn_mini"
        if candidate.is_dir():
            search_dirs.append(candidate)
    try:
        search_dirs.append(Path(get_libero_path("init_states")) / "libero_dyn_mini")
    except Exception:
        pass
    seen: set[Path] = set()
    deduped_dirs: list[Path] = []
    for path in search_dirs:
        path = path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        deduped_dirs.append(path)
    for base_dir in deduped_dirs:
        for name in _DYN_MINI_PLAN_CANDIDATES:
            candidate = base_dir / name
            if candidate.exists():
                return candidate
    return None


def _parse_task_ids(value: str) -> list[int] | None:
    text = value.strip()
    if not text:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed LIBERO episode-start cache for evaluation.")
    parser.add_argument("--env.task", dest="env_task", default="libero_dyn_mini")
    parser.add_argument("--env.task_ids", dest="env_task_ids", default="0")
    parser.add_argument("--env.camera_name", dest="env_camera_name", default="agentview_image,robot0_eye_in_hand_image")
    parser.add_argument("--env.obs_type", dest="env_obs_type", default="pixels_agent_pos")
    parser.add_argument("--env.render_mode", dest="env_render_mode", default="rgb_array")
    parser.add_argument("--env.control_mode", dest="env_control_mode", default="relative")
    parser.add_argument("--env.init_states", dest="env_init_states", action="store_true", default=True)
    parser.add_argument("--env.no_init_states", dest="env_init_states", action="store_false")
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=None)
    parser.add_argument("--env.init_plan_loop", dest="env_init_plan_loop", action="store_true", default=True)
    parser.add_argument("--env.no_init_plan_loop", dest="env_init_plan_loop", action="store_false")
    parser.add_argument("--env.init_plan_default_direction_deg", dest="env_init_plan_default_direction_deg", type=float, default=270.0)
    parser.add_argument("--env.init_plan_default_speed", dest="env_init_plan_default_speed", type=float, default=0.30)
    parser.add_argument("--env.init_plan_default_ball_start_x", dest="env_init_plan_default_ball_start_x", type=float, default=0.04)
    parser.add_argument("--env.init_plan_default_ball_start_y", dest="env_init_plan_default_ball_start_y", type=float, default=0.26)
    parser.add_argument("--env.init_plan_ball_start_xy_safety_scale", dest="env_init_plan_ball_start_xy_safety_scale", type=float, default=0.92)
    parser.add_argument("--env.init_plan_ball_start_z_clearance", dest="env_init_plan_ball_start_z_clearance", type=float, default=0.0015)
    parser.add_argument("--env.init_plan_launch_settle_steps", dest="env_init_plan_launch_settle_steps", type=int, default=6)
    parser.add_argument("--env.init_plan_launch_ramp_steps", dest="env_init_plan_launch_ramp_steps", type=int, default=8)
    parser.add_argument("--env.init_plan_warmup_steps", dest="env_init_plan_warmup_steps", type=int, default=0)
    parser.add_argument("--env.ball_grasp_eval_mode", dest="env_ball_grasp_eval_mode", default="strict")
    parser.add_argument(
        "--env.ball_grasp_strict_require_pad_contact",
        dest="env_ball_grasp_strict_require_pad_contact",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--env.no_ball_grasp_strict_require_pad_contact",
        dest="env_ball_grasp_strict_require_pad_contact",
        action="store_false",
    )
    parser.add_argument("--env.ball_grasp_strict_lift_multiplier", dest="env_ball_grasp_strict_lift_multiplier", type=float, default=1.2)
    parser.add_argument(
        "--env.ball_grasp_strict_grip_center_max_dist",
        dest="env_ball_grasp_strict_grip_center_max_dist",
        type=float,
        default=0.045,
    )
    parser.add_argument("--env.observation_height", dest="env_observation_height", type=int, default=360)
    parser.add_argument("--env.observation_width", dest="env_observation_width", type=int, default=360)
    parser.add_argument("--eval.batch_size", dest="eval_batch_size", type=int, default=2)
    parser.add_argument("--eval.n_episodes", dest="eval_n_episodes", type=int, default=100)
    parser.add_argument("--seed", dest="seed", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    task_ids = _parse_task_ids(args.env_task_ids)
    init_plan_path = args.env_init_plan_path
    if args.env_task == "libero_dyn_mini" and not init_plan_path:
        default_plan = _resolve_dyn_mini_default_plan_path()
        if default_plan is None:
            raise FileNotFoundError("Unable to resolve the default dyn-mini init plan.")
        init_plan_path = str(default_plan)

    gym_kwargs: dict[str, object] = {
        "obs_type": args.env_obs_type,
        "render_mode": args.env_render_mode,
        "observation_height": args.env_observation_height,
        "observation_width": args.env_observation_width,
        "ball_grasp_eval_mode": args.env_ball_grasp_eval_mode,
        "ball_grasp_strict_require_pad_contact": args.env_ball_grasp_strict_require_pad_contact,
        "ball_grasp_strict_lift_multiplier": args.env_ball_grasp_strict_lift_multiplier,
        "ball_grasp_strict_grip_center_max_dist": args.env_ball_grasp_strict_grip_center_max_dist,
    }
    if task_ids is not None:
        gym_kwargs["task_ids"] = task_ids
    if init_plan_path:
        gym_kwargs["init_plan_path"] = init_plan_path
        gym_kwargs["init_plan_loop"] = args.env_init_plan_loop
        gym_kwargs["init_plan_default_direction_deg"] = args.env_init_plan_default_direction_deg
        gym_kwargs["init_plan_default_speed"] = args.env_init_plan_default_speed
        gym_kwargs["init_plan_default_ball_start_x"] = args.env_init_plan_default_ball_start_x
        gym_kwargs["init_plan_default_ball_start_y"] = args.env_init_plan_default_ball_start_y
        gym_kwargs["init_plan_ball_start_xy_safety_scale"] = args.env_init_plan_ball_start_xy_safety_scale
        gym_kwargs["init_plan_ball_start_z_clearance"] = args.env_init_plan_ball_start_z_clearance
        gym_kwargs["init_plan_launch_settle_steps"] = args.env_init_plan_launch_settle_steps
        gym_kwargs["init_plan_launch_ramp_steps"] = args.env_init_plan_launch_ramp_steps
        gym_kwargs["init_plan_warmup_steps"] = args.env_init_plan_warmup_steps

    envs = create_libero_envs(
        task=args.env_task,
        n_envs=args.eval_batch_size,
        gym_kwargs=gym_kwargs,
        camera_name=args.env_camera_name,
        init_states=args.env_init_states,
        env_cls=gym.vector.SyncVectorEnv,
        control_mode=args.env_control_mode,
        episode_length=None,
    )

    task_group = args.env_task.split(",")[0].strip()
    task_map = envs[task_group]
    if len(task_map) != 1:
        raise ValueError(
            "Cache generation currently expects exactly one selected task. "
            f"Got task ids: {sorted(task_map.keys())}"
        )
    vec_env = next(iter(task_map.values()))
    n_batches = math.ceil(args.eval_n_episodes / vec_env.num_envs)
    sim_states: list[np.ndarray] = []
    table_body_quats: list[np.ndarray] = []
    saw_missing_table_quat = False

    try:
        for batch_ix in range(n_batches):
            seeds = list(range(args.seed + batch_ix * vec_env.num_envs, args.seed + (batch_ix + 1) * vec_env.num_envs))
            vec_env.reset(seed=seeds)
            remaining = args.eval_n_episodes - len(sim_states)
            capture_now = min(vec_env.num_envs, remaining)
            for env_ix in range(capture_now):
                env = vec_env.envs[env_ix]
                sim_states.append(env.get_sim_state())
                table_quat = env.get_table_body_quat()
                if table_quat is None:
                    saw_missing_table_quat = True
                else:
                    table_body_quats.append(np.asarray(table_quat, dtype=np.float64))
    finally:
        vec_env.close()

    table_quats_array = None if saw_missing_table_quat else np.stack(table_body_quats, axis=0)
    metadata = {
        "version": 1,
        "task": args.env_task,
        "task_ids": task_ids,
        "n_envs": args.eval_batch_size,
        "n_episodes": args.eval_n_episodes,
        "seed": args.seed,
        "init_plan_path": init_plan_path,
    }
    output_path = save_episode_start_states(
        args.output,
        sim_states=np.stack(sim_states, axis=0),
        table_body_quats=table_quats_array,
        metadata=metadata,
    )
    print(f"Saved {len(sim_states)} episode-start states to {output_path}")


if __name__ == "__main__":
    main()

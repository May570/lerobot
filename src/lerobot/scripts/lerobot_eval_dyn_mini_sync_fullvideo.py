#!/usr/bin/env python3
"""Sync eval wrapper for LIBERO dyn-mini with full video export and eval summaries."""

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

ROOT = Path(__file__).resolve().parents[4]
LIBERO_DYN_MINI_ROOT = ROOT / "LIBERO" / "libero_dyn_mini"
LEROBOT_SRC_ROOT = ROOT / "lerobot" / "src"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_DYN_MINI_ROOT / "config"))
sys.path.insert(0, str(LIBERO_DYN_MINI_ROOT / "py"))
sys.path.insert(0, str(LEROBOT_SRC_ROOT))

import libero_dyn_mini_v1  # noqa: F401

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.libero_compat import (
    apply_rename_map_to_preprocessor,
    resolve_libero_rename_map,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm


_DYN_MINI_PLAN_CANDIDATES = [
    "rolling_ball_to_bowl.collection_plan_balanced500_v1.jsonl",
    "rolling_ball_to_bowl.collection_plan_diverse_repro_v2.jsonl",
    "rolling_ball_to_bowl.collection_plan_repro_var_v2.jsonl",
    "rolling_ball_to_bowl.eval_plan_balanced_holdout_v1.jsonl",
]


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    rollout_fps: float | None = None,
) -> dict[str, Any]:
    """Run a synchronous rollout and record ball grasp events when available."""
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    all_ball_grasp_events = []

    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)

    track_ball_grasp = False
    grasp_prev = np.zeros(env.num_envs, dtype=bool)
    if hasattr(env, "call"):
        try:
            raw_grasp = env.call("is_ball_grasped")
            if isinstance(raw_grasp, (list, tuple)) and len(raw_grasp) == env.num_envs:
                grasp_prev = np.asarray(raw_grasp, dtype=bool)
                track_ball_grasp = True
                logging.info("[sync-eval][grasp] enabled grasp event tracking.")
        except Exception:  # noqa: BLE001
            track_ball_grasp = False

    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        if rollout_fps is not None and rollout_fps > 0 and OBS_STATE in observation:
            batch_size = int(observation[OBS_STATE].shape[0])
            obs_state = observation[OBS_STATE]
            ts = torch.full(
                (batch_size,),
                float(step) / float(rollout_fps),
                device=obs_state.device,
                dtype=obs_state.dtype,
            )
            observation["timestamp"] = ts

        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        grasp_event_step = np.zeros(env.num_envs, dtype=np.int32)
        if track_ball_grasp:
            raw_grasp = env.call("is_ball_grasped")
            if not isinstance(raw_grasp, (list, tuple)) or len(raw_grasp) != env.num_envs:
                raise RuntimeError(
                    "[sync-eval][grasp] env.call('is_ball_grasped') returned invalid shape "
                    f"{type(raw_grasp)} len={len(raw_grasp) if isinstance(raw_grasp, (list, tuple)) else 'NA'}."
                )
            grasp_now = np.asarray(raw_grasp, dtype=bool)
            grasp_event_step = np.logical_and(np.logical_not(grasp_prev), grasp_now).astype(np.int32)
            grasp_prev = grasp_now
            if int(np.sum(grasp_event_step)) > 0:
                logging.info("[sync-eval][grasp] step=%d new_grasps=%d", step, int(np.sum(grasp_event_step)))

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        all_ball_grasp_events.append(torch.from_numpy(grasp_event_step))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    ret: dict[str, Any] = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        "ball_grasp_event": torch.stack(all_ball_grasp_events, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    rollout_fps: float | None = None,
) -> dict[str, Any]:
    """Evaluate one vector env with synchronous inference and ball-grasp bookkeeping."""
    if return_episode_data:
        raise NotImplementedError("`return_episode_data=True` is not supported in this wrapper.")
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    start = time.time()
    policy.eval()

    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    all_successes: list[bool] = []
    ball_grasp_counts: list[int] = []
    ball_grasp_successes: list[bool] = []
    all_seeds: list[int | None] = []
    threads: list[threading.Thread] = []
    n_episodes_rendered = 0

    def render_frame(env: gym.vector.VectorEnv):
        nonlocal n_episodes_rendered
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )

        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=False,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            rollout_fps=rollout_fps,
        )

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()

        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())

        if "ball_grasp_event" in rollout_data:
            batch_grasp_counts = einops.reduce((rollout_data["ball_grasp_event"] * mask), "b n -> b", "sum")
            batch_grasp_counts = batch_grasp_counts.to(torch.int64)
            batch_grasp_success = batch_grasp_counts > 0
            ball_grasp_counts.extend(batch_grasp_counts.tolist())
            ball_grasp_successes.extend(batch_grasp_success.tolist())
        else:
            ball_grasp_counts.extend([0] * env.num_envs)
            ball_grasp_successes.extend([False] * env.num_envs)

        if seeds:
            all_seeds.extend(list(seeds))
        else:
            all_seeds.extend([None] * env.num_envs)

        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)
            for stacked_frames, done_index in zip(batch_stacked_frames, done_indices.flatten().tolist(), strict=False):
                if n_episodes_rendered >= max_episodes_rendered:
                    break
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix({"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"})

    for thread in threads:
        thread.join()

    per_episode_records = []
    for i in range(n_episodes):
        per_episode_records.append(
            {
                "episode_ix": i,
                "sum_reward": sum_rewards[i],
                "max_reward": max_rewards[i],
                "success": all_successes[i],
                "seed": all_seeds[i] if i < len(all_seeds) else None,
                "ball_grasp_count": int(ball_grasp_counts[i]) if i < len(ball_grasp_counts) else 0,
                "ball_grasp_success": bool(ball_grasp_successes[i]) if i < len(ball_grasp_successes) else False,
            }
        )

    info: dict[str, Any] = {
        "per_episode": per_episode_records,
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "avg_ball_grasp_count": float(np.nanmean(ball_grasp_counts[:n_episodes])),
            "pc_ball_grasp_success": float(np.nanmean(ball_grasp_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }
    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths
    return info


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    rollout_fps: float | None = None,
) -> dict[str, Any]:
    """Evaluate nested env dict and preserve ball-grasp statistics per task/group/overall."""
    if return_episode_data:
        raise NotImplementedError("`return_episode_data=True` is not supported in this wrapper.")
    if max_parallel_tasks != 1:
        logging.warning("Ignoring max_parallel_tasks=%s; this wrapper evaluates tasks sequentially.", max_parallel_tasks)

    start_t = time.time()

    overall_sum_rewards: list[float] = []
    overall_max_rewards: list[float] = []
    overall_successes: list[bool] = []
    overall_ball_grasp_counts: list[int] = []
    overall_ball_grasp_successes: list[bool] = []
    overall_video_paths: list[str] = []
    ball_grasp_success_per_episode: list[dict[str, Any]] = []
    per_task_infos: list[dict[str, Any]] = []
    global_episode_ix = 0
    per_group_raw: dict[str, dict[str, list[Any]]] = {}

    for task_group, group_envs in envs.items():
        group_sum_rewards: list[float] = []
        group_max_rewards: list[float] = []
        group_successes: list[bool] = []
        group_ball_grasp_counts: list[int] = []
        group_ball_grasp_successes: list[bool] = []
        group_video_paths: list[str] = []
        group_ball_grasp_success_episode_indices: list[int] = []

        for task_id, env in group_envs.items():
            task_videos_dir = None
            if videos_dir is not None:
                task_videos_dir = videos_dir / f"{task_group}_{task_id}"
                task_videos_dir.mkdir(parents=True, exist_ok=True)

            task_result = eval_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=n_episodes,
                max_episodes_rendered=max_episodes_rendered,
                videos_dir=task_videos_dir,
                return_episode_data=False,
                start_seed=start_seed,
                rollout_fps=rollout_fps,
            )

            per_episode = task_result["per_episode"]
            task_sum_rewards = [ep["sum_reward"] for ep in per_episode]
            task_max_rewards = [ep["max_reward"] for ep in per_episode]
            task_successes = [ep["success"] for ep in per_episode]
            task_ball_grasp_counts = [int(ep.get("ball_grasp_count", 0)) for ep in per_episode]
            task_ball_grasp_successes = [bool(ep.get("ball_grasp_success", False)) for ep in per_episode]
            task_video_paths = task_result.get("video_paths", [])
            task_ball_grasp_success_episode_indices = [
                int(ep.get("episode_ix", idx)) for idx, ep in enumerate(per_episode) if bool(ep.get("ball_grasp_success", False))
            ]

            group_sum_rewards.extend(task_sum_rewards)
            group_max_rewards.extend(task_max_rewards)
            group_successes.extend(task_successes)
            group_ball_grasp_counts.extend(task_ball_grasp_counts)
            group_ball_grasp_successes.extend(task_ball_grasp_successes)
            group_video_paths.extend(task_video_paths)

            overall_sum_rewards.extend(task_sum_rewards)
            overall_max_rewards.extend(task_max_rewards)
            overall_successes.extend(task_successes)
            overall_ball_grasp_counts.extend(task_ball_grasp_counts)
            overall_ball_grasp_successes.extend(task_ball_grasp_successes)
            overall_video_paths.extend(task_video_paths)

            for idx, ep in enumerate(per_episode):
                if bool(ep.get("ball_grasp_success", False)):
                    group_ball_grasp_success_episode_indices.append(global_episode_ix)
                    ball_grasp_success_per_episode.append(
                        {
                            "global_episode_ix": global_episode_ix,
                            "task_group": task_group,
                            "task_id": int(task_id),
                            "episode_ix_in_task": int(ep.get("episode_ix", idx)),
                            "success": bool(ep.get("success", False)),
                            "ball_grasp_count": int(ep.get("ball_grasp_count", 0)),
                            "seed": ep.get("seed"),
                        }
                    )
                global_episode_ix += 1

            per_task_infos.append(
                {
                    "task_group": task_group,
                    "task_id": task_id,
                    "metrics": {
                        "sum_rewards": task_sum_rewards,
                        "max_rewards": task_max_rewards,
                        "successes": task_successes,
                        "ball_grasp_counts": task_ball_grasp_counts,
                        "ball_grasp_successes": task_ball_grasp_successes,
                        "ball_grasp_success_episode_indices": task_ball_grasp_success_episode_indices,
                        "video_paths": task_video_paths,
                    },
                }
            )

        per_group_raw[task_group] = {
            "sum_rewards": group_sum_rewards,
            "max_rewards": group_max_rewards,
            "successes": group_successes,
            "ball_grasp_counts": group_ball_grasp_counts,
            "ball_grasp_successes": group_ball_grasp_successes,
            "ball_grasp_success_episode_indices": group_ball_grasp_success_episode_indices,
            "video_paths": group_video_paths,
        }

    def _agg_from_list(xs: list[Any]) -> float:
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    groups_aggregated = {}
    for group, acc in per_group_raw.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "avg_ball_grasp_count": _agg_from_list(acc["ball_grasp_counts"]),
            "pc_ball_grasp_success": (
                _agg_from_list(acc["ball_grasp_successes"]) * 100 if acc["ball_grasp_successes"] else float("nan")
            ),
            "ball_grasp_success_episode_indices": list(acc["ball_grasp_success_episode_indices"]),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall_sum_rewards),
        "avg_max_reward": _agg_from_list(overall_max_rewards),
        "pc_success": _agg_from_list(overall_successes) * 100 if overall_successes else float("nan"),
        "avg_ball_grasp_count": _agg_from_list(overall_ball_grasp_counts),
        "pc_ball_grasp_success": (
            _agg_from_list(overall_ball_grasp_successes) * 100 if overall_ball_grasp_successes else float("nan")
        ),
        "ball_grasp_success_episode_indices": [
            item["global_episode_ix"] for item in ball_grasp_success_per_episode
        ],
        "n_episodes": len(overall_sum_rewards),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall_sum_rewards)),
        "video_paths": list(overall_video_paths),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
        "ball_grasp_success_per_episode": ball_grasp_success_per_episode,
    }


def _resolve_dyn_mini_default_plan_path() -> Path | None:
    try:
        from libero.libero import get_libero_path
    except Exception:  # noqa: BLE001
        return None

    try:
        init_root = Path(get_libero_path("init_states")) / "libero_dyn_mini"
    except Exception:  # noqa: BLE001
        return None

    if not init_root.exists():
        return None

    for filename in _DYN_MINI_PLAN_CANDIDATES:
        candidate = init_root / filename
        if candidate.exists():
            return candidate

    return None


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.nanmean(np.asarray(xs, dtype=float)))


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(value):
        return None
    return value


def _success_count(n_episodes: int | None, pc_success: float | None) -> int | None:
    if n_episodes is None or pc_success is None:
        return None
    return int(round(float(n_episodes) * float(pc_success) / 100.0))


def _build_eval_info_summary(info: dict[str, Any]) -> dict[str, Any]:
    overall_raw = info.get("overall", {})
    overall_n_episodes = _int_or_none(overall_raw.get("n_episodes"))
    overall_pc_success = _float_or_none(overall_raw.get("pc_success"))
    overall_pc_ball_grasp_success = _float_or_none(overall_raw.get("pc_ball_grasp_success"))
    overall_summary = {
        "avg_sum_reward": _float_or_none(overall_raw.get("avg_sum_reward")),
        "avg_max_reward": _float_or_none(overall_raw.get("avg_max_reward")),
        "pc_success": overall_pc_success,
        "avg_ball_grasp_count": _float_or_none(overall_raw.get("avg_ball_grasp_count")),
        "pc_ball_grasp_success": overall_pc_ball_grasp_success,
        "n_episodes": overall_n_episodes,
        "n_success": _success_count(overall_n_episodes, overall_pc_success),
        "n_ball_grasp_success": _success_count(overall_n_episodes, overall_pc_ball_grasp_success),
        "ball_grasp_success_episode_indices": list(overall_raw.get("ball_grasp_success_episode_indices", []) or []),
        "eval_s": _float_or_none(overall_raw.get("eval_s")),
        "eval_ep_s": _float_or_none(overall_raw.get("eval_ep_s")),
        "n_videos": len(overall_raw.get("video_paths", []) or []),
    }

    per_group_summary: dict[str, dict[str, Any]] = {}
    for group, group_raw in (info.get("per_group") or {}).items():
        n_episodes = _int_or_none(group_raw.get("n_episodes"))
        pc_success = _float_or_none(group_raw.get("pc_success"))
        pc_ball_grasp_success = _float_or_none(group_raw.get("pc_ball_grasp_success"))
        per_group_summary[str(group)] = {
            "avg_sum_reward": _float_or_none(group_raw.get("avg_sum_reward")),
            "avg_max_reward": _float_or_none(group_raw.get("avg_max_reward")),
            "pc_success": pc_success,
            "avg_ball_grasp_count": _float_or_none(group_raw.get("avg_ball_grasp_count")),
            "pc_ball_grasp_success": pc_ball_grasp_success,
            "n_episodes": n_episodes,
            "n_success": _success_count(n_episodes, pc_success),
            "n_ball_grasp_success": _success_count(n_episodes, pc_ball_grasp_success),
            "ball_grasp_success_episode_indices": list(group_raw.get("ball_grasp_success_episode_indices", []) or []),
            "n_videos": len(group_raw.get("video_paths", []) or []),
        }

    per_task_summary: list[dict[str, Any]] = []
    for task_raw in info.get("per_task", []) or []:
        metrics = task_raw.get("metrics", {}) or {}
        sum_rewards = [float(x) for x in metrics.get("sum_rewards", []) or []]
        max_rewards = [float(x) for x in metrics.get("max_rewards", []) or []]
        successes = [bool(x) for x in metrics.get("successes", []) or []]
        video_paths = list(metrics.get("video_paths", []) or [])
        task_summary = {
            "task_group": str(task_raw.get("task_group")),
            "task_id": _int_or_none(task_raw.get("task_id")),
            "avg_sum_reward": _mean(sum_rewards),
            "avg_max_reward": _mean(max_rewards),
            "pc_success": (100.0 * float(np.mean(successes))) if successes else float("nan"),
            "avg_ball_grasp_count": _mean([float(x) for x in metrics.get("ball_grasp_counts", []) or []]),
            "pc_ball_grasp_success": (
                100.0 * float(np.mean([bool(x) for x in metrics.get("ball_grasp_successes", []) or []]))
                if metrics.get("ball_grasp_successes", [])
                else float("nan")
            ),
            "n_episodes": len(successes),
            "n_success": int(sum(successes)),
            "n_ball_grasp_success": int(sum(bool(x) for x in metrics.get("ball_grasp_successes", []) or [])),
            "ball_grasp_success_episode_indices": list(metrics.get("ball_grasp_success_episode_indices", []) or []),
            "n_videos": len(video_paths),
        }
        per_task_summary.append(task_summary)

    per_task_summary.sort(key=lambda item: (item["task_group"], item["task_id"] or -1))

    return {
        "overall": overall_summary,
        "per_group": per_group_summary,
        "per_task": per_task_summary,
        "ball_grasp_success_per_episode": list(info.get("ball_grasp_success_per_episode", []) or []),
    }


def _format_metric(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:  # noqa: BLE001
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _format_eval_info_summary_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    overall = summary.get("overall", {}) or {}
    lines.append("Overall")
    lines.append(
        "  avg_sum_reward={avg_sum} avg_max_reward={avg_max} pc_success={pc}% "
        "avg_ball_grasp_count={grasp_avg} pc_ball_grasp_success={grasp_pc}% "
        "n_episodes={n_ep} n_success={n_ok} n_ball_grasp_success={n_grasp_ok} "
        "n_videos={n_vid} eval_s={eval_s} eval_ep_s={eval_ep_s}".format(
            avg_sum=_format_metric(overall.get("avg_sum_reward")),
            avg_max=_format_metric(overall.get("avg_max_reward")),
            pc=_format_metric(overall.get("pc_success"), digits=1),
            grasp_avg=_format_metric(overall.get("avg_ball_grasp_count")),
            grasp_pc=_format_metric(overall.get("pc_ball_grasp_success"), digits=1),
            n_ep=overall.get("n_episodes", "NA"),
            n_ok=overall.get("n_success", "NA"),
            n_grasp_ok=overall.get("n_ball_grasp_success", "NA"),
            n_vid=overall.get("n_videos", "NA"),
            eval_s=_format_metric(overall.get("eval_s")),
            eval_ep_s=_format_metric(overall.get("eval_ep_s")),
        )
    )
    lines.append(f"  ball_grasp_success_episode_indices={overall.get('ball_grasp_success_episode_indices', [])}")

    lines.append("")
    lines.append("Per Group")
    for group, group_summary in sorted((summary.get("per_group") or {}).items(), key=lambda item: item[0]):
        lines.append(
            "  {group}: avg_sum_reward={avg_sum} avg_max_reward={avg_max} pc_success={pc}% "
            "avg_ball_grasp_count={grasp_avg} pc_ball_grasp_success={grasp_pc}% "
            "n_episodes={n_ep} n_success={n_ok} n_ball_grasp_success={n_grasp_ok} n_videos={n_vid}".format(
                group=group,
                avg_sum=_format_metric(group_summary.get("avg_sum_reward")),
                avg_max=_format_metric(group_summary.get("avg_max_reward")),
                pc=_format_metric(group_summary.get("pc_success"), digits=1),
                grasp_avg=_format_metric(group_summary.get("avg_ball_grasp_count")),
                grasp_pc=_format_metric(group_summary.get("pc_ball_grasp_success"), digits=1),
                n_ep=group_summary.get("n_episodes", "NA"),
                n_ok=group_summary.get("n_success", "NA"),
                n_grasp_ok=group_summary.get("n_ball_grasp_success", "NA"),
                n_vid=group_summary.get("n_videos", "NA"),
            )
        )
        lines.append(
            f"  {group}: ball_grasp_success_episode_indices={group_summary.get('ball_grasp_success_episode_indices', [])}"
        )

    lines.append("")
    lines.append("Per Task")
    for task_summary in summary.get("per_task", []) or []:
        lines.append(
            "  {group}/{task_id}: avg_sum_reward={avg_sum} avg_max_reward={avg_max} pc_success={pc}% "
            "avg_ball_grasp_count={grasp_avg} pc_ball_grasp_success={grasp_pc}% "
            "n_episodes={n_ep} n_success={n_ok} n_ball_grasp_success={n_grasp_ok} n_videos={n_vid}".format(
                group=task_summary.get("task_group", "NA"),
                task_id=task_summary.get("task_id", "NA"),
                avg_sum=_format_metric(task_summary.get("avg_sum_reward")),
                avg_max=_format_metric(task_summary.get("avg_max_reward")),
                pc=_format_metric(task_summary.get("pc_success"), digits=1),
                grasp_avg=_format_metric(task_summary.get("avg_ball_grasp_count")),
                grasp_pc=_format_metric(task_summary.get("pc_ball_grasp_success"), digits=1),
                n_ep=task_summary.get("n_episodes", "NA"),
                n_ok=task_summary.get("n_success", "NA"),
                n_grasp_ok=task_summary.get("n_ball_grasp_success", "NA"),
                n_vid=task_summary.get("n_videos", "NA"),
            )
        )
        lines.append(
            "  {group}/{task_id}: ball_grasp_success_episode_indices={indices}".format(
                group=task_summary.get("task_group", "NA"),
                task_id=task_summary.get("task_id", "NA"),
                indices=task_summary.get("ball_grasp_success_episode_indices", []),
            )
        )

    grasp_records = summary.get("ball_grasp_success_per_episode", []) or []
    lines.append("")
    lines.append("Ball Grasp Success Episodes")
    if not grasp_records:
        lines.append("  []")
    else:
        for record in grasp_records:
            lines.append(
                "  global_episode_ix={global_episode_ix} task={task_group}/{task_id} "
                "episode_ix_in_task={episode_ix_in_task} success={success} "
                "ball_grasp_count={ball_grasp_count} seed={seed}".format(**record)
            )

    return "\n".join(lines) + "\n"


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
    videos_dir = output_dir / "videos" / datetime.now().strftime("%Y%m%d_%H%M%S")

    if getattr(cfg.env, "type", None) == "libero" and getattr(cfg.env, "task", None) == "libero_dyn_mini":
        if hasattr(cfg.env, "task_ids") and cfg.env.task_ids is None:
            cfg.env.task_ids = [0]
        if hasattr(cfg.env, "init_states"):
            cfg.env.init_states = True
        if hasattr(cfg.env, "init_plan_path") and not getattr(cfg.env, "init_plan_path", None):
            default_plan = _resolve_dyn_mini_default_plan_path()
            if default_plan is not None:
                cfg.env.init_plan_path = str(default_plan)
                logging.info("[sync-dyn-mini] using init plan: %s", default_plan)

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")
    policy_feature_keys = []
    if cfg.policy is not None and cfg.policy.input_features:
        policy_feature_keys = list(cfg.policy.input_features.keys())
    effective_rename_map = resolve_libero_rename_map(
        enable_legacy_compat=cfg.libero_legacy_obs_compat,
        env_cfg=cfg.env,
        feature_keys=policy_feature_keys,
        user_rename_map=cfg.rename_map,
    )
    if effective_rename_map != cfg.rename_map:
        logging.warning(
            "Enabled LIBERO legacy observation compatibility mapping: %s",
            effective_rename_map,
        )

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=effective_rename_map,
    )
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": effective_rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    apply_rename_map_to_preprocessor(preprocessor, effective_rename_map)

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    info: dict[str, Any]
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=cfg.eval.n_episodes,
                videos_dir=videos_dir,
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
                rollout_fps=getattr(cfg.env, "fps", None),
            )
    finally:
        close_envs(envs)

    eval_info_path = output_dir / "eval_info.json"
    eval_info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = _build_eval_info_summary(info)
    summary_json_path = output_dir / "eval_info_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_text = _format_eval_info_summary_text(summary)
    summary_txt_path = output_dir / "eval_info_summary.txt"
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    print("Overall Aggregated Metrics:")
    print(info["overall"])
    print("\nEval Summary:")
    print(summary_text.rstrip())

    logging.info("Saved eval info to %s", eval_info_path)
    logging.info("Saved eval summary json to %s", summary_json_path)
    logging.info("Saved eval summary text to %s", summary_txt_path)
    logging.info("Saved videos under %s", videos_dir)
    logging.info("End of eval")


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sync dyn-mini eval wrapper that injects simulator ball_pos for policies that need it."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from tqdm import trange

from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.scripts import lerobot_eval_dyn_mini_sync_fullvideo as sync_eval
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import inside_slurm


def _policy_needs_ball_pos(policy: PreTrainedPolicy) -> tuple[bool, str, int]:
    policy_config = getattr(policy, "config", None)
    future_mode = str(getattr(policy_config, "model", "orig"))
    future_ball_pos_key = str(getattr(policy_config, "future_ball_pos_key", "observation.ball_pos"))
    input_features = getattr(policy_config, "input_features", None)

    needs_ball_pos = future_mode in {"scene_only", "robot_scene"}
    if not needs_ball_pos and isinstance(input_features, dict):
        needs_ball_pos = future_ball_pos_key in input_features

    ball_pos_dim = 3
    if isinstance(input_features, dict) and future_ball_pos_key in input_features:
        feature = input_features[future_ball_pos_key]
        shape = getattr(feature, "shape", None)
        if shape:
            ball_pos_dim = int(np.prod(shape))

    return needs_ball_pos, future_ball_pos_key, ball_pos_dim


def _inject_ball_pos_if_needed(
    env: gym.vector.VectorEnv,
    raw_observation: dict[str, Any],
    *,
    needs_ball_pos: bool,
    future_ball_pos_key: str,
    ball_pos_dim: int,
) -> dict[str, Any]:
    if not needs_ball_pos:
        return raw_observation

    if "ball_pos" in raw_observation or "observation.ball_pos" in raw_observation:
        return raw_observation

    if future_ball_pos_key in raw_observation:
        candidate = np.asarray(raw_observation[future_ball_pos_key], dtype=np.float32)
        if candidate.ndim == 1:
            candidate = candidate.reshape(1, -1)
        elif candidate.ndim >= 2:
            candidate = candidate.reshape(candidate.shape[0], -1)
        if candidate.shape[-1] < ball_pos_dim:
            raise RuntimeError(
                "[sync-ballpos] invalid ball position: "
                f"`{future_ball_pos_key}` last dim={candidate.shape[-1]} < expected {ball_pos_dim}."
            )
        candidate = candidate[:, :ball_pos_dim]
        if not np.isfinite(candidate).all():
            raise RuntimeError(f"[sync-ballpos] `{future_ball_pos_key}` contains NaN/Inf.")
        augmented = dict(raw_observation)
        augmented["ball_pos"] = candidate
        return augmented

    raw_ball_pos = env.call("get_ball_pos")
    if not isinstance(raw_ball_pos, (list, tuple)) or len(raw_ball_pos) == 0:
        raise RuntimeError("[sync-ballpos] env.call('get_ball_pos') returned no ball positions.")

    rows: list[np.ndarray] = []
    for env_idx, item in enumerate(raw_ball_pos):
        if item is None:
            raise RuntimeError(f"[sync-ballpos] missing ball position for env_idx={env_idx}.")
        row = np.asarray(item, dtype=np.float32).reshape(-1)
        if row.size < ball_pos_dim:
            raise RuntimeError(
                f"[sync-ballpos] invalid ball position dim for env_idx={env_idx}: "
                f"{row.size} < expected {ball_pos_dim}."
            )
        row = row[:ball_pos_dim]
        if not np.isfinite(row).all():
            raise RuntimeError(f"[sync-ballpos] ball position has NaN/Inf for env_idx={env_idx}.")
        rows.append(row.astype(np.float32, copy=False))

    if len(rows) != int(env.num_envs):
        raise RuntimeError(f"[sync-ballpos] ball_pos count mismatch: got {len(rows)}, expected {env.num_envs}.")

    augmented = dict(raw_observation)
    augmented["ball_pos"] = np.stack(rows, axis=0)
    return augmented


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Any | None = None,
    rollout_fps: float | None = None,
) -> dict[str, Any]:
    """Run the stock sync rollout, with ball_pos added before policy preprocessing."""
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    needs_ball_pos, future_ball_pos_key, ball_pos_dim = _policy_needs_ball_pos(policy)
    logging.info(
        "[sync-ballpos] needs_ball_pos=%d future_ball_pos_key=%s ball_pos_dim=%d",
        int(needs_ball_pos),
        future_ball_pos_key,
        ball_pos_dim,
    )

    policy.reset()
    observation, _ = env.reset(seed=seeds)
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
        observation = _inject_ball_pos_if_needed(
            env,
            observation,
            needs_ball_pos=needs_ball_pos,
            future_ball_pos_key=future_ball_pos_key,
            ball_pos_dim=ball_pos_dim,
        )
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        if rollout_fps is not None and rollout_fps > 0 and OBS_STATE in observation:
            batch_size = int(observation[OBS_STATE].shape[0])
            obs_state = observation[OBS_STATE]
            observation["timestamp"] = torch.full(
                (batch_size,),
                float(step) / float(rollout_fps),
                device=obs_state.device,
                dtype=obs_state.dtype,
            )

        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        action_numpy: np.ndarray = action.to("cpu").numpy()
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError("Unsupported `final_info` format: expected dict.")
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        grasp_event_step = np.zeros(env.num_envs, dtype=np.int32)
        if track_ball_grasp:
            raw_grasp = env.call("is_ball_grasped")
            if not isinstance(raw_grasp, (list, tuple)) or len(raw_grasp) != env.num_envs:
                raise RuntimeError("[sync-eval][grasp] env.call('is_ball_grasped') returned invalid shape.")
            grasp_now = np.asarray(raw_grasp, dtype=bool)
            grasp_event_step = np.logical_and(np.logical_not(grasp_prev), grasp_now).astype(np.int32)
            grasp_prev = grasp_now

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
        observation = _inject_ball_pos_if_needed(
            env,
            observation,
            needs_ball_pos=needs_ball_pos,
            future_ball_pos_key=future_ball_pos_key,
            ball_pos_dim=ball_pos_dim,
        )
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
        ret["observation"] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


sync_eval.rollout = rollout


if __name__ == "__main__":
    sync_eval.main()

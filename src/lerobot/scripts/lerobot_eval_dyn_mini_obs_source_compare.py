#!/usr/bin/env python3
"""Unified LIBERO dyn-mini eval with switchable observation source.

This script is intended for apples-to-apples comparisons between:
1. `dataset` observations used for policy inference, and
2. `online` environment observations used for policy inference.

The rollout logic is intentionally shared across both modes:
- the same policy weights are loaded
- the same action chunk is predicted each inference call
- the same `execute_n_action_steps` truncation is applied
- the same environment postprocessing is applied to actions

Only two behavioral differences are intentionally preserved:
- where the policy input observation comes from
- whether dataset-aligned episodes terminate early when the reference dataset episode is exhausted
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from contextlib import nullcontext
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
from torch import Tensor
from torch.utils.data._utils.collate import default_collate
from tqdm import trange

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.policies.utils import populate_queues
from lerobot.scripts.lerobot_eval_dyn_mini_dataset_obs import (
    DATASET_ROOT_DEFAULT,
    INIT_PLAN_DEFAULT_PATH,
    EpisodeBinding,
    ImageNoiseConfig,
    NoisyImageSaveConfig,
    apply_image_noise_to_batch,
    apply_rename_map_to_batch,
    build_dataset,
    build_processors,
    get_absolute_to_relative_index,
    get_dataset_item_by_absolute_index,
    parse_episode_selector,
    parse_int_list,
    postprocess_action_chunk,
    predict_action_chunk,
    resolve_dataset_init_plan_path,
    save_noisy_images_for_inference,
    str2bool,
    validate_episode_plan_alignment,
)
from lerobot.scripts.lerobot_eval_dyn_mini_sync_fullvideo import (
    _build_eval_info_summary,
    _format_eval_info_summary_text,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm


def build_cli_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [f"--device={args.policy_device}", f"--use_amp={'true' if args.policy_use_amp else 'false'}"]
    if args.policy_n_action_steps is not None:
        overrides.append(f"--n_action_steps={int(args.policy_n_action_steps)}")
    if args.policy_num_inference_steps is not None:
        overrides.append(f"--num_inference_steps={int(args.policy_num_inference_steps)}")
    return overrides


def _sanitize_path_token(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._")
    return cleaned or "unknown"


def infer_policy_variant_label(policy_path: str) -> str:
    path = Path(policy_path).resolve()
    parts = path.parts
    variant_label = None

    if "checkpoints" in parts:
        ckpt_ix = parts.index("checkpoints")
        if ckpt_ix >= 1:
            variant_label = _sanitize_path_token(parts[ckpt_ix - 1])
    if variant_label is None:
        variant_label = _sanitize_path_token(path.parent.name or "model")
    return variant_label


def make_output_dir(
    raw_output_dir: str | None,
    *,
    policy_path: str,
    observation_source: str,
    execute_n_action_steps: int,
) -> Path:
    if raw_output_dir:
        path = Path(raw_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_label = infer_policy_variant_label(policy_path)
        path = Path("outputs/compare") / (
            f"{policy_label}_{observation_source}_obs_exec{int(execute_n_action_steps)}_{stamp}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def zero_action_like(env: gym.vector.VectorEnv) -> np.ndarray:
    try:
        return np.zeros_like(env.action_space.sample())
    except Exception:  # noqa: BLE001
        shape = getattr(env.action_space, "shape", None)
        if shape is None:
            shape = (env.num_envs, 7)
        return np.zeros(shape, dtype=np.float32)


def prepare_online_policy_batch(
    *,
    env: gym.vector.VectorEnv,
    observation: dict[str, Any],
    env_preprocessor: Any,
    rollout_fps: float | None,
    step: int,
) -> dict[str, Any]:
    raw_batch = preprocess_observation(observation)
    raw_batch = add_envs_task(env, raw_batch)
    raw_batch = env_preprocessor(raw_batch)

    if rollout_fps is not None and rollout_fps > 0 and OBS_STATE in raw_batch:
        obs_state = raw_batch[OBS_STATE]
        batch_size = int(obs_state.shape[0])
        timestamp = torch.full(
            (batch_size,),
            float(step) / float(rollout_fps),
            device=obs_state.device,
            dtype=obs_state.dtype,
        )
        raw_batch = dict(raw_batch)
        raw_batch["timestamp"] = timestamp

    return raw_batch


def build_online_policy_chunk_input(
    *,
    policy: Any,
    raw_batch: dict[str, Any],
) -> dict[str, Any]:
    model_input = dict(raw_batch)
    if getattr(policy.config, "image_features", None) and OBS_IMAGES not in model_input:
        model_input[OBS_IMAGES] = torch.stack(
            [model_input[key] for key in policy.config.image_features],
            dim=-4,
        )

    model_input.pop(ACTION, None)
    model_input.pop(f"{ACTION}_is_pad", None)
    policy._queues = populate_queues(policy._queues, model_input)

    return {
        key: torch.stack(list(policy._queues[key]), dim=1)
        for key in model_input
        if key in policy._queues
    }


def rollout_with_observation_source(
    *,
    env: gym.vector.VectorEnv,
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    episode_bindings: list[EpisodeBinding],
    observation_source: str,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    image_noise_cfg: ImageNoiseConfig,
    noisy_image_save_cfg: NoisyImageSaveConfig,
    episode_output_indices: list[int],
    seeds: list[int] | None = None,
    render_callback: Any | None = None,
    rollout_fps: float | None = None,
) -> dict[str, Any]:
    if execute_n_action_steps <= 0:
        raise ValueError(f"execute_n_action_steps must be > 0. Got {execute_n_action_steps}.")
    if observation_source not in {"dataset", "online"}:
        raise ValueError(f"Unsupported observation_source={observation_source!r}.")

    policy.reset()
    policy.eval()
    observation, _ = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    num_envs = env.num_envs
    valid_slots = np.array([binding.dataset_episode_index >= 0 for binding in episode_bindings], dtype=bool)
    dataset_lengths = np.array(
        [max(0, binding.dataset_length) if binding.dataset_episode_index >= 0 else 0 for binding in episode_bindings],
        dtype=np.int64,
    )
    dataset_starts = np.array(
        [binding.dataset_from_index if binding.dataset_episode_index >= 0 else 0 for binding in episode_bindings],
        dtype=np.int64,
    )
    dataset_episode_indices = np.array(
        [binding.dataset_episode_index for binding in episode_bindings],
        dtype=np.int64,
    )

    current_frame_indices = np.zeros(num_envs, dtype=np.int64)
    executed_rollout_steps = np.zeros(num_envs, dtype=np.int64)
    inference_calls = np.zeros(num_envs, dtype=np.int64)
    observation_trace: list[list[int]] = [[] for _ in range(num_envs)]
    dataset_exhausted = np.logical_not(valid_slots).copy()

    all_actions: list[Tensor] = []
    all_rewards: list[Tensor] = []
    all_successes: list[Tensor] = []
    all_dones: list[Tensor] = []
    all_ball_grasp_events: list[Tensor] = []

    done = np.logical_not(valid_slots).copy()
    max_steps = env.call("_max_episode_steps")[0]
    zero_action = zero_action_like(env)
    check_env_attributes_and_types(env)

    progbar = trange(
        max_steps,
        desc=f"Running {observation_source}-observation rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )

    track_ball_grasp = False
    grasp_prev = np.zeros(num_envs, dtype=bool)
    if hasattr(env, "call"):
        try:
            raw_grasp = env.call("is_ball_grasped")
            if isinstance(raw_grasp, (list, tuple)) and len(raw_grasp) == num_envs:
                grasp_prev = np.asarray(raw_grasp, dtype=bool)
                track_ball_grasp = True
                logging.info("[%s-obs][grasp] enabled grasp event tracking.", observation_source)
        except Exception:  # noqa: BLE001
            track_ball_grasp = False

    step = 0
    while not np.all(done) and step < max_steps:
        observation_step_indices = np.zeros(num_envs, dtype=np.int64)
        inference_call_indices = np.full(num_envs, -1, dtype=np.int64)

        if observation_source == "dataset":
            batch_items: list[dict[str, Any]] = []
            for env_idx in range(num_envs):
                if not valid_slots[env_idx]:
                    abs_index = 0
                    observation_step_indices[env_idx] = 0
                else:
                    safe_frame = int(min(current_frame_indices[env_idx], max(dataset_lengths[env_idx] - 1, 0)))
                    abs_index = int(dataset_starts[env_idx] + safe_frame)
                    observation_step_indices[env_idx] = safe_frame
                    if not done[env_idx]:
                        inference_call_indices[env_idx] = inference_calls[env_idx]
                        observation_trace[env_idx].append(safe_frame)
                        inference_calls[env_idx] += 1
                batch_items.append(get_dataset_item_by_absolute_index(dataset, abs_index, absolute_to_relative))

            raw_batch = default_collate(batch_items)
            raw_batch = apply_rename_map_to_batch(raw_batch, effective_rename_map)
            raw_batch = apply_image_noise_to_batch(
                raw_batch,
                input_features=policy.config.input_features,
                noise_cfg=image_noise_cfg,
            )
            if noisy_image_save_cfg.enable:
                save_noisy_images_for_inference(
                    raw_batch,
                    input_features=policy.config.input_features,
                    save_cfg=noisy_image_save_cfg,
                    dataset_episode_indices=dataset_episode_indices,
                    episode_output_indices=episode_output_indices,
                    observed_frame_indices=observation_step_indices,
                    dataset_lengths=dataset_lengths,
                    execute_n_action_steps=execute_n_action_steps,
                    inference_call_indices=inference_call_indices,
                )
        else:
            online_raw_batch = prepare_online_policy_batch(
                env=env,
                observation=observation,
                env_preprocessor=env_preprocessor,
                rollout_fps=rollout_fps,
                step=step,
            )
            online_raw_batch = apply_image_noise_to_batch(
                online_raw_batch,
                input_features=policy.config.input_features,
                noise_cfg=image_noise_cfg,
            )
            for env_idx in range(num_envs):
                if not done[env_idx]:
                    observation_step_indices[env_idx] = step
                    inference_call_indices[env_idx] = inference_calls[env_idx]
                    observation_trace[env_idx].append(step)
                    inference_calls[env_idx] += 1
            raw_batch = online_raw_batch

        policy_batch = preprocessor(raw_batch)
        if observation_source == "online":
            policy_batch = build_online_policy_chunk_input(
                policy=policy,
                raw_batch=policy_batch,
            )

        with torch.inference_mode():
            pred_chunk_normalized = predict_action_chunk(policy, policy_batch)
        pred_chunk = postprocess_action_chunk(
            normalized_actions=pred_chunk_normalized,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        pred_chunk_cpu = pred_chunk.detach().to("cpu")
        chunk_exec_steps = min(int(execute_n_action_steps), int(pred_chunk_cpu.shape[1]))

        for chunk_step in range(chunk_exec_steps):
            action_step = pred_chunk_cpu[:, chunk_step].numpy()

            active_mask = valid_slots & np.logical_not(done)
            if not np.any(active_mask):
                break

            if observation_source == "dataset":
                can_execute = active_mask & (current_frame_indices < dataset_lengths)
            else:
                can_execute = active_mask
            action_step[~can_execute] = zero_action[~can_execute]

            observation, reward, terminated, truncated, info = env.step(action_step)
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
                successes = [False] * num_envs

            grasp_event_step = np.zeros(num_envs, dtype=np.int32)
            if track_ball_grasp:
                raw_grasp = env.call("is_ball_grasped")
                if not isinstance(raw_grasp, (list, tuple)) or len(raw_grasp) != num_envs:
                    raise RuntimeError(
                        f"[{observation_source}-obs][grasp] env.call('is_ball_grasped') returned invalid shape "
                        f"{type(raw_grasp)} len={len(raw_grasp) if isinstance(raw_grasp, (list, tuple)) else 'NA'}."
                    )
                grasp_now = np.asarray(raw_grasp, dtype=bool)
                grasp_event_step = np.logical_and(np.logical_not(grasp_prev), grasp_now).astype(np.int32)
                grasp_prev = grasp_now

            executed_rollout_steps[can_execute] += 1
            if observation_source == "dataset":
                current_frame_indices[can_execute] += 1
                newly_exhausted = can_execute & (current_frame_indices >= dataset_lengths)
                dataset_exhausted = dataset_exhausted | newly_exhausted

            done = done | terminated | truncated | dataset_exhausted
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)

            all_actions.append(torch.from_numpy(action_step))
            all_rewards.append(torch.from_numpy(reward))
            all_dones.append(torch.from_numpy(done.copy()))
            all_successes.append(torch.tensor(successes))
            all_ball_grasp_events.append(torch.from_numpy(grasp_event_step))

            step += 1
            running_success_rate = (
                einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
            )
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            progbar.update()

            if np.all(done) or step >= max_steps:
                break

    return {
        ACTION: torch.stack(all_actions, dim=1) if all_actions else torch.empty((num_envs, 0, 7)),
        "reward": torch.stack(all_rewards, dim=1) if all_rewards else torch.empty((num_envs, 0)),
        "success": torch.stack(all_successes, dim=1) if all_successes else torch.empty((num_envs, 0), dtype=torch.bool),
        "done": torch.stack(all_dones, dim=1) if all_dones else torch.empty((num_envs, 0), dtype=torch.bool),
        "ball_grasp_event": (
            torch.stack(all_ball_grasp_events, dim=1)
            if all_ball_grasp_events
            else torch.empty((num_envs, 0), dtype=torch.int32)
        ),
        "reference_dataset_episode_indices": dataset_episode_indices.tolist(),
        "reference_dataset_lengths": dataset_lengths.tolist(),
        "executed_rollout_steps": executed_rollout_steps.tolist(),
        "inference_calls": inference_calls.tolist(),
        "observation_trace": observation_trace,
        "dataset_exhausted": dataset_exhausted.tolist(),
    }


def eval_policy_unified(
    *,
    env: gym.vector.VectorEnv,
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    episode_bindings: list[EpisodeBinding],
    observation_source: str,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    image_noise_cfg: ImageNoiseConfig,
    noisy_image_save_cfg: NoisyImageSaveConfig,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    start_seed: int | None,
    rollout_fps: float | None,
) -> dict[str, Any]:
    if max_episodes_rendered > 0 and videos_dir is None:
        raise ValueError("videos_dir is required when rendering videos.")

    start = time.time()
    n_episodes = len(episode_bindings)
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    all_successes: list[bool] = []
    ball_grasp_counts: list[int] = []
    ball_grasp_successes: list[bool] = []
    all_reference_dataset_episode_indices: list[int] = []
    all_reference_dataset_lengths: list[int] = []
    all_executed_rollout_steps: list[int] = []
    all_inference_calls: list[int] = []
    all_observation_traces: list[list[int]] = []
    all_dataset_exhausted: list[bool] = []
    all_seeds: list[int | None] = []
    threads: list[threading.Thread] = []
    n_episodes_rendered = 0
    video_paths: list[str] = []

    def render_frame(active_env: gym.vector.VectorEnv):
        nonlocal n_episodes_rendered
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, active_env.num_envs)
        if isinstance(active_env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([active_env.envs[i].render() for i in range(n_to_render_now)]))
        elif isinstance(active_env, gym.vector.AsyncVectorEnv):
            ep_frames.append(np.stack(active_env.call("render")[:n_to_render_now]))

    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        batch_start = batch_ix * env.num_envs
        batch_end = min(batch_start + env.num_envs, n_episodes)
        active_count = batch_end - batch_start
        batch_bindings = list(episode_bindings[batch_start:batch_end])
        episode_output_indices = list(range(batch_start, batch_end))
        while len(batch_bindings) < env.num_envs:
            batch_bindings.append(
                EpisodeBinding(
                    dataset_episode_index=-1,
                    plan_row_index=-1,
                    dataset_from_index=0,
                    dataset_length=0,
                )
            )
            episode_output_indices.append(-1)

        if start_seed is None:
            seeds = None
        else:
            seeds = list(
                range(
                    start_seed + batch_start,
                    start_seed + batch_start + env.num_envs,
                )
            )

        rollout_data = rollout_with_observation_source(
            env=env,
            policy=policy,
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            episode_bindings=batch_bindings,
            observation_source=observation_source,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            effective_rename_map=effective_rename_map,
            execute_n_action_steps=execute_n_action_steps,
            image_noise_cfg=image_noise_cfg,
            noisy_image_save_cfg=noisy_image_save_cfg,
            episode_output_indices=episode_output_indices,
            seeds=seeds,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            rollout_fps=rollout_fps,
        )

        valid_slice = slice(0, active_count)
        done_tensor = rollout_data["done"][valid_slice]
        if done_tensor.numel() == 0:
            continue
        n_steps = done_tensor.shape[1]
        done_indices = torch.argmax(done_tensor.to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()

        reward_tensor = rollout_data["reward"][valid_slice]
        success_tensor = rollout_data["success"][valid_slice]
        grasp_tensor = rollout_data["ball_grasp_event"][valid_slice]

        batch_sum_rewards = einops.reduce((reward_tensor * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((reward_tensor * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((success_tensor * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())

        batch_grasp_counts = einops.reduce((grasp_tensor * mask), "b n -> b", "sum").to(torch.int64)
        batch_grasp_success = batch_grasp_counts > 0
        ball_grasp_counts.extend(batch_grasp_counts.tolist())
        ball_grasp_successes.extend(batch_grasp_success.tolist())

        all_reference_dataset_episode_indices.extend(rollout_data["reference_dataset_episode_indices"][0:active_count])
        all_reference_dataset_lengths.extend(rollout_data["reference_dataset_lengths"][0:active_count])
        all_executed_rollout_steps.extend(rollout_data["executed_rollout_steps"][0:active_count])
        all_inference_calls.extend(rollout_data["inference_calls"][0:active_count])
        all_observation_traces.extend(rollout_data["observation_trace"][0:active_count])
        all_dataset_exhausted.extend(rollout_data["dataset_exhausted"][0:active_count])

        if seeds is not None:
            all_seeds.extend(seeds[:active_count])
        else:
            all_seeds.extend([None] * active_count)

        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)
            for slot_ix, (stacked_frames, done_index) in enumerate(
                zip(batch_stacked_frames, done_indices.flatten().tolist(), strict=False)
            ):
                if slot_ix >= active_count or n_episodes_rendered >= max_episodes_rendered:
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

        progbar.set_postfix({"running_success_rate": f"{np.mean(all_successes) * 100:.1f}%"})

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
                "ball_grasp_count": int(ball_grasp_counts[i]),
                "ball_grasp_success": bool(ball_grasp_successes[i]),
                "reference_dataset_episode_index": int(all_reference_dataset_episode_indices[i]),
                "reference_dataset_length": int(all_reference_dataset_lengths[i]),
                "executed_rollout_steps": int(all_executed_rollout_steps[i]),
                "inference_calls": int(all_inference_calls[i]),
                "dataset_exhausted": bool(all_dataset_exhausted[i]),
                "observation_trace": list(all_observation_traces[i]),
            }
        )

    info: dict[str, Any] = {
        "per_episode": per_episode_records,
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards)),
            "avg_max_reward": float(np.nanmean(max_rewards)),
            "pc_success": float(np.nanmean(all_successes) * 100),
            "avg_ball_grasp_count": float(np.nanmean(ball_grasp_counts)),
            "pc_ball_grasp_success": float(np.nanmean(ball_grasp_successes) * 100),
            "avg_executed_rollout_steps": float(np.nanmean(all_executed_rollout_steps)),
            "avg_inference_calls": float(np.nanmean(all_inference_calls)),
            "pc_dataset_exhausted": float(np.nanmean(all_dataset_exhausted) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / max(1, n_episodes),
        },
        "video_paths": video_paths,
    }
    return info


def eval_policy_all_observation_sources(
    *,
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    all_episode_bindings: list[EpisodeBinding],
    observation_source: str,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    image_noise_cfg: ImageNoiseConfig,
    noisy_image_save_cfg: NoisyImageSaveConfig,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    start_seed: int | None,
    rollout_fps: float | None,
) -> dict[str, Any]:
    start_t = time.time()

    overall_sum_rewards: list[float] = []
    overall_max_rewards: list[float] = []
    overall_successes: list[bool] = []
    overall_ball_grasp_counts: list[int] = []
    overall_ball_grasp_successes: list[bool] = []
    overall_video_paths: list[str] = []
    overall_executed_rollout_steps: list[int] = []
    overall_inference_calls: list[int] = []
    overall_dataset_exhausted: list[bool] = []
    ball_grasp_success_per_episode: list[dict[str, Any]] = []
    per_task_infos: list[dict[str, Any]] = []
    global_episode_ix = 0
    per_group_raw: dict[str, dict[str, list[Any]]] = {}

    remaining_bindings = list(all_episode_bindings)

    for task_group, group_envs in envs.items():
        group_sum_rewards: list[float] = []
        group_max_rewards: list[float] = []
        group_successes: list[bool] = []
        group_ball_grasp_counts: list[int] = []
        group_ball_grasp_successes: list[bool] = []
        group_video_paths: list[str] = []
        group_executed_rollout_steps: list[int] = []
        group_inference_calls: list[int] = []
        group_dataset_exhausted: list[bool] = []
        group_ball_grasp_success_episode_indices: list[int] = []

        for task_id, env in group_envs.items():
            task_episode_count = min(n_episodes, len(remaining_bindings))
            task_bindings = remaining_bindings[:task_episode_count]
            remaining_bindings = remaining_bindings[task_episode_count:]

            task_videos_dir = None
            if videos_dir is not None:
                task_videos_dir = videos_dir / f"{task_group}_{task_id}"
                task_videos_dir.mkdir(parents=True, exist_ok=True)

            task_result = eval_policy_unified(
                env=env,
                policy=policy,
                dataset=dataset,
                absolute_to_relative=absolute_to_relative,
                episode_bindings=task_bindings,
                observation_source=observation_source,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                effective_rename_map=effective_rename_map,
                execute_n_action_steps=execute_n_action_steps,
                image_noise_cfg=image_noise_cfg,
                noisy_image_save_cfg=noisy_image_save_cfg,
                max_episodes_rendered=max_episodes_rendered,
                videos_dir=task_videos_dir,
                start_seed=start_seed,
                rollout_fps=rollout_fps,
            )

            per_episode = task_result["per_episode"]
            task_sum_rewards = [float(ep["sum_reward"]) for ep in per_episode]
            task_max_rewards = [float(ep["max_reward"]) for ep in per_episode]
            task_successes = [bool(ep["success"]) for ep in per_episode]
            task_ball_grasp_counts = [int(ep.get("ball_grasp_count", 0)) for ep in per_episode]
            task_ball_grasp_successes = [bool(ep.get("ball_grasp_success", False)) for ep in per_episode]
            task_video_paths = list(task_result.get("video_paths", []) or [])
            task_executed_rollout_steps = [int(ep.get("executed_rollout_steps", 0)) for ep in per_episode]
            task_inference_calls = [int(ep.get("inference_calls", 0)) for ep in per_episode]
            task_dataset_exhausted = [bool(ep.get("dataset_exhausted", False)) for ep in per_episode]
            task_ball_grasp_success_episode_indices = [
                int(ep.get("episode_ix", idx)) for idx, ep in enumerate(per_episode) if bool(ep.get("ball_grasp_success", False))
            ]

            group_sum_rewards.extend(task_sum_rewards)
            group_max_rewards.extend(task_max_rewards)
            group_successes.extend(task_successes)
            group_ball_grasp_counts.extend(task_ball_grasp_counts)
            group_ball_grasp_successes.extend(task_ball_grasp_successes)
            group_video_paths.extend(task_video_paths)
            group_executed_rollout_steps.extend(task_executed_rollout_steps)
            group_inference_calls.extend(task_inference_calls)
            group_dataset_exhausted.extend(task_dataset_exhausted)

            overall_sum_rewards.extend(task_sum_rewards)
            overall_max_rewards.extend(task_max_rewards)
            overall_successes.extend(task_successes)
            overall_ball_grasp_counts.extend(task_ball_grasp_counts)
            overall_ball_grasp_successes.extend(task_ball_grasp_successes)
            overall_video_paths.extend(task_video_paths)
            overall_executed_rollout_steps.extend(task_executed_rollout_steps)
            overall_inference_calls.extend(task_inference_calls)
            overall_dataset_exhausted.extend(task_dataset_exhausted)

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
                        "executed_rollout_steps": task_executed_rollout_steps,
                        "inference_calls": task_inference_calls,
                        "dataset_exhausted": task_dataset_exhausted,
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
            "executed_rollout_steps": group_executed_rollout_steps,
            "inference_calls": group_inference_calls,
            "dataset_exhausted": group_dataset_exhausted,
            "ball_grasp_success_episode_indices": group_ball_grasp_success_episode_indices,
            "video_paths": group_video_paths,
        }

    def _agg_from_list(xs: list[Any]) -> float:
        if not xs:
            return float("nan")
        arr = np.asarray(xs, dtype=float)
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
            "avg_executed_rollout_steps": _agg_from_list(acc["executed_rollout_steps"]),
            "avg_inference_calls": _agg_from_list(acc["inference_calls"]),
            "pc_dataset_exhausted": (
                _agg_from_list(acc["dataset_exhausted"]) * 100 if acc["dataset_exhausted"] else float("nan")
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
        "avg_executed_rollout_steps": _agg_from_list(overall_executed_rollout_steps),
        "avg_inference_calls": _agg_from_list(overall_inference_calls),
        "pc_dataset_exhausted": (
            _agg_from_list(overall_dataset_exhausted) * 100 if overall_dataset_exhausted else float("nan")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LIBERO dyn-mini with a unified rollout and switchable observation source."
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--policy.device", dest="policy_device", default="cuda")
    parser.add_argument("--policy.use_amp", dest="policy_use_amp", type=str2bool, default=True)
    parser.add_argument("--policy.n_action_steps", dest="policy_n_action_steps", type=int, default=None)
    parser.add_argument("--policy.num_inference_steps", dest="policy_num_inference_steps", type=int, default=None)
    parser.add_argument("--env.type", dest="env_type", default="libero")
    parser.add_argument("--env.task", dest="env_task", default="libero_dyn_mini")
    parser.add_argument("--env.task_ids", dest="env_task_ids", default="0")
    parser.add_argument("--env.fps", dest="env_fps", type=int, default=20)
    parser.add_argument("--env.episode_length", dest="env_episode_length", type=int, default=300)
    parser.add_argument("--env.control_mode", dest="env_control_mode", default="relative")
    parser.add_argument("--env.init_states", dest="env_init_states", type=str2bool, default=True)
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=str(INIT_PLAN_DEFAULT_PATH))
    parser.add_argument("--env.episode_start_states_path", dest="env_episode_start_states_path", default=None)
    parser.add_argument("--env.init_plan_loop", dest="env_init_plan_loop", type=str2bool, default=False)
    parser.add_argument("--env.ball_grasp_eval_mode", dest="env_ball_grasp_eval_mode", default="strict")
    parser.add_argument(
        "--env.ball_grasp_strict_require_pad_contact",
        dest="env_ball_grasp_strict_require_pad_contact",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--env.ball_grasp_strict_lift_multiplier",
        dest="env_ball_grasp_strict_lift_multiplier",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--env.ball_grasp_strict_grip_center_max_dist",
        dest="env_ball_grasp_strict_grip_center_max_dist",
        type=float,
        default=0.045,
    )
    parser.add_argument(
        "--rollout.observation_source",
        dest="rollout_observation_source",
        choices=["dataset", "online"],
        default="dataset",
    )
    parser.add_argument("--rollout.execute_n_action_steps", dest="rollout_execute_n_action_steps", type=int, default=4)
    parser.add_argument(
        "--dataset.repo_id",
        dest="dataset_repo_id",
        default="local/libero_dyn_mini_balanced500_scripted_v2",
    )
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(DATASET_ROOT_DEFAULT))
    parser.add_argument("--dataset.episodes", dest="dataset_episodes", default="0:100")
    parser.add_argument("--dataset.tolerance_s", dest="dataset_tolerance_s", type=float, default=1e-4)
    parser.add_argument("--eval.n_episodes", dest="eval_n_episodes", type=int, default=100)
    parser.add_argument("--eval.batch_size", dest="eval_batch_size", type=int, default=2)
    parser.add_argument("--dataset.image_noise.enable", dest="dataset_image_noise_enable", type=str2bool, default=False)
    parser.add_argument("--dataset.image_noise.std", dest="dataset_image_noise_std", type=float, default=0.0)
    parser.add_argument("--dataset.image_noise.clip_min", dest="dataset_image_noise_clip_min", type=float, default=0.0)
    parser.add_argument("--dataset.image_noise.clip_max", dest="dataset_image_noise_clip_max", type=float, default=1.0)
    parser.add_argument(
        "--dataset.image_noise.save_images.enable",
        dest="dataset_image_noise_save_images_enable",
        type=str2bool,
        default=False,
    )
    parser.add_argument("--output_dir", dest="output_dir", default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=1000)
    parser.add_argument("--libero_legacy_obs_compat", dest="libero_legacy_obs_compat", type=str2bool, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    init_logging()
    register_third_party_plugins()
    logging.info(pformat(vars(args)))

    if args.env_type != "libero":
        raise ValueError(f"This script only supports env.type=libero. Got {args.env_type!r}.")
    if args.env_task != "libero_dyn_mini":
        raise ValueError(f"This script is intended for env.task=libero_dyn_mini. Got {args.env_task!r}.")
    if args.eval_batch_size <= 0:
        raise ValueError("eval.batch_size must be > 0.")
    if args.eval_n_episodes <= 0:
        raise ValueError("eval.n_episodes must be > 0.")
    if args.rollout_execute_n_action_steps <= 0:
        raise ValueError("rollout.execute_n_action_steps must be > 0.")
    if args.dataset_image_noise_std < 0:
        raise ValueError("dataset.image_noise.std must be >= 0.")
    if args.dataset_image_noise_clip_min >= args.dataset_image_noise_clip_max:
        raise ValueError("dataset.image_noise.clip_min must be < dataset.image_noise.clip_max.")

    device = get_safe_torch_device(args.policy_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    image_noise_cfg = ImageNoiseConfig(
        enable=bool(args.dataset_image_noise_enable and args.dataset_image_noise_std > 0),
        std=float(args.dataset_image_noise_std),
        clip_min=float(args.dataset_image_noise_clip_min),
        clip_max=float(args.dataset_image_noise_clip_max),
    )

    policy_cfg = PreTrainedConfig.from_pretrained(
        args.policy_path,
        cli_overrides=build_cli_overrides(args),
    )
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = str(device)
    policy_cfg.use_amp = bool(args.policy_use_amp and policy_cfg.use_amp)

    env_cfg = LiberoEnvConfig(
        task=args.env_task,
        task_ids=parse_int_list(args.env_task_ids) or [0],
        fps=int(args.env_fps),
        episode_length=args.env_episode_length,
        control_mode=args.env_control_mode,
        init_states=bool(args.env_init_states),
        init_plan_path=str(resolve_dataset_init_plan_path(args.env_init_plan_path)),
        episode_start_states_path=(
            str(Path(args.env_episode_start_states_path).expanduser().resolve())
            if args.env_episode_start_states_path
            else None
        ),
        init_plan_loop=bool(args.env_init_plan_loop),
        ball_grasp_eval_mode=args.env_ball_grasp_eval_mode,
        ball_grasp_strict_require_pad_contact=bool(args.env_ball_grasp_strict_require_pad_contact),
        ball_grasp_strict_lift_multiplier=float(args.env_ball_grasp_strict_lift_multiplier),
        ball_grasp_strict_grip_center_max_dist=float(args.env_ball_grasp_strict_grip_center_max_dist),
    )

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    selected_episodes = parse_episode_selector(args.dataset_episodes)
    dataset, effective_rename_map = build_dataset(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=dataset_root,
        dataset_episodes=selected_episodes,
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
        tolerance_s=float(args.dataset_tolerance_s),
        legacy_obs_compat=bool(args.libero_legacy_obs_compat),
        rename_map={},
    )
    logging.info("Dataset loaded: %d samples across %d selected episodes", dataset.num_frames, dataset.num_episodes)
    logging.info("Effective rename map: %s", effective_rename_map)

    plan_path = resolve_dataset_init_plan_path(args.env_init_plan_path)
    plan_rows = []
    with plan_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                plan_rows.append(payload)

    if selected_episodes is None:
        selected_episodes = list(dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes))
    episode_bindings = validate_episode_plan_alignment(
        dataset=dataset,
        plan_rows=plan_rows,
        selected_episode_indices=list(selected_episodes),
        eval_n_episodes=int(args.eval_n_episodes),
    )

    output_dir = make_output_dir(
        args.output_dir,
        policy_path=args.policy_path,
        observation_source=args.rollout_observation_source,
        execute_n_action_steps=int(args.rollout_execute_n_action_steps),
    )
    videos_dir = output_dir / "videos" / datetime.now().strftime("%Y%m%d_%H%M%S")
    noisy_image_save_cfg = NoisyImageSaveConfig(
        enable=bool(
            args.rollout_observation_source == "dataset"
            and args.dataset_image_noise_save_images_enable
            and image_noise_cfg.enable
        ),
        output_dir=output_dir / "noisy_inputs",
    )
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    # Intentionally build the policy and normalizers from dataset metadata for both modes so that
    # the rollout comparison only changes observation source, not action-chunk scheduling or stats.
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=dataset.meta,
        rename_map=effective_rename_map,
    )
    policy.eval()
    preprocessor, postprocessor = build_processors(
        policy_cfg=policy_cfg,
        policy=policy,
        dataset_meta=dataset.meta,
        effective_rename_map=effective_rename_map,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    logging.info("Making environment.")
    envs = make_env(
        env_cfg,
        n_envs=int(args.eval_batch_size),
        use_async_envs=False,
        trust_remote_code=False,
    )

    absolute_to_relative = get_absolute_to_relative_index(dataset)

    info: dict[str, Any]
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext():
            info = eval_policy_all_observation_sources(
                envs=envs,
                policy=policy,
                dataset=dataset,
                absolute_to_relative=absolute_to_relative,
                all_episode_bindings=episode_bindings,
                observation_source=args.rollout_observation_source,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                effective_rename_map=effective_rename_map,
                execute_n_action_steps=int(args.rollout_execute_n_action_steps),
                image_noise_cfg=image_noise_cfg,
                noisy_image_save_cfg=noisy_image_save_cfg,
                n_episodes=int(args.eval_n_episodes),
                max_episodes_rendered=int(args.eval_n_episodes),
                videos_dir=videos_dir,
                start_seed=int(args.seed),
                rollout_fps=float(args.env_fps),
            )
    finally:
        close_envs(envs)

    info["comparison_eval"] = {
        "mode": "unified_rollout_observation_source_switch",
        "observation_source": args.rollout_observation_source,
        "policy_path": str(Path(args.policy_path).resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "selected_dataset_episodes": list(selected_episodes[: args.eval_n_episodes]),
        "init_plan_path": str(plan_path),
        "episode_start_states_path": (
            str(Path(args.env_episode_start_states_path).expanduser().resolve())
            if args.env_episode_start_states_path
            else None
        ),
        "execute_n_action_steps": int(args.rollout_execute_n_action_steps),
        "eval_n_episodes": int(args.eval_n_episodes),
        "eval_batch_size": int(args.eval_batch_size),
        "dataset_image_noise": asdict(image_noise_cfg),
        "dataset_image_noise_save_images": {
            "enable": bool(noisy_image_save_cfg.enable),
            "output_dir": str(noisy_image_save_cfg.output_dir) if noisy_image_save_cfg.output_dir else None,
        },
        "effective_rename_map": dict(effective_rename_map),
        "env": asdict(env_cfg),
        "policy_config_overrides": {
            "policy_n_action_steps": args.policy_n_action_steps,
            "policy_num_inference_steps": args.policy_num_inference_steps,
        },
    }

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


if __name__ == "__main__":
    main()

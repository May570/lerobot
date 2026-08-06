#!/usr/bin/env python3
"""Validation probes for dataset-vs-online observation effects on LIBERO dyn-mini."""

from __future__ import annotations

import argparse
import gymnasium as gym
import json
import logging
import math
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate
from tqdm import trange

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.scripts.lerobot_eval_dyn_mini_dataset_obs import (
    DATASET_ROOT_DEFAULT,
    INIT_PLAN_DEFAULT_PATH,
    EpisodeBinding,
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
    str2bool,
    validate_episode_plan_alignment,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def build_cli_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [f"--device={args.policy_device}", f"--use_amp={'true' if args.policy_use_amp else 'false'}"]
    if args.policy_n_action_steps is not None:
        overrides.append(f"--n_action_steps={int(args.policy_n_action_steps)}")
    if args.policy_num_inference_steps is not None:
        overrides.append(f"--num_inference_steps={int(args.policy_num_inference_steps)}")
    return overrides


def capture_vector_frame_batch(env: Any, n_to_render: int) -> np.ndarray:
    if n_to_render <= 0:
        raise ValueError(f"n_to_render must be > 0, got {n_to_render}.")
    if isinstance(env, gym.vector.SyncVectorEnv):
        return np.stack([env.envs[i].render() for i in range(n_to_render)], axis=0)
    if isinstance(env, gym.vector.AsyncVectorEnv):
        return np.stack(env.call("render")[:n_to_render], axis=0)
    rendered = env.render()
    if isinstance(rendered, list):
        return np.stack(rendered[:n_to_render], axis=0)
    if isinstance(rendered, np.ndarray) and rendered.ndim == 3:
        return np.expand_dims(rendered, axis=0)
    raise RuntimeError(f"Unsupported render output type for vector env: {type(rendered)!r}")


def capture_single_env_frame(single_env: Any) -> np.ndarray:
    rendered = single_env._env.render() if hasattr(single_env, "_env") else single_env.render()
    if not isinstance(rendered, np.ndarray):
        rendered = np.asarray(rendered)
    if rendered.ndim != 3:
        raise RuntimeError(f"Unsupported render output shape for single env: {rendered.shape!r}")
    return rendered


def save_video_frames(video_path: Path, frames: list[np.ndarray], fps: int) -> str | None:
    if not frames:
        return None
    video_path.parent.mkdir(parents=True, exist_ok=True)
    stacked_frames = np.stack(frames, axis=0)
    write_video(str(video_path), stacked_frames, fps=max(1, int(fps)))
    return str(video_path)


def save_batched_episode_videos(
    *,
    frame_batches: list[np.ndarray],
    executed_steps: np.ndarray,
    episode_indices: list[int],
    output_dir: Path,
    fps: int,
) -> dict[int, str]:
    if not frame_batches:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(frame_batches, axis=1)
    saved: dict[int, str] = {}
    for slot_ix, episode_ix in enumerate(episode_indices):
        n_frames = max(1, min(int(executed_steps[slot_ix]) + 1, int(stacked.shape[1])))
        path = output_dir / f"episode_{int(episode_ix):03d}.mp4"
        write_video(str(path), stacked[slot_ix, :n_frames], fps=max(1, int(fps)))
        saved[int(episode_ix)] = str(path)
    return saved


class PerEnvHistoryBuffer:
    def __init__(self, num_envs: int, maxlen: int):
        self.num_envs = int(num_envs)
        self.maxlen = int(maxlen)
        self.histories: list[dict[str, deque[Tensor]]] = [dict() for _ in range(self.num_envs)]

    def update_batched(self, frame_batch: dict[str, Any], active_mask: np.ndarray | None = None) -> None:
        if active_mask is None:
            active_mask = np.ones(self.num_envs, dtype=bool)
        for env_idx in range(self.num_envs):
            if not active_mask[env_idx]:
                continue
            env_hist = self.histories[env_idx]
            for key, value in frame_batch.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if key == ACTION or key == f"{ACTION}_is_pad":
                    continue
                if value.shape[0] != self.num_envs:
                    continue
                if key not in env_hist:
                    env_hist[key] = deque(maxlen=self.maxlen)
                queue = env_hist[key]
                sample = value[env_idx].detach().clone()
                if len(queue) == 0:
                    while len(queue) < queue.maxlen:
                        queue.append(sample.clone())
                else:
                    queue.append(sample)

    def build_batch(self) -> dict[str, Tensor]:
        all_keys = sorted({key for hist in self.histories for key in hist})
        batch: dict[str, Tensor] = {}
        for key in all_keys:
            per_env = []
            for env_idx in range(self.num_envs):
                queue = self.histories[env_idx].get(key)
                if queue is None or len(queue) == 0:
                    raise RuntimeError(f"History for env_idx={env_idx} key={key!r} is empty.")
                per_env.append(torch.stack(list(queue), dim=0))
            batch[key] = torch.stack(per_env, dim=0)
        return batch

    def clone(self) -> "PerEnvHistoryBuffer":
        copied = PerEnvHistoryBuffer(num_envs=self.num_envs, maxlen=self.maxlen)
        for env_idx in range(self.num_envs):
            env_hist = self.histories[env_idx]
            copied_hist: dict[str, deque[Tensor]] = {}
            for key, queue in env_hist.items():
                copied_queue: deque[Tensor] = deque(maxlen=self.maxlen)
                for item in queue:
                    copied_queue.append(item.detach().clone())
                copied_hist[key] = copied_queue
            copied.histories[env_idx] = copied_hist
        return copied


def flatten_l2_per_env(a: Tensor, b: Tensor) -> list[float]:
    diff = (a - b).reshape(a.shape[0], -1)
    return torch.linalg.vector_norm(diff, dim=1).detach().cpu().tolist()


def flatten_l1_mean_per_env(a: Tensor, b: Tensor) -> list[float]:
    diff = (a - b).abs().reshape(a.shape[0], -1)
    return diff.mean(dim=1).detach().cpu().tolist()


def flatten_mse_per_env(a: Tensor, b: Tensor) -> list[float]:
    diff = (a - b).pow(2).reshape(a.shape[0], -1)
    return diff.mean(dim=1).detach().cpu().tolist()


def tensor_metrics_by_env(a: Tensor, b: Tensor) -> dict[str, list[float]]:
    return {
        "l2": flatten_l2_per_env(a, b),
        "l1_mean": flatten_l1_mean_per_env(a, b),
        "mse": flatten_mse_per_env(a, b),
    }


def summarize_metric_lists(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "max": float("nan"), "min": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "max": float(np.max(arr)), "min": float(np.min(arr))}


def prepare_online_single_step_frame(
    *,
    env: Any,
    observation: dict[str, Any],
    env_preprocessor: Any,
    preprocessor: Any,
    rollout_fps: float | None,
    step: int,
) -> dict[str, Tensor]:
    raw_batch = preprocess_observation(observation)
    raw_batch = add_envs_task(env, raw_batch)
    raw_batch = env_preprocessor(raw_batch)

    if rollout_fps is not None and rollout_fps > 0 and "observation.state" in raw_batch:
        obs_state = raw_batch["observation.state"]
        batch_size = int(obs_state.shape[0])
        timestamp = torch.full(
            (batch_size,),
            float(step) / float(rollout_fps),
            device=obs_state.device,
            dtype=obs_state.dtype,
        )
        raw_batch = dict(raw_batch)
        raw_batch["timestamp"] = timestamp

    processed = preprocessor(raw_batch)
    processed = dict(processed)
    processed.pop(ACTION, None)
    processed.pop(f"{ACTION}_is_pad", None)
    return processed


def maybe_batchify_formatted_obs(formatted_obs: dict[str, Any]) -> dict[str, Any]:
    needs_batch = False
    robot_state = formatted_obs.get("robot_state")
    if isinstance(robot_state, dict):
        try:
            quat = robot_state["eef"]["quat"]
            if isinstance(quat, np.ndarray) and quat.ndim == 1:
                needs_batch = True
        except Exception:  # noqa: BLE001
            needs_batch = False
    if not needs_batch:
        pixels = formatted_obs.get("pixels")
        if isinstance(pixels, dict) and len(pixels) > 0:
            first_image = next(iter(pixels.values()))
            if isinstance(first_image, np.ndarray) and first_image.ndim == 3:
                needs_batch = True

    if not needs_batch:
        return dict(formatted_obs)

    def _add_batch(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _add_batch(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return np.expand_dims(value, axis=0)
        return value

    return _add_batch(dict(formatted_obs))


def prepare_online_single_step_frame_from_formatted_obs(
    *,
    env: Any,
    formatted_obs: dict[str, Any],
    env_preprocessor: Any,
    preprocessor: Any,
    rollout_fps: float | None,
    step: int,
) -> dict[str, Tensor]:
    raw_batch = preprocess_observation(maybe_batchify_formatted_obs(formatted_obs))
    raw_batch = add_envs_task(env, raw_batch)
    raw_batch = env_preprocessor(raw_batch)

    if rollout_fps is not None and rollout_fps > 0 and "observation.state" in raw_batch:
        obs_state = raw_batch["observation.state"]
        batch_size = int(obs_state.shape[0])
        timestamp = torch.full(
            (batch_size,),
            float(step) / float(rollout_fps),
            device=obs_state.device,
            dtype=obs_state.dtype,
        )
        raw_batch = dict(raw_batch)
        raw_batch["timestamp"] = timestamp

    processed = preprocessor(raw_batch)
    processed = dict(processed)
    processed.pop(ACTION, None)
    processed.pop(f"{ACTION}_is_pad", None)
    return processed


def prepare_dataset_policy_batch(
    *,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    effective_rename_map: dict[str, str],
    preprocessor: Any,
    episode_bindings: list[EpisodeBinding],
    frame_indices: np.ndarray,
) -> dict[str, Tensor]:
    batch_items = []
    for env_idx, binding in enumerate(episode_bindings):
        if binding.dataset_episode_index < 0:
            abs_index = 0
        else:
            safe_frame = int(min(frame_indices[env_idx], max(binding.dataset_length - 1, 0)))
            abs_index = int(binding.dataset_from_index + safe_frame)
        batch_items.append(get_dataset_item_by_absolute_index(dataset, abs_index, absolute_to_relative))

    raw_batch = default_collate(batch_items)
    raw_batch = apply_rename_map_to_batch(raw_batch, effective_rename_map)
    processed = preprocessor(raw_batch)
    processed = dict(processed)
    processed.pop(ACTION, None)
    processed.pop(f"{ACTION}_is_pad", None)
    return processed


def predict_env_action_chunk(
    *,
    policy: Any,
    policy_batch: dict[str, Tensor],
    postprocessor: Any,
    env_postprocessor: Any,
) -> Tensor:
    with torch.inference_mode():
        pred_chunk_normalized = predict_action_chunk(policy, policy_batch)
    pred_chunk = postprocess_action_chunk(
        normalized_actions=pred_chunk_normalized,
        postprocessor=postprocessor,
        env_postprocessor=env_postprocessor,
    )
    return pred_chunk.detach().to("cpu")


def compare_policy_batches_by_env(
    online_batch: dict[str, Tensor],
    dataset_batch: dict[str, Tensor],
) -> list[dict[str, Any]]:
    num_envs = next(iter(online_batch.values())).shape[0]
    per_env = [dict(obs={}) for _ in range(num_envs)]
    shared_keys = sorted(set(online_batch.keys()) & set(dataset_batch.keys()))
    for key in shared_keys:
        if not isinstance(online_batch[key], torch.Tensor) or not isinstance(dataset_batch[key], torch.Tensor):
            continue
        metrics = tensor_metrics_by_env(online_batch[key].to(torch.float32), dataset_batch[key].to(torch.float32))
        for env_idx in range(num_envs):
            per_env[env_idx]["obs"][key] = {
                "l2": float(metrics["l2"][env_idx]),
                "l1_mean": float(metrics["l1_mean"][env_idx]),
                "mse": float(metrics["mse"][env_idx]),
            }
    return per_env


def first_env_metrics(obs_metrics: list[dict[str, Any]], act_metrics: dict[str, list[float]]) -> tuple[dict[str, Any], dict[str, float]]:
    return (
        dict(obs_metrics[0]["obs"]),
        {
            "l2": float(act_metrics["l2"][0]),
            "l1_mean": float(act_metrics["l1_mean"][0]),
            "mse": float(act_metrics["mse"][0]),
        },
    )


def summarize_scalar_records(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(item[key]) for item in records if key in item and np.isfinite(float(item[key]))]
    return summarize_metric_lists(values)


def build_obs_summary(per_episode_records: list[dict[str, Any]]) -> dict[str, Any]:
    key_to_l2: dict[str, list[float]] = {}
    for record in per_episode_records:
        for key, metrics in record.get("obs", {}).items():
            key_to_l2.setdefault(key, []).append(float(metrics["l2"]))
    return {key: summarize_metric_lists(values) for key, values in key_to_l2.items()}


def make_step_record(
    *,
    episode_ix: int,
    seed: int | None,
    reference_dataset_episode_index: int,
    chunk_ix: int,
    step_ix: int,
    obs_metrics: dict[str, Any],
    action_metrics: dict[str, float],
    observation_source: str | None = None,
) -> dict[str, Any]:
    record = {
        "episode_ix": int(episode_ix),
        "seed": seed,
        "reference_dataset_episode_index": int(reference_dataset_episode_index),
        "chunk_ix": int(chunk_ix),
        "step_ix": int(step_ix),
        "obs": obs_metrics,
        "action_chunk": action_metrics,
    }
    if observation_source is not None:
        record["observation_source"] = str(observation_source)
    return record


def aggregate_chunk_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_to_obs: dict[int, list[float]] = {}
    chunk_to_act: dict[int, list[float]] = {}
    for record in records:
        chunk_ix = int(record["chunk_ix"])
        obs_l2 = float(record.get("obs", {}).get("observation.state", {}).get("l2", float("nan")))
        act_l2 = float(record.get("action_chunk", {}).get("l2", float("nan")))
        if np.isfinite(obs_l2):
            chunk_to_obs.setdefault(chunk_ix, []).append(obs_l2)
        if np.isfinite(act_l2):
            chunk_to_act.setdefault(chunk_ix, []).append(act_l2)
    summary: dict[str, Any] = {"per_chunk": {}}
    for chunk_ix in sorted(set(chunk_to_obs) | set(chunk_to_act)):
        summary["per_chunk"][str(chunk_ix)] = {
            "obs_state_l2": summarize_metric_lists(chunk_to_obs.get(chunk_ix, [])),
            "action_chunk_l2": summarize_metric_lists(chunk_to_act.get(chunk_ix, [])),
        }
    return summary


def select_observation_source(
    *,
    mode: str,
    chunk_ix: int,
    refresh_interval: int,
    online_run_chunks: int,
    dataset_run_chunks: int,
    source_before: str,
    source_after: str,
    switch_after_chunk: int,
) -> str:
    if mode == "oracle_refresh":
        if refresh_interval <= 0:
            return "online"
        return "dataset" if (chunk_ix + 1) % refresh_interval == 0 else "online"
    if mode == "periodic_teacher_obs":
        n_online = max(0, int(online_run_chunks))
        n_dataset = max(0, int(dataset_run_chunks))
        cycle = n_online + n_dataset
        if cycle <= 0:
            return "online"
        pos = int(chunk_ix) % int(cycle)
        return "online" if pos < n_online else "dataset"
    if mode == "switch":
        return source_before if chunk_ix < switch_after_chunk else source_after
    raise ValueError(f"Unsupported mode for source selection: {mode}")


def run_validation_rollout(
    *,
    mode: str,
    env: Any,
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    episode_bindings: list[EpisodeBinding],
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    batch_size: int,
    n_episodes: int,
    start_seed: int,
    rollout_fps: float | None,
    max_chunks: int,
    refresh_interval: int,
    online_run_chunks: int,
    dataset_run_chunks: int,
    source_before: str,
    source_after: str,
    switch_after_chunk: int,
    videos_dir: Path | None,
    max_videos: int,
    video_fps: int,
) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    saved_video_paths: dict[int, str] = {}
    n_batches = math.ceil(n_episodes / batch_size)
    progbar = trange(n_batches, desc=mode)

    for batch_ix in progbar:
        batch_start = batch_ix * batch_size
        batch_end = min(batch_start + batch_size, n_episodes)
        active_count = batch_end - batch_start
        batch_bindings = list(episode_bindings[batch_start:batch_end])
        while len(batch_bindings) < batch_size:
            batch_bindings.append(
                EpisodeBinding(dataset_episode_index=-1, plan_row_index=-1, dataset_from_index=0, dataset_length=0)
            )

        seeds = list(range(start_seed + batch_start, start_seed + batch_start + batch_size))
        policy.reset()
        check_env_attributes_and_types(env)
        observation, _ = env.reset(seed=seeds)
        frame_batches: list[np.ndarray] = []
        batch_episode_indices = list(range(batch_start, batch_end))
        n_video_slots_remaining = max(0, int(max_videos) - len(saved_video_paths))
        n_video_slots = min(active_count, n_video_slots_remaining)
        should_render_batch = videos_dir is not None and n_video_slots > 0
        if should_render_batch:
            frame_batches.append(capture_vector_frame_batch(env, n_video_slots))

        history = PerEnvHistoryBuffer(num_envs=batch_size, maxlen=int(policy.config.n_obs_steps))
        initial_frame = prepare_online_single_step_frame(
            env=env,
            observation=observation,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            rollout_fps=rollout_fps,
            step=0,
        )
        history.update_batched(initial_frame)

        current_frame_indices = np.zeros(batch_size, dtype=np.int64)
        done = np.array([binding.dataset_episode_index < 0 for binding in batch_bindings], dtype=bool)
        success = np.zeros(batch_size, dtype=bool)
        source_trace: list[list[str]] = [[] for _ in range(batch_size)]
        chunk_ix = 0
        step_ix = 0
        max_steps = int(env.call("_max_episode_steps")[0])

        while not np.all(done) and chunk_ix < max_chunks and step_ix < max_steps:
            online_batch = history.build_batch()
            dataset_batch = prepare_dataset_policy_batch(
                dataset=dataset,
                absolute_to_relative=absolute_to_relative,
                effective_rename_map=effective_rename_map,
                preprocessor=preprocessor,
                episode_bindings=batch_bindings,
                frame_indices=current_frame_indices.copy(),
            )

            online_actions = predict_env_action_chunk(
                policy=policy,
                policy_batch=online_batch,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
            )
            dataset_actions = predict_env_action_chunk(
                policy=policy,
                policy_batch=dataset_batch,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
            )
            obs_by_env = compare_policy_batches_by_env(online_batch, dataset_batch)
            act_metrics = tensor_metrics_by_env(online_actions.to(torch.float32), dataset_actions.to(torch.float32))

            if mode == "chunk_drift":
                selected_source = "online"
                selected_actions = online_actions
            else:
                selected_source = select_observation_source(
                    mode=mode,
                    chunk_ix=chunk_ix,
                    refresh_interval=refresh_interval,
                    online_run_chunks=online_run_chunks,
                    dataset_run_chunks=dataset_run_chunks,
                    source_before=source_before,
                    source_after=source_after,
                    switch_after_chunk=switch_after_chunk,
                )
                selected_actions = dataset_actions if selected_source == "dataset" else online_actions

            for local_idx in range(active_count):
                if done[local_idx]:
                    continue
                source_trace[local_idx].append(selected_source)
                all_records.append(
                    make_step_record(
                        episode_ix=batch_start + local_idx,
                        seed=seeds[local_idx],
                        reference_dataset_episode_index=int(batch_bindings[local_idx].dataset_episode_index),
                        chunk_ix=chunk_ix,
                        step_ix=step_ix,
                        obs_metrics=obs_by_env[local_idx]["obs"],
                        action_metrics={
                            "l2": float(act_metrics["l2"][local_idx]),
                            "l1_mean": float(act_metrics["l1_mean"][local_idx]),
                            "mse": float(act_metrics["mse"][local_idx]),
                        },
                        observation_source=None if mode == "chunk_drift" else selected_source,
                    )
                )

            chunk_exec_steps = min(int(execute_n_action_steps), int(selected_actions.shape[1]))
            for chunk_step in range(chunk_exec_steps):
                action_step = selected_actions[:, chunk_step].numpy()
                active_mask = np.logical_not(done)
                if not np.any(active_mask):
                    break

                prev_active_mask = active_mask.copy()
                current_frame_indices[prev_active_mask] += 1
                action_step[~prev_active_mask] = 0.0
                observation, _, terminated, truncated, info = env.step(action_step)
                if should_render_batch:
                    frame_batches.append(capture_vector_frame_batch(env, n_video_slots))

                if "final_info" in info and isinstance(info["final_info"], dict):
                    successes = np.asarray(info["final_info"].get("is_success", [False] * batch_size), dtype=bool)
                    success = success | successes

                done = done | terminated | truncated
                step_ix += 1

                if step_ix >= max_steps or np.all(done):
                    break

                frame = prepare_online_single_step_frame(
                    env=env,
                    observation=observation,
                    env_preprocessor=env_preprocessor,
                    preprocessor=preprocessor,
                    rollout_fps=rollout_fps,
                    step=step_ix,
                )
                history.update_batched(frame, active_mask=prev_active_mask & np.logical_not(done))

            chunk_ix += 1

        batch_video_paths: dict[int, str] = {}
        if should_render_batch and videos_dir is not None:
            batch_video_paths = save_batched_episode_videos(
                frame_batches=frame_batches,
                executed_steps=current_frame_indices[:n_video_slots].copy(),
                episode_indices=batch_episode_indices[:n_video_slots],
                output_dir=videos_dir,
                fps=video_fps,
            )
            saved_video_paths.update(batch_video_paths)

        for local_idx in range(active_count):
            episode_results.append(
                {
                    "episode_ix": batch_start + local_idx,
                    "seed": seeds[local_idx],
                    "reference_dataset_episode_index": int(batch_bindings[local_idx].dataset_episode_index),
                    "success": bool(success[local_idx]),
                    "n_chunks_executed": int(len(source_trace[local_idx])),
                    "source_trace": list(source_trace[local_idx]),
                    "video_path": batch_video_paths.get(batch_start + local_idx),
                }
            )

    summary = aggregate_chunk_records(all_records)
    summary["pc_success"] = float(np.mean([bool(item["success"]) for item in episode_results]) * 100.0) if episode_results else float("nan")
    summary["n_episodes"] = len(episode_results)
    return {
        "mode": mode,
        "per_chunk_record": all_records,
        "per_episode": episode_results,
        "summary": summary,
        "video_paths": [saved_video_paths[key] for key in sorted(saved_video_paths)],
    }


def run_first_chunk_compare(
    *,
    env: Any,
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    episode_bindings: list[EpisodeBinding],
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    batch_size: int,
    n_episodes: int,
    start_seed: int,
    rollout_fps: float | None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    n_batches = math.ceil(n_episodes / batch_size)
    progbar = trange(n_batches, desc="first_chunk_compare")
    for batch_ix in progbar:
        batch_start = batch_ix * batch_size
        batch_end = min(batch_start + batch_size, n_episodes)
        active_count = batch_end - batch_start
        batch_bindings = list(episode_bindings[batch_start:batch_end])
        while len(batch_bindings) < batch_size:
            batch_bindings.append(
                EpisodeBinding(dataset_episode_index=-1, plan_row_index=-1, dataset_from_index=0, dataset_length=0)
            )

        seeds = list(range(start_seed + batch_start, start_seed + batch_start + batch_size))
        policy.reset()
        observation, _ = env.reset(seed=seeds)
        history = PerEnvHistoryBuffer(num_envs=batch_size, maxlen=int(policy.config.n_obs_steps))
        history.update_batched(
            prepare_online_single_step_frame(
                env=env,
                observation=observation,
                env_preprocessor=env_preprocessor,
                preprocessor=preprocessor,
                rollout_fps=rollout_fps,
                step=0,
            )
        )
        online_batch = history.build_batch()
        dataset_batch = prepare_dataset_policy_batch(
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            effective_rename_map=effective_rename_map,
            preprocessor=preprocessor,
            episode_bindings=batch_bindings,
            frame_indices=np.zeros(batch_size, dtype=np.int64),
        )
        online_actions = predict_env_action_chunk(
            policy=policy,
            policy_batch=online_batch,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        dataset_actions = predict_env_action_chunk(
            policy=policy,
            policy_batch=dataset_batch,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        obs_by_env = compare_policy_batches_by_env(online_batch, dataset_batch)
        act_metrics = tensor_metrics_by_env(online_actions.to(torch.float32), dataset_actions.to(torch.float32))
        first_act_metrics = tensor_metrics_by_env(
            online_actions[:, 0].to(torch.float32),
            dataset_actions[:, 0].to(torch.float32),
        )

        for local_idx in range(active_count):
            records.append(
                {
                    "episode_ix": batch_start + local_idx,
                    "seed": seeds[local_idx],
                    "reference_dataset_episode_index": int(batch_bindings[local_idx].dataset_episode_index),
                    "obs": obs_by_env[local_idx]["obs"],
                    "action_chunk": {
                        "l2": float(act_metrics["l2"][local_idx]),
                        "l1_mean": float(act_metrics["l1_mean"][local_idx]),
                        "mse": float(act_metrics["mse"][local_idx]),
                    },
                    "first_action": {
                        "l2": float(first_act_metrics["l2"][local_idx]),
                        "l1_mean": float(first_act_metrics["l1_mean"][local_idx]),
                        "mse": float(first_act_metrics["mse"][local_idx]),
                    },
                }
            )

    return {
        "mode": "first_chunk_compare",
        "n_episodes": len(records),
        "per_episode": records,
        "obs_summary": build_obs_summary(records),
        "action_chunk_l2": summarize_metric_lists([float(item["action_chunk"]["l2"]) for item in records]),
        "first_action_l2": summarize_metric_lists([float(item["first_action"]["l2"]) for item in records]),
    }


def restore_single_env_snapshot(vec_env: Any, single_env: Any, snapshot_state: np.ndarray, seed: int | None) -> None:
    vec_env.reset(seed=[seed] if seed is not None else None)
    single_env._env.set_init_state(np.asarray(snapshot_state, dtype=np.float64).copy())


def run_single_branch_rollout(
    *,
    vec_env: Any,
    single_env: Any,
    policy: Any,
    observation_source: str,
    history: PerEnvHistoryBuffer | None,
    snapshot_state: np.ndarray,
    seed: int | None,
    chunk_start_ix: int,
    max_chunks: int,
    execute_n_action_steps: int,
    dataset: Any | None,
    absolute_to_relative: dict[int, int] | None,
    effective_rename_map: dict[str, str],
    preprocessor: Any,
    episode_binding: EpisodeBinding | None,
    dataset_frame_index: int,
    env_preprocessor: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    rollout_fps: float | None,
    video_path: Path | None,
    video_fps: int,
) -> dict[str, Any]:
    restore_single_env_snapshot(vec_env, single_env, snapshot_state=snapshot_state, seed=seed)
    branch_history = history.clone() if history is not None else None
    chunk_records: list[dict[str, Any]] = []
    success = False
    terminated = False
    executed_steps = 0
    current_step_ix = int(chunk_start_ix * execute_n_action_steps)
    current_dataset_frame_index = int(dataset_frame_index)
    branch_frames: list[np.ndarray] = []
    if video_path is not None:
        branch_frames.append(capture_single_env_frame(single_env))

    for branch_chunk_offset in range(max_chunks):
        if observation_source == "online":
            if branch_history is None:
                raise RuntimeError("Online branch requires history.")
            policy_batch = branch_history.build_batch()
        elif observation_source == "dataset":
            if dataset is None or episode_binding is None:
                raise RuntimeError("Dataset branch requires dataset and episode binding.")
            policy_batch = prepare_dataset_policy_batch(
                dataset=dataset,
                absolute_to_relative=absolute_to_relative,
                effective_rename_map=effective_rename_map,
                preprocessor=preprocessor,
                episode_bindings=[episode_binding],
                frame_indices=np.asarray([current_dataset_frame_index], dtype=np.int64),
            )
        else:
            raise ValueError(f"Unsupported observation_source={observation_source!r}")
        pred_chunk = predict_env_action_chunk(
            policy=policy,
            policy_batch=policy_batch,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        chunk_exec_steps = min(int(execute_n_action_steps), int(pred_chunk.shape[1]))
        first_action = pred_chunk[0, 0].to(torch.float32)
        chunk_records.append(
            {
                "branch_chunk_offset": int(branch_chunk_offset),
                "global_chunk_ix": int(chunk_start_ix + branch_chunk_offset),
                "first_action_norm": float(torch.linalg.vector_norm(first_action).item()),
                "chunk_action_norm": float(torch.linalg.vector_norm(pred_chunk[0].to(torch.float32).reshape(-1)).item()),
            }
        )

        for chunk_step in range(chunk_exec_steps):
            action_step = pred_chunk[0, chunk_step].numpy().astype(np.float32, copy=False)
            raw_obs, _, done_env, info = single_env._env.step(action_step)
            if video_path is not None:
                branch_frames.append(capture_single_env_frame(single_env))
            step_success = bool(single_env._env.check_success())
            success = success or step_success
            executed_steps += 1
            current_step_ix += 1
            if observation_source == "dataset":
                current_dataset_frame_index += 1

            if bool(done_env) or step_success or current_step_ix >= int(single_env._max_episode_steps):
                terminated = True
                break

            if observation_source == "online":
                formatted = single_env._format_raw_obs(raw_obs)
                frame = prepare_online_single_step_frame_from_formatted_obs(
                    env=vec_env,
                    formatted_obs=formatted,
                    env_preprocessor=env_preprocessor,
                    preprocessor=preprocessor,
                    rollout_fps=rollout_fps,
                    step=current_step_ix,
                )
                branch_history.update_batched(frame)

        if terminated:
            break

    return {
        "success": bool(success),
        "terminated": bool(terminated),
        "executed_steps": int(executed_steps),
        "chunk_records": chunk_records,
        "video_path": save_video_frames(video_path, branch_frames, video_fps) if video_path is not None else None,
    }


def run_teacher_observation_counterfactual(
    *,
    mode: str,
    vec_env: Any,
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    episode_bindings: list[EpisodeBinding],
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    n_episodes: int,
    start_seed: int,
    rollout_fps: float | None,
    probe_chunk_ix: int,
    branch_max_chunks: int,
    videos_dir: Path | None,
    max_videos: int,
    video_fps: int,
) -> dict[str, Any]:
    if vec_env.num_envs != 1:
        raise ValueError(f"{mode} requires eval.batch_size=1, got {vec_env.num_envs}.")

    single_env = vec_env.envs[0]
    records: list[dict[str, Any]] = []

    for episode_ix in trange(n_episodes, desc=mode):
        binding = episode_bindings[episode_ix]
        seed = int(start_seed + episode_ix)

        policy.reset()
        check_env_attributes_and_types(vec_env)
        observation, _ = vec_env.reset(seed=[seed])
        save_episode_video = videos_dir is not None and int(episode_ix) < int(max_videos)
        prefix_video_path = (videos_dir / f"episode_{int(episode_ix):03d}_prefix.mp4") if save_episode_video else None
        prefix_frames: list[np.ndarray] = []
        if save_episode_video:
            prefix_frames.append(capture_single_env_frame(single_env))

        history = PerEnvHistoryBuffer(num_envs=1, maxlen=int(policy.config.n_obs_steps))
        history.update_batched(
            prepare_online_single_step_frame(
                env=vec_env,
                observation=observation,
                env_preprocessor=env_preprocessor,
                preprocessor=preprocessor,
                rollout_fps=rollout_fps,
                step=0,
            )
        )

        current_frame_index = 0
        current_step_ix = 0
        snapshot_state: np.ndarray | None = None
        snapshot_history: PerEnvHistoryBuffer | None = None
        reached_probe = False
        terminal_before_probe = False

        for chunk_ix in range(probe_chunk_ix + 1):
            online_batch = history.build_batch()
            pred_chunk = predict_env_action_chunk(
                policy=policy,
                policy_batch=online_batch,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
            )
            chunk_exec_steps = min(int(execute_n_action_steps), int(pred_chunk.shape[1]))

            if chunk_ix == probe_chunk_ix:
                snapshot_state = np.asarray(single_env._env.sim.get_state().flatten().copy(), dtype=np.float64)
                snapshot_history = history.clone()
                reached_probe = True
                break

            for chunk_step in range(chunk_exec_steps):
                action_step = pred_chunk[0, chunk_step].numpy().astype(np.float32, copy=False)
                raw_obs, _, done_env, _ = single_env._env.step(action_step)
                if save_episode_video:
                    prefix_frames.append(capture_single_env_frame(single_env))
                step_success = bool(single_env._env.check_success())
                current_frame_index += 1
                current_step_ix += 1
                if bool(done_env) or step_success or current_step_ix >= int(single_env._max_episode_steps):
                    terminal_before_probe = True
                    break
                formatted = single_env._format_raw_obs(raw_obs)
                frame = prepare_online_single_step_frame_from_formatted_obs(
                    env=vec_env,
                    formatted_obs=formatted,
                    env_preprocessor=env_preprocessor,
                    preprocessor=preprocessor,
                    rollout_fps=rollout_fps,
                    step=current_step_ix,
                )
                history.update_batched(frame)
            if terminal_before_probe:
                break

        if not reached_probe or snapshot_state is None or snapshot_history is None:
            records.append(
                {
                    "episode_ix": int(episode_ix),
                    "seed": int(seed),
                    "reference_dataset_episode_index": int(binding.dataset_episode_index),
                    "reached_probe": False,
                    "terminal_before_probe": bool(terminal_before_probe),
                    "prefix_video_path": save_video_frames(prefix_video_path, prefix_frames, video_fps) if prefix_video_path is not None else None,
                }
            )
            continue

        online_history = snapshot_history.clone()

        online_batch = online_history.build_batch()
        teacher_batch = prepare_dataset_policy_batch(
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            effective_rename_map=effective_rename_map,
            preprocessor=preprocessor,
            episode_bindings=[binding],
            frame_indices=np.asarray([current_frame_index], dtype=np.int64),
        )
        online_actions = predict_env_action_chunk(
            policy=policy,
            policy_batch=online_batch,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        teacher_actions = predict_env_action_chunk(
            policy=policy,
            policy_batch=teacher_batch,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        obs_by_env = compare_policy_batches_by_env(online_batch, teacher_batch)
        act_metrics = tensor_metrics_by_env(online_actions.to(torch.float32), teacher_actions.to(torch.float32))
        first_act_metrics = tensor_metrics_by_env(
            online_actions[:, 0].to(torch.float32),
            teacher_actions[:, 0].to(torch.float32),
        )
        obs_metrics, action_chunk_metrics = first_env_metrics(obs_by_env, act_metrics)
        _, first_action_metrics = first_env_metrics(obs_by_env, first_act_metrics)

        record: dict[str, Any] = {
            "episode_ix": int(episode_ix),
            "seed": int(seed),
            "reference_dataset_episode_index": int(binding.dataset_episode_index),
            "reached_probe": True,
            "probe_chunk_ix": int(probe_chunk_ix),
            "probe_frame_index": int(current_frame_index),
            "obs": obs_metrics,
            "action_chunk": action_chunk_metrics,
            "first_action": first_action_metrics,
            "prefix_video_path": save_video_frames(prefix_video_path, prefix_frames, video_fps) if prefix_video_path is not None else None,
        }

        if mode == "teacher_obs_rescue":
            online_branch = run_single_branch_rollout(
                vec_env=vec_env,
                single_env=single_env,
                policy=policy,
                observation_source="online",
                history=online_history,
                snapshot_state=snapshot_state,
                seed=seed,
                chunk_start_ix=probe_chunk_ix,
                max_chunks=branch_max_chunks,
                execute_n_action_steps=execute_n_action_steps,
                dataset=None,
                absolute_to_relative=absolute_to_relative,
                effective_rename_map=effective_rename_map,
                preprocessor=preprocessor,
                episode_binding=None,
                dataset_frame_index=current_frame_index,
                env_preprocessor=env_preprocessor,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
                rollout_fps=rollout_fps,
                video_path=(videos_dir / f"episode_{int(episode_ix):03d}_online_branch.mp4") if save_episode_video else None,
                video_fps=video_fps,
            )
            teacher_branch = run_single_branch_rollout(
                vec_env=vec_env,
                single_env=single_env,
                policy=policy,
                observation_source="dataset",
                history=None,
                snapshot_state=snapshot_state,
                seed=seed,
                chunk_start_ix=probe_chunk_ix,
                max_chunks=branch_max_chunks,
                execute_n_action_steps=execute_n_action_steps,
                dataset=dataset,
                absolute_to_relative=absolute_to_relative,
                effective_rename_map=effective_rename_map,
                preprocessor=preprocessor,
                episode_binding=binding,
                dataset_frame_index=current_frame_index,
                env_preprocessor=env_preprocessor,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
                rollout_fps=rollout_fps,
                video_path=(videos_dir / f"episode_{int(episode_ix):03d}_teacher_branch.mp4") if save_episode_video else None,
                video_fps=video_fps,
            )
            record["online_branch"] = online_branch
            record["teacher_branch"] = teacher_branch
            record["teacher_minus_online_success"] = int(bool(teacher_branch["success"])) - int(bool(online_branch["success"]))
            record["teacher_minus_online_steps"] = int(teacher_branch["executed_steps"]) - int(online_branch["executed_steps"])

        records.append(record)

    reached_records = [item for item in records if item.get("reached_probe")]
    result: dict[str, Any] = {
        "mode": mode,
        "n_episodes": len(records),
        "per_episode": records,
        "n_reached_probe": int(len(reached_records)),
        "obs_summary": build_obs_summary(reached_records),
        "action_chunk_l2": summarize_metric_lists([float(item["action_chunk"]["l2"]) for item in reached_records]) if reached_records else summarize_metric_lists([]),
        "first_action_l2": summarize_metric_lists([float(item["first_action"]["l2"]) for item in reached_records]) if reached_records else summarize_metric_lists([]),
    }

    if mode == "teacher_obs_rescue":
        result["summary"] = {
            "online_success_rate": float(np.mean([bool(item["online_branch"]["success"]) for item in reached_records]) * 100.0) if reached_records else float("nan"),
            "teacher_success_rate": float(np.mean([bool(item["teacher_branch"]["success"]) for item in reached_records]) * 100.0) if reached_records else float("nan"),
            "teacher_minus_online_success": summarize_scalar_records(reached_records, "teacher_minus_online_success"),
            "teacher_minus_online_steps": summarize_scalar_records(reached_records, "teacher_minus_online_steps"),
        }

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observation validation probes for LIBERO dyn-mini.")
    parser.add_argument(
        "--mode",
        choices=[
            "first_chunk_compare",
            "chunk_drift",
            "oracle_refresh",
            "periodic_teacher_obs",
            "switch",
            "teacher_obs_probe",
            "teacher_obs_rescue",
        ],
        required=True,
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
    parser.add_argument("--env.ball_grasp_strict_require_pad_contact", dest="env_ball_grasp_strict_require_pad_contact", type=str2bool, default=True)
    parser.add_argument("--env.ball_grasp_strict_lift_multiplier", dest="env_ball_grasp_strict_lift_multiplier", type=float, default=1.2)
    parser.add_argument("--env.ball_grasp_strict_grip_center_max_dist", dest="env_ball_grasp_strict_grip_center_max_dist", type=float, default=0.045)
    parser.add_argument("--rollout.execute_n_action_steps", dest="rollout_execute_n_action_steps", type=int, default=4)
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="local/libero_dyn_mini_balanced500_scripted_v2")
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(DATASET_ROOT_DEFAULT))
    parser.add_argument("--dataset.episodes", dest="dataset_episodes", default="0:100")
    parser.add_argument("--dataset.tolerance_s", dest="dataset_tolerance_s", type=float, default=1e-4)
    parser.add_argument("--eval.n_episodes", dest="eval_n_episodes", type=int, default=20)
    parser.add_argument("--eval.batch_size", dest="eval_batch_size", type=int, default=2)
    parser.add_argument("--probe.max_chunks", dest="probe_max_chunks", type=int, default=10)
    parser.add_argument("--schedule.refresh_interval", dest="schedule_refresh_interval", type=int, default=2)
    parser.add_argument("--schedule.online_run_chunks", dest="schedule_online_run_chunks", type=int, default=2)
    parser.add_argument("--schedule.dataset_run_chunks", dest="schedule_dataset_run_chunks", type=int, default=1)
    parser.add_argument("--schedule.source_before", dest="schedule_source_before", choices=["online", "dataset"], default="online")
    parser.add_argument("--schedule.source_after", dest="schedule_source_after", choices=["online", "dataset"], default="dataset")
    parser.add_argument("--schedule.switch_after_chunk", dest="schedule_switch_after_chunk", type=int, default=1)
    parser.add_argument("--probe.target_chunk", dest="probe_target_chunk", type=int, default=1)
    parser.add_argument("--probe.branch_max_chunks", dest="probe_branch_max_chunks", type=int, default=6)
    parser.add_argument("--render.save_videos", dest="render_save_videos", type=str2bool, default=False)
    parser.add_argument("--render.max_episodes", dest="render_max_episodes", type=int, default=0)
    parser.add_argument("--render.video_fps", dest="render_video_fps", type=int, default=None)
    parser.add_argument("--output_dir", dest="output_dir", required=True)
    parser.add_argument("--seed", dest="seed", type=int, default=1000)
    parser.add_argument("--libero_legacy_obs_compat", dest="libero_legacy_obs_compat", type=str2bool, default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    init_logging()
    register_third_party_plugins()
    logging.info(pformat(vars(args)))

    device = get_safe_torch_device(args.policy_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

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

    plan_path = resolve_dataset_init_plan_path(args.env_init_plan_path)
    plan_rows: list[dict[str, Any]] = []
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
    envs = make_env(env_cfg, n_envs=int(args.eval_batch_size), use_async_envs=False, trust_remote_code=False)
    absolute_to_relative = get_absolute_to_relative_index(dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output dir: %s", output_dir)
    videos_dir = output_dir / "videos" if bool(args.render_save_videos) else None
    render_max_episodes = int(args.render_max_episodes) if int(args.render_max_episodes) > 0 else int(args.eval_n_episodes)
    video_fps = int(args.render_video_fps) if args.render_video_fps is not None else int(args.env_fps)

    try:
        with torch.no_grad(), torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext():
            env = next(iter(next(iter(envs.values())).values()))
            if args.mode == "first_chunk_compare":
                result = run_first_chunk_compare(
                    env=env,
                    policy=policy,
                    dataset=dataset,
                    absolute_to_relative=absolute_to_relative,
                    episode_bindings=episode_bindings,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    effective_rename_map=effective_rename_map,
                    batch_size=int(args.eval_batch_size),
                    n_episodes=int(args.eval_n_episodes),
                    start_seed=int(args.seed),
                    rollout_fps=float(args.env_fps),
                )
            elif args.mode in {"teacher_obs_probe", "teacher_obs_rescue"}:
                result = run_teacher_observation_counterfactual(
                    mode=args.mode,
                    vec_env=env,
                    policy=policy,
                    dataset=dataset,
                    absolute_to_relative=absolute_to_relative,
                    episode_bindings=episode_bindings,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    effective_rename_map=effective_rename_map,
                    execute_n_action_steps=int(args.rollout_execute_n_action_steps),
                    n_episodes=int(args.eval_n_episodes),
                    start_seed=int(args.seed),
                    rollout_fps=float(args.env_fps),
                    probe_chunk_ix=int(args.probe_target_chunk),
                    branch_max_chunks=int(args.probe_branch_max_chunks),
                    videos_dir=videos_dir,
                    max_videos=render_max_episodes,
                    video_fps=video_fps,
                )
            else:
                result = run_validation_rollout(
                    mode=args.mode,
                    env=env,
                    policy=policy,
                    dataset=dataset,
                    absolute_to_relative=absolute_to_relative,
                    episode_bindings=episode_bindings,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    effective_rename_map=effective_rename_map,
                    execute_n_action_steps=int(args.rollout_execute_n_action_steps),
                    batch_size=int(args.eval_batch_size),
                    n_episodes=int(args.eval_n_episodes),
                    start_seed=int(args.seed),
                    rollout_fps=float(args.env_fps),
                    max_chunks=int(args.probe_max_chunks),
                    refresh_interval=int(args.schedule_refresh_interval),
                    online_run_chunks=int(args.schedule_online_run_chunks),
                    dataset_run_chunks=int(args.schedule_dataset_run_chunks),
                    source_before=str(args.schedule_source_before),
                    source_after=str(args.schedule_source_after),
                    switch_after_chunk=int(args.schedule_switch_after_chunk),
                    videos_dir=videos_dir,
                    max_videos=render_max_episodes,
                    video_fps=video_fps,
                )
    finally:
        close_envs(envs)

    result["config"] = {
        "mode": args.mode,
        "policy_path": str(Path(args.policy_path).resolve()),
        "dataset_root": str(dataset_root),
        "dataset_episodes": args.dataset_episodes,
        "eval_n_episodes": int(args.eval_n_episodes),
        "eval_batch_size": int(args.eval_batch_size),
        "execute_n_action_steps": int(args.rollout_execute_n_action_steps),
        "probe_target_chunk": int(args.probe_target_chunk),
        "probe_branch_max_chunks": int(args.probe_branch_max_chunks),
        "schedule_online_run_chunks": int(args.schedule_online_run_chunks),
        "schedule_dataset_run_chunks": int(args.schedule_dataset_run_chunks),
        "render_save_videos": bool(args.render_save_videos),
        "render_max_episodes": int(render_max_episodes),
        "render_video_fps": int(video_fps),
        "seed": int(args.seed),
        "env": asdict(env_cfg),
    }

    out_path = output_dir / "probe_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved result to %s", out_path)


if __name__ == "__main__":
    main()

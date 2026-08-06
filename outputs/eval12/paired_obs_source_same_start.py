#!/usr/bin/env python3
"""Paired LIBERO dyn-mini eval with shared initial state and shared first inference input.

This script runs paired rollouts where:
1. both lanes start from the exact same cached environment start state,
2. both lanes use the exact same first policy input batch,
3. after the first action chunk, one lane continues with online env observations,
4. the other lane continues with offline dataset observations.

The intent is a cleaner causal comparison than separate eval runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Any

import einops
import gymnasium as gym
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from termcolor import colored
from torch import Tensor
from torch.utils.data._utils.collate import default_collate
from tqdm import trange

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LIBERO_DYN_MINI_ROOT = ROOT.parent / "LIBERO" / "libero_dyn_mini"
LEROBOT_SRC_ROOT = ROOT / "src"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_DYN_MINI_ROOT / "config"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("HF_HOME", "/tmp/hf-home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf-hub")
sys.path.insert(0, str(LIBERO_DYN_MINI_ROOT / "py"))
sys.path.insert(0, str(LEROBOT_SRC_ROOT))

import libero_dyn_mini_v1  # noqa: F401

from libero.libero import benchmark

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.policies.utils import populate_queues
from lerobot.scripts.lerobot_eval_dyn_mini_dataset_obs import (
    DATASET_ROOT_DEFAULT,
    INIT_PLAN_DEFAULT_PATH,
    EpisodeBinding,
    ImageNoiseConfig,
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
from lerobot.scripts.lerobot_eval_dyn_mini_obs_source_compare import build_cli_overrides, infer_policy_variant_label, prepare_online_policy_batch
from lerobot.scripts.lerobot_eval_dyn_mini_sync_fullvideo import (
    _build_eval_info_summary,
    _format_eval_info_summary_text,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm


def make_output_dir(raw_output_dir: str | None, *, policy_path: str) -> Path:
    if raw_output_dir:
        path = Path(raw_output_dir)
    else:
        path = Path("outputs/eval12/run2")
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_paired_env(
    *,
    env_cfg: LiberoEnvConfig,
    task_id: int,
) -> gym.vector.SyncVectorEnv:
    suite = benchmark.get_benchmark_dict()[env_cfg.task]()
    camera_name = env_cfg.camera_name
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)

    def _make_one() -> LiberoEnv:
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=env_cfg.task,
            camera_name=camera_name,
            init_states=env_cfg.init_states,
            episode_length=env_cfg.episode_length,
            episode_index=0,
            n_envs=1,
            control_mode=env_cfg.control_mode,
            **gym_kwargs,
        )

    return gym.vector.SyncVectorEnv([_make_one, _make_one], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)


def zero_action_like(env: gym.vector.VectorEnv) -> np.ndarray:
    try:
        return np.zeros_like(env.action_space.sample())
    except Exception:  # noqa: BLE001
        return np.zeros((env.num_envs, 7), dtype=np.float32)


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = value
    return cloned


def slice_env_batch(batch: dict[str, Any], env_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[env_idx : env_idx + 1].detach().clone()
        else:
            out[key] = value
    return out


def stack_two_batches(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in left:
        left_value = left[key]
        if key not in right:
            continue
        right_value = right[key]
        if isinstance(left_value, torch.Tensor) and isinstance(right_value, torch.Tensor):
            out[key] = torch.cat([left_value, right_value], dim=0)
    return out


def get_dataset_policy_batch(
    *,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    binding: EpisodeBinding,
    effective_rename_map: dict[str, str],
    preprocessor: Any,
) -> dict[str, Any]:
    abs_index = int(binding.dataset_from_index)
    raw_item = get_dataset_item_by_absolute_index(dataset, abs_index, absolute_to_relative)
    raw_batch = default_collate([raw_item])
    if effective_rename_map:
        renamed: dict[str, Any] = {}
        for key, value in raw_batch.items():
            renamed[effective_rename_map.get(key, key)] = value
        raw_batch = renamed
    policy_batch = preprocessor(raw_batch)
    policy_batch = dict(policy_batch)
    policy_batch.pop(ACTION, None)
    policy_batch.pop(f"{ACTION}_is_pad", None)
    return clone_batch(policy_batch)


def filter_policy_input_batch(
    *,
    policy: Any,
    raw_batch: dict[str, Any],
) -> dict[str, Any]:
    model_input = dict(raw_batch)
    if getattr(policy.config, "image_features", None) and "observation.images" not in model_input:
        image_keys = [key for key in policy.config.image_features if key in model_input]
        if image_keys:
            model_input["observation.images"] = torch.stack([model_input[key] for key in image_keys], dim=-4)
    model_input.pop(ACTION, None)
    model_input.pop(f"{ACTION}_is_pad", None)
    allowed_keys = set(getattr(policy, "_queues", {}).keys()) or set(getattr(policy.config, "input_features", {}).keys())
    return {key: value for key, value in model_input.items() if key in allowed_keys or key == "timestamp"}


def snapshot_torch_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_state


def restore_torch_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def predict_chunk_from_batch(
    *,
    policy: Any,
    batch: dict[str, Any],
    seed: int,
    postprocessor: Any,
    env_postprocessor: Any,
) -> Tensor:
    saved_rng = snapshot_torch_rng()
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        with torch.inference_mode():
            pred_chunk_normalized = predict_action_chunk(policy, batch)
        pred_chunk = postprocess_action_chunk(
            normalized_actions=pred_chunk_normalized,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        return pred_chunk.detach().to("cpu")
    finally:
        restore_torch_rng(saved_rng)


def build_online_lane_batch_from_frame(
    *,
    policy: Any,
    raw_frame_batch: dict[str, Any],
    lane_queues: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_input = dict(raw_frame_batch)
    if getattr(policy.config, "image_features", None) and OBS_IMAGES not in model_input:
        image_keys = [key for key in policy.config.image_features if key in model_input]
        if image_keys:
            model_input[OBS_IMAGES] = torch.stack([model_input[key] for key in image_keys], dim=-4)
    model_input.pop(ACTION, None)
    model_input.pop(f"{ACTION}_is_pad", None)

    updated_queues = populate_queues(deepcopy(lane_queues), model_input)
    stacked_batch = {
        key: torch.stack(list(updated_queues[key]), dim=1)
        for key in model_input
        if key in updated_queues and isinstance(model_input[key], torch.Tensor)
    }
    return stacked_batch, updated_queues


def seed_online_lane_queues_from_stacked_batch(
    *,
    stacked_batch: dict[str, Any],
    lane_queues: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(lane_queues)
    for key, queue in updated.items():
        if key == ACTION or key not in stacked_batch:
            continue
        value = stacked_batch[key]
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            continue
        queue.clear()
        for t in range(value.shape[1]):
            queue.append(value[:, t].detach().clone())
    return updated


def compute_tensor_rmse(left: Tensor | None, right: Tensor | None) -> float | None:
    if left is None or right is None:
        return None
    left_cpu = left.detach().to("cpu", dtype=torch.float32)
    right_cpu = right.detach().to("cpu", dtype=torch.float32)
    if left_cpu.shape != right_cpu.shape:
        return None
    return float(torch.sqrt(torch.mean((left_cpu - right_cpu) ** 2)).item())


def nested_get(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def resolve_image_tensor(batch: dict[str, Any], direct_key: str, image_index: int) -> Tensor | None:
    direct = batch.get(direct_key)
    if isinstance(direct, torch.Tensor):
        return direct
    stacked = batch.get(OBS_IMAGES)
    if not isinstance(stacked, torch.Tensor):
        return None
    camera_dim = stacked.ndim - 4
    if camera_dim < 0 or stacked.shape[camera_dim] <= image_index:
        return None
    return stacked.select(dim=camera_dim, index=image_index)


def compress_state_tensor(state_tensor: Tensor | None) -> Tensor | None:
    if state_tensor is None:
        return None
    state = state_tensor.detach().to(torch.float32)
    if state.shape[-1] < 8:
        return state
    pos = state[..., :3]
    angle = state[..., 3:6]
    gripper_aperture = 0.5 * (state[..., 6:7] - state[..., 7:8])
    return torch.cat([pos, angle, gripper_aperture], dim=-1)


def compute_state_component_metrics(left_state: Tensor | None, right_state: Tensor | None) -> dict[str, float | None]:
    left = compress_state_tensor(left_state)
    right = compress_state_tensor(right_state)
    if left is None or right is None or left.shape != right.shape or left.shape[-1] < 7:
        return {"pos": None, "angle": None, "gripper": None, "total": None}
    return {
        "pos": compute_tensor_rmse(left[..., :3], right[..., :3]),
        "angle": compute_tensor_rmse(left[..., 3:6], right[..., 3:6]),
        "gripper": compute_tensor_rmse(left[..., 6:7], right[..., 6:7]),
        "total": compute_tensor_rmse(left, right),
    }


def action_gripper_binary(action_tensor: Tensor | None) -> Tensor | None:
    if action_tensor is None:
        return None
    action = action_tensor.detach().to(torch.float32)
    return (action[..., 6:7] > 0).to(torch.float32)


def compute_action_component_metrics(left_action: Tensor | None, right_action: Tensor | None) -> dict[str, float | None]:
    if left_action is None or right_action is None:
        return {"pos": None, "angle": None, "gripper": None, "total": None}
    left = left_action.detach().to(torch.float32)
    right = right_action.detach().to(torch.float32)
    if left.shape != right.shape or left.shape[-1] < 7:
        return {"pos": None, "angle": None, "gripper": None, "total": None}
    left_total = torch.cat([left[..., :6], action_gripper_binary(left)], dim=-1)
    right_total = torch.cat([right[..., :6], action_gripper_binary(right)], dim=-1)
    return {
        "pos": compute_tensor_rmse(left[..., :3], right[..., :3]),
        "angle": compute_tensor_rmse(left[..., 3:6], right[..., 3:6]),
        "gripper": compute_tensor_rmse(action_gripper_binary(left), action_gripper_binary(right)),
        "total": compute_tensor_rmse(left_total, right_total),
    }


def compute_input_diff_metrics(
    dataset_batch: dict[str, Any],
    online_batch: dict[str, Any],
) -> dict[str, float | None]:
    state_components = compute_state_component_metrics(
        dataset_batch.get("observation.state"),
        online_batch.get("observation.state"),
    )
    return {
        "img1": compute_tensor_rmse(
            resolve_image_tensor(dataset_batch, "observation.images.image", 0),
            resolve_image_tensor(online_batch, "observation.images.image", 0),
        ),
        "img2": compute_tensor_rmse(
            resolve_image_tensor(dataset_batch, "observation.images.image2", 1),
            resolve_image_tensor(online_batch, "observation.images.image2", 1),
        ),
        "state": state_components["total"],
        "state_pos": state_components["pos"],
        "state_angle": state_components["angle"],
        "state_gripper": state_components["gripper"],
    }


def compute_observation_frame_diff_metrics(
    left_batch: dict[str, Any],
    right_batch: dict[str, Any],
) -> dict[str, float | None]:
    state_components = compute_state_component_metrics(
        left_batch.get("observation.state"),
        right_batch.get("observation.state"),
    )
    return {
        "img1": compute_tensor_rmse(
            resolve_image_tensor(left_batch, "observation.images.image", 0),
            resolve_image_tensor(right_batch, "observation.images.image", 0),
        ),
        "img2": compute_tensor_rmse(
            resolve_image_tensor(left_batch, "observation.images.image2", 1),
            resolve_image_tensor(right_batch, "observation.images.image2", 1),
        ),
        "state": state_components["total"],
        "state_pos": state_components["pos"],
        "state_angle": state_components["angle"],
        "state_gripper": state_components["gripper"],
    }


def extract_latest_observation_frame(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["observation.state", "observation.images.image", "observation.images.image2"]:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if key == "observation.state" and value.ndim >= 3:
            out[key] = value[:, -1].detach().clone()
        elif key.startswith("observation.images.") and value.ndim >= 5:
            out[key] = value[:, -1].detach().clone()
        else:
            out[key] = value.detach().clone()
    return out


def build_demo_latest_observation_frame(
    *,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    binding: EpisodeBinding,
    dataset_frame_idx: int,
    effective_rename_map: dict[str, str],
    preprocessor: Any,
) -> tuple[dict[str, Any], int]:
    safe_frame = int(min(dataset_frame_idx, max(binding.dataset_length - 1, 0)))
    dataset_item = get_dataset_item_by_absolute_index(
        dataset,
        int(binding.dataset_from_index + safe_frame),
        absolute_to_relative,
    )
    dataset_raw_batch = default_collate([dataset_item])
    if effective_rename_map:
        renamed: dict[str, Any] = {}
        for key, value in dataset_raw_batch.items():
            renamed[effective_rename_map.get(key, key)] = value
        dataset_raw_batch = renamed
    dataset_processed = preprocessor(dataset_raw_batch)
    return extract_latest_observation_frame(dataset_processed), safe_frame


def compute_next_obs_diff_bundle(
    *,
    env: gym.vector.VectorEnv,
    observation: dict[str, Any],
    env_preprocessor: Any,
    preprocessor: Any,
    rollout_fps: float | None,
    step: int,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    binding: EpisodeBinding,
    dataset_frame_idx: int,
    effective_rename_map: dict[str, str],
    dataset_lane_idx: int,
    online_lane_idx: int,
) -> tuple[dict[str, Any], int]:
    online_raw_batch = prepare_online_policy_batch(
        env=env,
        observation=observation,
        env_preprocessor=env_preprocessor,
        rollout_fps=rollout_fps,
        step=step,
    )
    online_processed = preprocessor(online_raw_batch)
    dataset_env_frame = extract_latest_observation_frame(slice_env_batch(online_processed, dataset_lane_idx))
    online_env_frame = extract_latest_observation_frame(slice_env_batch(online_processed, online_lane_idx))
    demo_frame, safe_frame = build_demo_latest_observation_frame(
        dataset=dataset,
        absolute_to_relative=absolute_to_relative,
        binding=binding,
        dataset_frame_idx=dataset_frame_idx,
        effective_rename_map=effective_rename_map,
        preprocessor=preprocessor,
    )
    return (
        {
            "dataset_env_vs_demo": compute_observation_frame_diff_metrics(dataset_env_frame, demo_frame),
            "online_env_vs_dataset_env": compute_observation_frame_diff_metrics(online_env_frame, dataset_env_frame),
            "online_env_vs_demo": compute_observation_frame_diff_metrics(online_env_frame, demo_frame),
        },
        safe_frame,
    )


def compute_output_diff_metrics(
    *,
    dataset_chunk: Tensor,
    online_chunk: Tensor,
    demo_chunk: Tensor | None,
    execute_n_action_steps: int,
    actual_executed_steps: int,
) -> dict[str, float | int | None]:
    compared_steps = min(
        int(execute_n_action_steps),
        int(actual_executed_steps),
        int(dataset_chunk.shape[1]),
        int(online_chunk.shape[1]),
    )
    if demo_chunk is not None:
        compared_steps = min(compared_steps, int(demo_chunk.shape[1]))
    if compared_steps <= 0:
        return {
            "dataset_vs_online": None,
            "dataset_vs_demo": None,
            "online_vs_demo": None,
            "dataset_vs_online_pos": None,
            "dataset_vs_online_angle": None,
            "dataset_vs_online_gripper": None,
            "dataset_vs_demo_pos": None,
            "dataset_vs_demo_angle": None,
            "dataset_vs_demo_gripper": None,
            "online_vs_demo_pos": None,
            "online_vs_demo_angle": None,
            "online_vs_demo_gripper": None,
            "compared_action_steps": 0,
        }
    dataset_vs_online_components = compute_action_component_metrics(
        dataset_chunk[:, :compared_steps],
        online_chunk[:, :compared_steps],
    )
    dataset_vs_demo_components = compute_action_component_metrics(
        dataset_chunk[:, :compared_steps],
        demo_chunk[:, :compared_steps] if demo_chunk is not None else None,
    )
    online_vs_demo_components = compute_action_component_metrics(
        online_chunk[:, :compared_steps],
        demo_chunk[:, :compared_steps] if demo_chunk is not None else None,
    )
    return {
        "dataset_vs_online": dataset_vs_online_components["total"],
        "dataset_vs_demo": dataset_vs_demo_components["total"],
        "online_vs_demo": online_vs_demo_components["total"],
        "dataset_vs_online_pos": dataset_vs_online_components["pos"],
        "dataset_vs_online_angle": dataset_vs_online_components["angle"],
        "dataset_vs_online_gripper": dataset_vs_online_components["gripper"],
        "dataset_vs_demo_pos": dataset_vs_demo_components["pos"],
        "dataset_vs_demo_angle": dataset_vs_demo_components["angle"],
        "dataset_vs_demo_gripper": dataset_vs_demo_components["gripper"],
        "online_vs_demo_pos": online_vs_demo_components["pos"],
        "online_vs_demo_angle": online_vs_demo_components["angle"],
        "online_vs_demo_gripper": online_vs_demo_components["gripper"],
        "compared_action_steps": int(compared_steps),
    }


def extract_demo_action_chunk(
    *,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    binding: EpisodeBinding,
    dataset_frame_idx: int,
    execute_n_action_steps: int,
    effective_rename_map: dict[str, str],
) -> Tensor | None:
    abs_index = int(binding.dataset_from_index + int(dataset_frame_idx))
    raw_item = get_dataset_item_by_absolute_index(dataset, abs_index, absolute_to_relative)
    action_key = effective_rename_map.get(ACTION, ACTION)
    raw_action = raw_item.get(action_key, raw_item.get(ACTION))
    if raw_action is None:
        return None
    action_tensor = torch.as_tensor(raw_action, dtype=torch.float32)
    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(0)

    action_is_pad_key = effective_rename_map.get(f"{ACTION}_is_pad", f"{ACTION}_is_pad")
    raw_action_is_pad = raw_item.get(action_is_pad_key, raw_item.get(f"{ACTION}_is_pad"))
    if raw_action_is_pad is not None:
        pad_tensor = torch.as_tensor(raw_action_is_pad, dtype=torch.bool).reshape(-1)
        if pad_tensor.numel() >= action_tensor.shape[0]:
            valid_steps = int((~pad_tensor[: action_tensor.shape[0]]).sum().item())
            action_tensor = action_tensor[:valid_steps]

    action_tensor = action_tensor[: int(execute_n_action_steps)]
    if action_tensor.ndim == 2:
        action_tensor = action_tensor.unsqueeze(0)
    return action_tensor.detach().to("cpu")


def draw_text_with_outline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))


def annotate_pair_frame(frame_pair: np.ndarray, overlay: dict[str, Any] | None) -> np.ndarray:
    left = Image.fromarray(frame_pair[0])
    right = Image.fromarray(frame_pair[1])

    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)

    if overlay is None:
        left_text = "dataset"
        right_text = "online"
    else:
        inference_ix = int(overlay.get("inference_ix", 0))
        action_ix = int(overlay.get("action_step_in_chunk", 0))
        action_total = int(max(overlay.get("chunk_exec_steps", 0), 0))
        left_status = " done" if bool(overlay.get("dataset_done", False)) else ""
        right_status = " done" if bool(overlay.get("online_done", False)) else ""
        left_text = f"dataset inf={inference_ix} act={action_ix}/{action_total}{left_status}"
        right_text = f"online inf={inference_ix} act={action_ix}/{action_total}{right_status}"

    draw_text_with_outline(left_draw, (8, 8), left_text)
    draw_text_with_outline(right_draw, (8, 8), right_text)
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=1)


def _plot_series(ax: Any, records: list[dict[str, Any]], field_path: tuple[str, str], label: str) -> None:
    xs: list[int] = []
    ys: list[float] = []
    top_key, inner_key = field_path
    for record in records:
        value = record.get(top_key, {}).get(inner_key)
        if value is None:
            continue
        xs.append(int(record["inference_ix"]))
        ys.append(float(value))
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=label)


def _plot_nested_series(ax: Any, records: list[dict[str, Any]], field_path: tuple[str, ...], label: str) -> None:
    xs: list[int] = []
    ys: list[float] = []
    for record in records:
        value = nested_get(record, field_path)
        if value is None:
            continue
        xs.append(int(record["inference_ix"]))
        ys.append(float(value))
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=label)


def _format_diff_axis(ax: Any, *, title: str, max_inference_count: int, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if max_inference_count <= 1:
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(0, max_inference_count - 1)
    ax.grid(True, alpha=0.35)
    if ax.lines:
        ax.legend()


def write_pair_combined_diff_plot(
    *,
    records: list[dict[str, Any]],
    output_path: Path,
    max_inference_count: int,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    input_ax, output_ax = axes

    _plot_series(input_ax, records, ("input_diff", "img1"), "img1")
    _plot_series(input_ax, records, ("input_diff", "img2"), "img2")
    _plot_series(input_ax, records, ("input_diff", "state"), "state")
    _format_diff_axis(
        input_ax,
        title="Input RMSE per Inference",
        max_inference_count=max_inference_count,
        ylabel="Input RMSE",
    )

    _plot_series(output_ax, records, ("output_diff", "dataset_vs_online"), "dataset_vs_online")
    _plot_series(output_ax, records, ("output_diff", "dataset_vs_demo"), "dataset_vs_demo")
    _plot_series(output_ax, records, ("output_diff", "online_vs_demo"), "online_vs_demo")
    _format_diff_axis(
        output_ax,
        title="Output Action RMSE per Inference",
        max_inference_count=max_inference_count,
        ylabel="Output RMSE",
    )
    output_ax.set_xlabel("Inference index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def write_pair_component_diff_plot(
    *,
    records: list[dict[str, Any]],
    output_path: Path,
    max_inference_count: int,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(8, 14), sharex=True)
    state_ax, action_ds_on_ax, action_ds_demo_ax, action_on_demo_ax = axes

    _plot_nested_series(state_ax, records, ("input_diff", "state_pos"), "state_pos")
    _plot_nested_series(state_ax, records, ("input_diff", "state_angle"), "state_angle")
    _plot_nested_series(state_ax, records, ("input_diff", "state_gripper"), "state_gripper")
    _format_diff_axis(
        state_ax,
        title="State Component RMSE per Inference",
        max_inference_count=max_inference_count,
        ylabel="State RMSE",
    )

    _plot_nested_series(action_ds_on_ax, records, ("output_diff", "dataset_vs_online_pos"), "pos")
    _plot_nested_series(action_ds_on_ax, records, ("output_diff", "dataset_vs_online_angle"), "angle")
    _plot_nested_series(action_ds_on_ax, records, ("output_diff", "dataset_vs_online_gripper"), "gripper")
    _format_diff_axis(
        action_ds_on_ax,
        title="Action Component RMSE: dataset vs online",
        max_inference_count=max_inference_count,
        ylabel="Action RMSE",
    )

    _plot_nested_series(action_ds_demo_ax, records, ("output_diff", "dataset_vs_demo_pos"), "pos")
    _plot_nested_series(action_ds_demo_ax, records, ("output_diff", "dataset_vs_demo_angle"), "angle")
    _plot_nested_series(action_ds_demo_ax, records, ("output_diff", "dataset_vs_demo_gripper"), "gripper")
    _format_diff_axis(
        action_ds_demo_ax,
        title="Action Component RMSE: dataset vs demo",
        max_inference_count=max_inference_count,
        ylabel="Action RMSE",
    )

    _plot_nested_series(action_on_demo_ax, records, ("output_diff", "online_vs_demo_pos"), "pos")
    _plot_nested_series(action_on_demo_ax, records, ("output_diff", "online_vs_demo_angle"), "angle")
    _plot_nested_series(action_on_demo_ax, records, ("output_diff", "online_vs_demo_gripper"), "gripper")
    _format_diff_axis(
        action_on_demo_ax,
        title="Action Component RMSE: online vs demo",
        max_inference_count=max_inference_count,
        ylabel="Action RMSE",
    )
    action_on_demo_ax.set_xlabel("Inference index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def write_pair_next_obs_overview_plot(
    *,
    records: list[dict[str, Any]],
    output_path: Path,
    max_inference_count: int,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    configs = [
        ("dataset_env_vs_demo", "Next obs: dataset-env vs demo"),
        ("online_env_vs_dataset_env", "Next obs: online-env vs dataset-env"),
        ("online_env_vs_demo", "Next obs: online-env vs demo"),
    ]
    for ax, (prefix, title) in zip(axes, configs, strict=True):
        _plot_nested_series(ax, records, ("next_obs_diff", prefix, "img1"), "img1")
        _plot_nested_series(ax, records, ("next_obs_diff", prefix, "img2"), "img2")
        _plot_nested_series(ax, records, ("next_obs_diff", prefix, "state"), "state")
        _format_diff_axis(ax, title=title, max_inference_count=max_inference_count, ylabel="Obs RMSE")
    axes[-1].set_xlabel("Inference index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def write_pair_next_obs_state_component_plot(
    *,
    records: list[dict[str, Any]],
    output_path: Path,
    max_inference_count: int,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    configs = [
        ("dataset_env_vs_demo", "Next state: dataset-env vs demo"),
        ("online_env_vs_dataset_env", "Next state: online-env vs dataset-env"),
        ("online_env_vs_demo", "Next state: online-env vs demo"),
    ]
    for ax, (prefix, title) in zip(axes, configs, strict=True):
        _plot_nested_series(ax, records, ("next_obs_diff", prefix, "state_pos"), "state_pos")
        _plot_nested_series(ax, records, ("next_obs_diff", prefix, "state_angle"), "state_angle")
        _plot_nested_series(ax, records, ("next_obs_diff", prefix, "state_gripper"), "state_gripper")
        _format_diff_axis(ax, title=title, max_inference_count=max_inference_count, ylabel="State RMSE")
    axes[-1].set_xlabel("Inference index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def write_pair_numeric_records(
    *,
    pair_ix: int,
    records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"pair_{pair_ix:04d}_diff_records.jsonl"
    csv_path = output_dir / f"pair_{pair_ix:04d}_diff_records.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = [
        "inference_ix",
        "dataset_frame_idx",
        "online_env_step_before_inference",
        "next_obs_demo_frame_idx",
        "input_img1_rmse",
        "input_img2_rmse",
        "input_state_rmse",
        "input_state_pos_rmse",
        "input_state_angle_rmse",
        "input_state_gripper_rmse",
        "output_dataset_vs_online_rmse",
        "output_dataset_vs_online_pos_rmse",
        "output_dataset_vs_online_angle_rmse",
        "output_dataset_vs_online_gripper_rmse",
        "output_dataset_vs_demo_rmse",
        "output_dataset_vs_demo_pos_rmse",
        "output_dataset_vs_demo_angle_rmse",
        "output_dataset_vs_demo_gripper_rmse",
        "output_online_vs_demo_rmse",
        "output_online_vs_demo_pos_rmse",
        "output_online_vs_demo_angle_rmse",
        "output_online_vs_demo_gripper_rmse",
        "next_obs_dataset_env_vs_demo_img1_rmse",
        "next_obs_dataset_env_vs_demo_img2_rmse",
        "next_obs_dataset_env_vs_demo_state_rmse",
        "next_obs_dataset_env_vs_demo_state_pos_rmse",
        "next_obs_dataset_env_vs_demo_state_angle_rmse",
        "next_obs_dataset_env_vs_demo_state_gripper_rmse",
        "next_obs_online_env_vs_dataset_env_img1_rmse",
        "next_obs_online_env_vs_dataset_env_img2_rmse",
        "next_obs_online_env_vs_dataset_env_state_rmse",
        "next_obs_online_env_vs_dataset_env_state_pos_rmse",
        "next_obs_online_env_vs_dataset_env_state_angle_rmse",
        "next_obs_online_env_vs_dataset_env_state_gripper_rmse",
        "next_obs_online_env_vs_demo_img1_rmse",
        "next_obs_online_env_vs_demo_img2_rmse",
        "next_obs_online_env_vs_demo_state_rmse",
        "next_obs_online_env_vs_demo_state_pos_rmse",
        "next_obs_online_env_vs_demo_state_angle_rmse",
        "next_obs_online_env_vs_demo_state_gripper_rmse",
        "compared_action_steps",
        "executed_action_steps",
        "dataset_done_after_chunk",
        "online_done_after_chunk",
        "stopped_due_to_early_done",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "inference_ix": record.get("inference_ix"),
                    "dataset_frame_idx": record.get("dataset_frame_idx"),
                    "online_env_step_before_inference": record.get("online_env_step_before_inference"),
                    "next_obs_demo_frame_idx": record.get("next_obs_demo_frame_idx"),
                    "input_img1_rmse": record.get("input_diff", {}).get("img1"),
                    "input_img2_rmse": record.get("input_diff", {}).get("img2"),
                    "input_state_rmse": record.get("input_diff", {}).get("state"),
                    "input_state_pos_rmse": record.get("input_diff", {}).get("state_pos"),
                    "input_state_angle_rmse": record.get("input_diff", {}).get("state_angle"),
                    "input_state_gripper_rmse": record.get("input_diff", {}).get("state_gripper"),
                    "output_dataset_vs_online_rmse": record.get("output_diff", {}).get("dataset_vs_online"),
                    "output_dataset_vs_online_pos_rmse": record.get("output_diff", {}).get("dataset_vs_online_pos"),
                    "output_dataset_vs_online_angle_rmse": record.get("output_diff", {}).get("dataset_vs_online_angle"),
                    "output_dataset_vs_online_gripper_rmse": record.get("output_diff", {}).get("dataset_vs_online_gripper"),
                    "output_dataset_vs_demo_rmse": record.get("output_diff", {}).get("dataset_vs_demo"),
                    "output_dataset_vs_demo_pos_rmse": record.get("output_diff", {}).get("dataset_vs_demo_pos"),
                    "output_dataset_vs_demo_angle_rmse": record.get("output_diff", {}).get("dataset_vs_demo_angle"),
                    "output_dataset_vs_demo_gripper_rmse": record.get("output_diff", {}).get("dataset_vs_demo_gripper"),
                    "output_online_vs_demo_rmse": record.get("output_diff", {}).get("online_vs_demo"),
                    "output_online_vs_demo_pos_rmse": record.get("output_diff", {}).get("online_vs_demo_pos"),
                    "output_online_vs_demo_angle_rmse": record.get("output_diff", {}).get("online_vs_demo_angle"),
                    "output_online_vs_demo_gripper_rmse": record.get("output_diff", {}).get("online_vs_demo_gripper"),
                    "next_obs_dataset_env_vs_demo_img1_rmse": nested_get(record, ("next_obs_diff", "dataset_env_vs_demo", "img1")),
                    "next_obs_dataset_env_vs_demo_img2_rmse": nested_get(record, ("next_obs_diff", "dataset_env_vs_demo", "img2")),
                    "next_obs_dataset_env_vs_demo_state_rmse": nested_get(record, ("next_obs_diff", "dataset_env_vs_demo", "state")),
                    "next_obs_dataset_env_vs_demo_state_pos_rmse": nested_get(record, ("next_obs_diff", "dataset_env_vs_demo", "state_pos")),
                    "next_obs_dataset_env_vs_demo_state_angle_rmse": nested_get(record, ("next_obs_diff", "dataset_env_vs_demo", "state_angle")),
                    "next_obs_dataset_env_vs_demo_state_gripper_rmse": nested_get(record, ("next_obs_diff", "dataset_env_vs_demo", "state_gripper")),
                    "next_obs_online_env_vs_dataset_env_img1_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_dataset_env", "img1")),
                    "next_obs_online_env_vs_dataset_env_img2_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_dataset_env", "img2")),
                    "next_obs_online_env_vs_dataset_env_state_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_dataset_env", "state")),
                    "next_obs_online_env_vs_dataset_env_state_pos_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_dataset_env", "state_pos")),
                    "next_obs_online_env_vs_dataset_env_state_angle_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_dataset_env", "state_angle")),
                    "next_obs_online_env_vs_dataset_env_state_gripper_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_dataset_env", "state_gripper")),
                    "next_obs_online_env_vs_demo_img1_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_demo", "img1")),
                    "next_obs_online_env_vs_demo_img2_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_demo", "img2")),
                    "next_obs_online_env_vs_demo_state_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_demo", "state")),
                    "next_obs_online_env_vs_demo_state_pos_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_demo", "state_pos")),
                    "next_obs_online_env_vs_demo_state_angle_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_demo", "state_angle")),
                    "next_obs_online_env_vs_demo_state_gripper_rmse": nested_get(record, ("next_obs_diff", "online_env_vs_demo", "state_gripper")),
                    "compared_action_steps": record.get("output_diff", {}).get("compared_action_steps"),
                    "executed_action_steps": record.get("executed_action_steps"),
                    "dataset_done_after_chunk": record.get("dataset_done_after_chunk"),
                    "online_done_after_chunk": record.get("online_done_after_chunk"),
                    "stopped_due_to_early_done": record.get("stopped_due_to_early_done"),
                }
            )

    return str(jsonl_path), str(csv_path)


def write_run_numeric_summary(*, pair_records: list[dict[str, Any]], output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "all_pairs_diff_summary.csv"
    fieldnames = [
        "pair_ix",
        "seed",
        "reference_dataset_episode_index",
        "n_inference_records",
        "max_inference_count",
        "first_input_equal",
        "dataset_success",
        "online_success",
        "dataset_sum_reward",
        "online_sum_reward",
        "dataset_ball_grasp_count",
        "online_ball_grasp_count",
        "mean_input_img1_rmse",
        "mean_input_img2_rmse",
        "mean_input_state_rmse",
        "mean_output_dataset_vs_online_rmse",
        "mean_output_dataset_vs_demo_rmse",
        "mean_output_online_vs_demo_rmse",
        "pair_jsonl_path",
        "pair_csv_path",
        "video_path",
        "combined_diff_plot_path",
        "component_diff_plot_path",
        "next_obs_overview_plot_path",
        "next_obs_state_component_plot_path",
    ]

    def mean_metric(records: list[dict[str, Any]], top_key: str, inner_key: str) -> float | None:
        values = [record.get(top_key, {}).get(inner_key) for record in records]
        values = [float(v) for v in values if v is not None]
        if not values:
            return None
        return float(np.mean(values))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pair_records:
            records = list(pair.get("per_inference_records", []))
            writer.writerow(
                {
                    "pair_ix": pair.get("pair_ix"),
                    "seed": pair.get("seed"),
                    "reference_dataset_episode_index": pair.get("reference_dataset_episode_index"),
                    "n_inference_records": len(records),
                    "max_inference_count": pair.get("max_inference_count"),
                    "first_input_equal": pair.get("first_input_equal"),
                    "dataset_success": pair.get("dataset_lane", {}).get("success"),
                    "online_success": pair.get("online_lane", {}).get("success"),
                    "dataset_sum_reward": pair.get("dataset_lane", {}).get("sum_reward"),
                    "online_sum_reward": pair.get("online_lane", {}).get("sum_reward"),
                    "dataset_ball_grasp_count": pair.get("dataset_lane", {}).get("ball_grasp_count"),
                    "online_ball_grasp_count": pair.get("online_lane", {}).get("ball_grasp_count"),
                    "mean_input_img1_rmse": mean_metric(records, "input_diff", "img1"),
                    "mean_input_img2_rmse": mean_metric(records, "input_diff", "img2"),
                    "mean_input_state_rmse": mean_metric(records, "input_diff", "state"),
                    "mean_output_dataset_vs_online_rmse": mean_metric(records, "output_diff", "dataset_vs_online"),
                    "mean_output_dataset_vs_demo_rmse": mean_metric(records, "output_diff", "dataset_vs_demo"),
                    "mean_output_online_vs_demo_rmse": mean_metric(records, "output_diff", "online_vs_demo"),
                    "pair_jsonl_path": pair.get("numeric_records_jsonl_path"),
                    "pair_csv_path": pair.get("numeric_records_csv_path"),
                    "video_path": pair.get("video_path"),
                    "combined_diff_plot_path": pair.get("combined_diff_plot_path"),
                    "component_diff_plot_path": pair.get("component_diff_plot_path"),
                    "next_obs_overview_plot_path": pair.get("next_obs_overview_plot_path"),
                    "next_obs_state_component_plot_path": pair.get("next_obs_state_component_plot_path"),
                }
            )

    return str(csv_path)


def execute_chunk(
    *,
    env: gym.vector.VectorEnv,
    pred_chunk_cpu: Tensor,
    execute_n_action_steps: int,
    zero_action: np.ndarray,
    dataset_lane_idx: int,
    dataset_binding: EpisodeBinding,
    dataset_frame_idx: int,
    step: int,
    render_callback: Any | None,
    inference_ix: int,
    grasp_prev: np.ndarray,
    track_ball_grasp: bool,
) -> tuple[
    dict[str, Any],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    np.ndarray,
    int,
    int,
    list[dict[str, Any]],
    bool,
]:
    all_actions: list[Tensor] = []
    all_rewards: list[Tensor] = []
    all_dones: list[Tensor] = []
    all_successes: list[Tensor] = []
    all_ball_grasp_events: list[Tensor] = []
    frame_overlays: list[dict[str, Any]] = []

    observation: dict[str, Any] | None = None
    current_step = int(step)
    chunk_exec_steps = min(int(execute_n_action_steps), int(pred_chunk_cpu.shape[1]))
    dataset_exhausted = False

    for chunk_step in range(chunk_exec_steps):
        action_step = pred_chunk_cpu[:, chunk_step].numpy()
        if dataset_frame_idx >= int(dataset_binding.dataset_length):
            action_step[dataset_lane_idx] = zero_action[dataset_lane_idx]
            dataset_exhausted = True

        observation, reward, terminated, truncated, info = env.step(action_step)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError("Unsupported `final_info` format.")
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        grasp_event_step = np.zeros(env.num_envs, dtype=np.int32)
        if track_ball_grasp:
            raw_grasp = env.call("is_ball_grasped")
            grasp_now = np.asarray(raw_grasp, dtype=bool)
            grasp_event_step = np.logical_and(np.logical_not(grasp_prev), grasp_now).astype(np.int32)
            grasp_prev = grasp_now

        done = terminated | truncated
        if dataset_exhausted:
            done[dataset_lane_idx] = True

        overlay = {
            "inference_ix": int(inference_ix),
            "action_step_in_chunk": int(chunk_step + 1),
            "chunk_exec_steps": int(chunk_exec_steps),
            "dataset_done": bool(done[dataset_lane_idx]),
            "online_done": bool(done[1 - dataset_lane_idx]),
        }
        frame_overlays.append(overlay)
        if render_callback is not None:
            render_callback(env, overlay)

        all_actions.append(torch.from_numpy(action_step))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done.copy()))
        all_successes.append(torch.tensor(successes))
        all_ball_grasp_events.append(torch.from_numpy(grasp_event_step))

        current_step += 1
        if not dataset_exhausted:
            dataset_frame_idx += 1
            if dataset_frame_idx >= int(dataset_binding.dataset_length):
                dataset_exhausted = True

        if np.all(done):
            break

    if observation is None:
        raise RuntimeError("Chunk execution produced no environment observation.")

    return (
        observation,
        all_actions,
        all_rewards,
        all_dones,
        all_successes,
        all_ball_grasp_events,
        grasp_prev,
        current_step,
        dataset_frame_idx,
        frame_overlays,
        dataset_exhausted,
    )


def rollout_paired_episode(
    *,
    env: gym.vector.SyncVectorEnv,
    policy: Any,
    dataset: Any,
    absolute_to_relative: dict[int, int] | None,
    binding: EpisodeBinding,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    shared_first_input_source: str,
    pair_seed: int,
    render_callback: Any | None,
    rollout_fps: float | None,
) -> dict[str, Any]:
    if shared_first_input_source not in {"dataset", "online"}:
        raise ValueError(f"Unsupported shared_first_input_source={shared_first_input_source!r}")

    policy.reset()
    policy.eval()
    initial_lane_queues = deepcopy(getattr(policy, "_queues", {}))
    lane_queues: dict[str, dict[str, Any]] = {
        "dataset": deepcopy(initial_lane_queues),
        "online": deepcopy(initial_lane_queues),
    }
    observation, _ = env.reset(seed=[0, 0])
    if render_callback is not None:
        render_callback(env)

    zero_action = zero_action_like(env)
    check_env_attributes_and_types(env)
    max_steps = int(env.call("_max_episode_steps")[0])

    track_ball_grasp = False
    grasp_prev = np.zeros(env.num_envs, dtype=bool)
    if hasattr(env, "call"):
        try:
            raw_grasp = env.call("is_ball_grasped")
            if isinstance(raw_grasp, (list, tuple)) and len(raw_grasp) == env.num_envs:
                grasp_prev = np.asarray(raw_grasp, dtype=bool)
                track_ball_grasp = True
        except Exception:  # noqa: BLE001
            track_ball_grasp = False

    step = 0
    dataset_frame_idx = 0
    dataset_lane_idx = 0
    online_lane_idx = 1
    dataset_trace: list[int] = []
    online_trace: list[int] = []
    all_actions: list[Tensor] = []
    all_rewards: list[Tensor] = []
    all_dones: list[Tensor] = []
    all_successes: list[Tensor] = []
    all_ball_grasp_events: list[Tensor] = []
    first_input_equal = False
    dataset_exhausted = False
    per_inference_records: list[dict[str, Any]] = []
    max_inference_count = 0

    progbar = trange(max_steps, desc="Running paired rollout", disable=inside_slurm(), leave=False)

    if shared_first_input_source == "dataset":
        shared_first_raw_batch = get_dataset_policy_batch(
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            binding=binding,
            effective_rename_map=effective_rename_map,
            preprocessor=preprocessor,
        )
    else:
        online_batch = prepare_online_policy_batch(
            env=env,
            observation=observation,
            env_preprocessor=env_preprocessor,
            rollout_fps=rollout_fps,
            step=step,
        )
        shared_first_raw_batch = preprocessor(online_batch)

    first_lane_batch = filter_policy_input_batch(policy=policy, raw_batch=clone_batch(shared_first_raw_batch))
    first_input_equal = True
    first_input_diff = compute_input_diff_metrics(first_lane_batch, first_lane_batch)
    lane_queues["online"] = seed_online_lane_queues_from_stacked_batch(
        stacked_batch=first_lane_batch,
        lane_queues=lane_queues["online"],
    )
    dataset_first_chunk = predict_chunk_from_batch(
        policy=policy,
        batch=first_lane_batch,
        seed=int(pair_seed),
        postprocessor=postprocessor,
        env_postprocessor=env_postprocessor,
    )
    online_first_chunk = predict_chunk_from_batch(
        policy=policy,
        batch=first_lane_batch,
        seed=int(pair_seed),
        postprocessor=postprocessor,
        env_postprocessor=env_postprocessor,
    )
    demo_first_chunk = extract_demo_action_chunk(
        dataset=dataset,
        absolute_to_relative=absolute_to_relative,
        binding=binding,
        dataset_frame_idx=0,
        execute_n_action_steps=execute_n_action_steps,
        effective_rename_map=effective_rename_map,
    )
    pred_chunk = torch.cat([dataset_first_chunk, online_first_chunk], dim=0)
    (
        observation,
        chunk_actions,
        chunk_rewards,
        chunk_dones,
        chunk_successes,
        chunk_grasps,
        grasp_prev,
        step,
        dataset_frame_idx,
        frame_overlays,
        dataset_exhausted,
    ) = execute_chunk(
        env=env,
        pred_chunk_cpu=pred_chunk.detach().to("cpu"),
        execute_n_action_steps=execute_n_action_steps,
        zero_action=zero_action,
        dataset_lane_idx=dataset_lane_idx,
        dataset_binding=binding,
        dataset_frame_idx=dataset_frame_idx,
        step=step,
        render_callback=render_callback,
        inference_ix=0,
        grasp_prev=grasp_prev,
        track_ball_grasp=track_ball_grasp,
    )
    next_obs_diff, next_obs_demo_frame_idx = compute_next_obs_diff_bundle(
        env=env,
        observation=observation,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
        rollout_fps=rollout_fps,
        step=step,
        dataset=dataset,
        absolute_to_relative=absolute_to_relative,
        binding=binding,
        dataset_frame_idx=dataset_frame_idx,
        effective_rename_map=effective_rename_map,
        dataset_lane_idx=dataset_lane_idx,
        online_lane_idx=online_lane_idx,
    )
    max_inference_count = 1
    per_inference_records.append(
        {
            "inference_ix": 0,
            "dataset_frame_idx": 0,
            "online_env_step_before_inference": 0,
            "input_diff": first_input_diff,
            "output_diff": compute_output_diff_metrics(
                dataset_chunk=dataset_first_chunk,
                online_chunk=online_first_chunk,
                demo_chunk=demo_first_chunk,
                execute_n_action_steps=execute_n_action_steps,
                actual_executed_steps=len(chunk_actions),
            ),
            "stopped_due_to_early_done": False,
            "dataset_done_after_chunk": bool(chunk_dones[-1][dataset_lane_idx].item()) if chunk_dones else False,
            "online_done_after_chunk": bool(chunk_dones[-1][online_lane_idx].item()) if chunk_dones else False,
            "executed_action_steps": int(len(chunk_actions)),
            "next_obs_demo_frame_idx": int(next_obs_demo_frame_idx),
            "next_obs_diff": next_obs_diff,
        }
    )
    all_actions.extend(chunk_actions)
    all_rewards.extend(chunk_rewards)
    all_dones.extend(chunk_dones)
    all_successes.extend(chunk_successes)
    all_ball_grasp_events.extend(chunk_grasps)
    dataset_trace.append(0)
    online_trace.append(0)
    progbar.update(len(chunk_actions))

    while step < max_steps:
        last_done = all_dones[-1].numpy()
        if np.all(last_done):
            break

        online_raw_batch = prepare_online_policy_batch(
            env=env,
            observation=observation,
            env_preprocessor=env_preprocessor,
            rollout_fps=rollout_fps,
            step=step,
        )
        online_processed = preprocessor(online_raw_batch)
        online_lane_frame = filter_policy_input_batch(policy=policy, raw_batch=slice_env_batch(online_processed, online_lane_idx))
        online_lane_batch, lane_queues["online"] = build_online_lane_batch_from_frame(
            policy=policy,
            raw_frame_batch=online_lane_frame,
            lane_queues=lane_queues["online"],
        )

        safe_frame = int(min(dataset_frame_idx, max(binding.dataset_length - 1, 0)))
        dataset_item = get_dataset_item_by_absolute_index(
            dataset,
            int(binding.dataset_from_index + safe_frame),
            absolute_to_relative,
        )
        dataset_raw_batch = default_collate([dataset_item])
        if effective_rename_map:
            renamed: dict[str, Any] = {}
            for key, value in dataset_raw_batch.items():
                renamed[effective_rename_map.get(key, key)] = value
            dataset_raw_batch = renamed
        dataset_processed = preprocessor(dataset_raw_batch)
        dataset_lane_batch = filter_policy_input_batch(policy=policy, raw_batch=dataset_processed)
        inference_ix = len(per_inference_records)
        stop_diff_recording = bool(last_done[dataset_lane_idx] or last_done[online_lane_idx])
        dataset_chunk = predict_chunk_from_batch(
            policy=policy,
            batch=dataset_lane_batch,
            seed=int(pair_seed) + step,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        online_chunk = predict_chunk_from_batch(
            policy=policy,
            batch=online_lane_batch,
            seed=int(pair_seed) + step,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        demo_chunk = extract_demo_action_chunk(
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            binding=binding,
            dataset_frame_idx=safe_frame,
            execute_n_action_steps=execute_n_action_steps,
            effective_rename_map=effective_rename_map,
        )
        if not stop_diff_recording:
            per_inference_records.append(
                {
                    "inference_ix": int(inference_ix),
                    "dataset_frame_idx": int(safe_frame),
                    "online_env_step_before_inference": int(step),
                    "input_diff": compute_input_diff_metrics(dataset_lane_batch, online_lane_batch),
                    "output_diff": compute_output_diff_metrics(
                        dataset_chunk=dataset_chunk,
                        online_chunk=online_chunk,
                        demo_chunk=demo_chunk,
                        execute_n_action_steps=execute_n_action_steps,
                        actual_executed_steps=min(
                            int(execute_n_action_steps),
                            int(dataset_chunk.shape[1]),
                            int(online_chunk.shape[1]),
                            int(demo_chunk.shape[1]) if demo_chunk is not None else int(execute_n_action_steps),
                        ),
                    ),
                    "stopped_due_to_early_done": False,
                    "dataset_done_after_chunk": False,
                    "online_done_after_chunk": False,
                    "executed_action_steps": None,
                }
            )
        pred_chunk = torch.cat([dataset_chunk, online_chunk], dim=0)
        (
            observation,
            chunk_actions,
            chunk_rewards,
            chunk_dones,
            chunk_successes,
            chunk_grasps,
            grasp_prev,
            step,
            dataset_frame_idx,
            frame_overlays,
            dataset_exhausted,
        ) = execute_chunk(
            env=env,
            pred_chunk_cpu=pred_chunk.detach().to("cpu"),
            execute_n_action_steps=execute_n_action_steps,
            zero_action=zero_action,
            dataset_lane_idx=dataset_lane_idx,
            dataset_binding=binding,
            dataset_frame_idx=dataset_frame_idx,
            step=step,
            render_callback=render_callback,
            inference_ix=inference_ix,
            grasp_prev=grasp_prev,
            track_ball_grasp=track_ball_grasp,
        )
        next_obs_diff, next_obs_demo_frame_idx = compute_next_obs_diff_bundle(
            env=env,
            observation=observation,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            rollout_fps=rollout_fps,
            step=step,
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            binding=binding,
            dataset_frame_idx=dataset_frame_idx,
            effective_rename_map=effective_rename_map,
            dataset_lane_idx=dataset_lane_idx,
            online_lane_idx=online_lane_idx,
        )
        max_inference_count = max(max_inference_count, int(inference_ix) + 1)
        if per_inference_records and per_inference_records[-1]["inference_ix"] == inference_ix:
            per_inference_records[-1]["dataset_done_after_chunk"] = bool(chunk_dones[-1][dataset_lane_idx].item()) if chunk_dones else False
            per_inference_records[-1]["online_done_after_chunk"] = bool(chunk_dones[-1][online_lane_idx].item()) if chunk_dones else False
            per_inference_records[-1]["executed_action_steps"] = int(len(chunk_actions))
            per_inference_records[-1]["next_obs_demo_frame_idx"] = int(next_obs_demo_frame_idx)
            per_inference_records[-1]["next_obs_diff"] = next_obs_diff
        all_actions.extend(chunk_actions)
        all_rewards.extend(chunk_rewards)
        all_dones.extend(chunk_dones)
        all_successes.extend(chunk_successes)
        all_ball_grasp_events.extend(chunk_grasps)
        dataset_trace.append(safe_frame)
        online_trace.append(step)
        progbar.update(len(chunk_actions))

        if dataset_exhausted and all_dones[-1][dataset_lane_idx]:
            pass

    actions = torch.stack(all_actions, dim=1) if all_actions else torch.empty((2, 0, 7))
    rewards = torch.stack(all_rewards, dim=1) if all_rewards else torch.empty((2, 0))
    dones = torch.stack(all_dones, dim=1) if all_dones else torch.empty((2, 0), dtype=torch.bool)
    successes = torch.stack(all_successes, dim=1) if all_successes else torch.empty((2, 0), dtype=torch.bool)
    grasps = (
        torch.stack(all_ball_grasp_events, dim=1)
        if all_ball_grasp_events
        else torch.empty((2, 0), dtype=torch.int32)
    )

    return {
        ACTION: actions,
        "reward": rewards,
        "done": dones,
        "success": successes,
        "ball_grasp_event": grasps,
        "dataset_trace": dataset_trace,
        "online_trace": online_trace,
        "first_input_equal": first_input_equal,
        "shared_first_input_source": shared_first_input_source,
        "reference_dataset_episode_index": int(binding.dataset_episode_index),
        "reference_dataset_length": int(binding.dataset_length),
        "dataset_exhausted": bool(dataset_exhausted),
        "per_inference_records": per_inference_records,
        "max_inference_count": int(max_inference_count),
    }


def summarize_lane(rollout_data: dict[str, Any], lane_idx: int) -> dict[str, Any]:
    done_tensor = rollout_data["done"][lane_idx]
    reward_tensor = rollout_data["reward"][lane_idx]
    success_tensor = rollout_data["success"][lane_idx]
    grasp_tensor = rollout_data["ball_grasp_event"][lane_idx]

    if done_tensor.numel() == 0:
        return {
            "sum_reward": 0.0,
            "max_reward": 0.0,
            "success": False,
            "ball_grasp_count": 0,
        }

    done_idx = int(torch.argmax(done_tensor.to(int)).item())
    mask = (torch.arange(done_tensor.shape[0]) <= done_idx).to(torch.float32)
    sum_reward = float(torch.sum(reward_tensor * mask).item())
    max_reward = float(torch.max(reward_tensor * mask).item())
    success = bool(torch.any(success_tensor[: done_idx + 1]).item())
    ball_grasp_count = int(torch.sum(grasp_tensor[: done_idx + 1]).item())
    return {
        "sum_reward": sum_reward,
        "max_reward": max_reward,
        "success": success,
        "ball_grasp_count": ball_grasp_count,
    }


def build_pair_record(
    *,
    pair_ix: int,
    binding: EpisodeBinding,
    rollout_data: dict[str, Any],
    seed: int | None,
    video_path: str | None = None,
    combined_diff_plot_path: str | None = None,
    component_diff_plot_path: str | None = None,
    next_obs_overview_plot_path: str | None = None,
    next_obs_state_component_plot_path: str | None = None,
) -> dict[str, Any]:
    dataset_lane = summarize_lane(rollout_data, 0)
    online_lane = summarize_lane(rollout_data, 1)
    return {
        "pair_ix": int(pair_ix),
        "seed": seed,
        "reference_dataset_episode_index": int(binding.dataset_episode_index),
        "reference_dataset_length": int(binding.dataset_length),
        "shared_first_input_source": rollout_data["shared_first_input_source"],
        "first_input_equal": bool(rollout_data["first_input_equal"]),
        "dataset_lane": dataset_lane,
        "online_lane": online_lane,
        "dataset_trace": list(rollout_data["dataset_trace"]),
        "online_trace": list(rollout_data["online_trace"]),
        "dataset_exhausted": bool(rollout_data["dataset_exhausted"]),
        "per_inference_records": list(rollout_data["per_inference_records"]),
        "max_inference_count": int(rollout_data["max_inference_count"]),
        "video_path": video_path,
        "combined_diff_plot_path": combined_diff_plot_path,
        "component_diff_plot_path": component_diff_plot_path,
        "next_obs_overview_plot_path": next_obs_overview_plot_path,
        "next_obs_state_component_plot_path": next_obs_state_component_plot_path,
    }


def maybe_write_pair_video(
    *,
    frames: list[np.ndarray],
    output_path: Path,
    fps: int,
) -> str | None:
    if not frames:
        return None
    stacked = np.stack(frames, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=write_video, args=(str(output_path), stacked, fps))
    thread.start()
    thread.join()
    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired LIBERO dyn-mini eval with shared start state and shared first inference input."
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
    parser.add_argument("--env.episode_length", dest="env_episode_length", type=int, default=260)
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
    parser.add_argument("--rollout.execute_n_action_steps", dest="rollout_execute_n_action_steps", type=int, default=4)
    parser.add_argument(
        "--rollout.shared_first_input_source",
        dest="rollout_shared_first_input_source",
        choices=["dataset", "online"],
        default="dataset",
    )
    parser.add_argument(
        "--dataset.repo_id",
        dest="dataset_repo_id",
        default="local/libero_dyn_mini_balanced500_scripted_v2",
    )
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(DATASET_ROOT_DEFAULT))
    parser.add_argument("--dataset.episodes", dest="dataset_episodes", default="0:20")
    parser.add_argument("--dataset.tolerance_s", dest="dataset_tolerance_s", type=float, default=1e-4)
    parser.add_argument("--eval.n_episodes", dest="eval_n_episodes", type=int, default=20)
    parser.add_argument("--eval.batch_size", dest="eval_batch_size", type=int, default=2)
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
    if args.eval_batch_size != 2:
        raise ValueError("This paired script requires eval.batch_size=2 exactly.")
    if args.eval_n_episodes <= 0:
        raise ValueError("eval.n_episodes must be > 0.")
    if args.rollout_execute_n_action_steps <= 0:
        raise ValueError("rollout.execute_n_action_steps must be > 0.")

    device = get_safe_torch_device(args.policy_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    image_noise_cfg = ImageNoiseConfig(enable=False, std=0.0, clip_min=0.0, clip_max=1.0)

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

    output_dir = make_output_dir(args.output_dir, policy_path=args.policy_path)
    videos_dir = output_dir / "videos"
    plots_dir = output_dir / "plots"
    numeric_dir = output_dir / "numeric"
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=effective_rename_map)
    policy.eval()
    preprocessor, postprocessor = build_processors(
        policy_cfg=policy_cfg,
        policy=policy,
        dataset_meta=dataset.meta,
        effective_rename_map=effective_rename_map,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    absolute_to_relative = get_absolute_to_relative_index(dataset)

    task_ids = parse_int_list(args.env_task_ids) or [0]
    if task_ids != [0]:
        raise ValueError("This script currently supports exactly one task id: 0.")

    env = make_paired_env(env_cfg=env_cfg, task_id=task_ids[0])

    pair_records: list[dict[str, Any]] = []
    video_paths: list[str] = []
    pair_frames: list[np.ndarray] = []

    def render_frame(active_env: gym.vector.VectorEnv, overlay: dict[str, Any] | None = None) -> None:
        if isinstance(active_env, gym.vector.SyncVectorEnv):
            frame_pair = np.stack([active_env.envs[i].render() for i in range(active_env.num_envs)], axis=0)
            combined = annotate_pair_frame(frame_pair, overlay)
            pair_frames.append(combined)

    start_t = time.time()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext():
            for pair_ix, binding in enumerate(episode_bindings):
                pair_frames = []
                rollout_data = rollout_paired_episode(
                    env=env,
                    policy=policy,
                    dataset=dataset,
                    absolute_to_relative=absolute_to_relative,
                    binding=binding,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    effective_rename_map=effective_rename_map,
                    execute_n_action_steps=int(args.rollout_execute_n_action_steps),
                    shared_first_input_source=args.rollout_shared_first_input_source,
                    pair_seed=args.seed + pair_ix,
                    render_callback=render_frame,
                    rollout_fps=float(args.env_fps),
                )
                video_path = maybe_write_pair_video(
                    frames=pair_frames,
                    output_path=videos_dir / f"pair_{pair_ix:04d}.mp4",
                    fps=int(env.unwrapped.metadata["render_fps"]),
                )
                combined_diff_plot_path = write_pair_combined_diff_plot(
                    records=rollout_data["per_inference_records"],
                    output_path=plots_dir / f"pair_{pair_ix:04d}_combined_diff.png",
                    max_inference_count=int(rollout_data["max_inference_count"]),
                )
                component_diff_plot_path = write_pair_component_diff_plot(
                    records=rollout_data["per_inference_records"],
                    output_path=plots_dir / f"pair_{pair_ix:04d}_component_diff.png",
                    max_inference_count=int(rollout_data["max_inference_count"]),
                )
                next_obs_overview_plot_path = write_pair_next_obs_overview_plot(
                    records=rollout_data["per_inference_records"],
                    output_path=plots_dir / f"pair_{pair_ix:04d}_next_obs_overview.png",
                    max_inference_count=int(rollout_data["max_inference_count"]),
                )
                next_obs_state_component_plot_path = write_pair_next_obs_state_component_plot(
                    records=rollout_data["per_inference_records"],
                    output_path=plots_dir / f"pair_{pair_ix:04d}_next_obs_state_component.png",
                    max_inference_count=int(rollout_data["max_inference_count"]),
                )
                numeric_jsonl_path, numeric_csv_path = write_pair_numeric_records(
                    pair_ix=pair_ix,
                    records=rollout_data["per_inference_records"],
                    output_dir=numeric_dir,
                )
                pair_record = build_pair_record(
                    pair_ix=pair_ix,
                    binding=binding,
                    rollout_data=rollout_data,
                    seed=args.seed + pair_ix,
                    video_path=video_path,
                    combined_diff_plot_path=combined_diff_plot_path,
                    component_diff_plot_path=component_diff_plot_path,
                    next_obs_overview_plot_path=next_obs_overview_plot_path,
                    next_obs_state_component_plot_path=next_obs_state_component_plot_path,
                )
                pair_record["numeric_records_jsonl_path"] = numeric_jsonl_path
                pair_record["numeric_records_csv_path"] = numeric_csv_path
                pair_records.append(pair_record)
                if video_path is not None:
                    video_paths.append(video_path)
    finally:
        close_envs({args.env_task: {task_ids[0]: env}})

    overall_success_dataset = [bool(item["dataset_lane"]["success"]) for item in pair_records]
    overall_success_online = [bool(item["online_lane"]["success"]) for item in pair_records]
    overall_sum_reward_dataset = [float(item["dataset_lane"]["sum_reward"]) for item in pair_records]
    overall_sum_reward_online = [float(item["online_lane"]["sum_reward"]) for item in pair_records]
    overall_ball_grasp_dataset = [int(item["dataset_lane"]["ball_grasp_count"]) for item in pair_records]
    overall_ball_grasp_online = [int(item["online_lane"]["ball_grasp_count"]) for item in pair_records]

    info: dict[str, Any] = {
        "per_episode": [
            {
                "episode_ix": int(item["pair_ix"]) * 2,
                "sum_reward": float(item["dataset_lane"]["sum_reward"]),
                "max_reward": float(item["dataset_lane"]["max_reward"]),
                "success": bool(item["dataset_lane"]["success"]),
                "seed": item["seed"],
                "ball_grasp_count": int(item["dataset_lane"]["ball_grasp_count"]),
                "ball_grasp_success": bool(item["dataset_lane"]["ball_grasp_count"] > 0),
            }
            for item in pair_records
        ]
        + [
            {
                "episode_ix": int(item["pair_ix"]) * 2 + 1,
                "sum_reward": float(item["online_lane"]["sum_reward"]),
                "max_reward": float(item["online_lane"]["max_reward"]),
                "success": bool(item["online_lane"]["success"]),
                "seed": item["seed"],
                "ball_grasp_count": int(item["online_lane"]["ball_grasp_count"]),
                "ball_grasp_success": bool(item["online_lane"]["ball_grasp_count"] > 0),
            }
            for item in pair_records
        ],
        "overall": {
            "avg_sum_reward_dataset_lane": float(np.mean(overall_sum_reward_dataset)) if overall_sum_reward_dataset else float("nan"),
            "avg_sum_reward_online_lane": float(np.mean(overall_sum_reward_online)) if overall_sum_reward_online else float("nan"),
            "pc_success_dataset_lane": float(np.mean(overall_success_dataset) * 100) if overall_success_dataset else float("nan"),
            "pc_success_online_lane": float(np.mean(overall_success_online) * 100) if overall_success_online else float("nan"),
            "avg_ball_grasp_count_dataset_lane": float(np.mean(overall_ball_grasp_dataset)) if overall_ball_grasp_dataset else float("nan"),
            "avg_ball_grasp_count_online_lane": float(np.mean(overall_ball_grasp_online)) if overall_ball_grasp_online else float("nan"),
            "n_pairs": len(pair_records),
            "n_episodes": len(pair_records) * 2,
            "eval_s": time.time() - start_t,
            "eval_pair_s": (time.time() - start_t) / max(1, len(pair_records)),
        },
        "paired_rollout": {
            "shared_first_input_source": args.rollout_shared_first_input_source,
            "first_input_equal_all_pairs": bool(all(item["first_input_equal"] for item in pair_records)),
            "pairs": pair_records,
            "video_paths": video_paths,
        },
    }

    eval_info_path = output_dir / "eval_info.json"
    eval_info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    run_numeric_summary_path = write_run_numeric_summary(pair_records=pair_records, output_dir=numeric_dir)

    summary = {
        "overall": info["overall"],
        "paired_rollout": {
            "shared_first_input_source": args.rollout_shared_first_input_source,
            "first_input_equal_all_pairs": bool(all(item["first_input_equal"] for item in pair_records)),
            "n_pairs": len(pair_records),
        },
    }
    summary_json_path = output_dir / "eval_info_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_text = json.dumps(summary, indent=2, ensure_ascii=False)
    summary_txt_path = output_dir / "eval_info_summary.txt"
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    run_meta = {
        "mode": "paired_obs_source_same_start",
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
        "shared_first_input_source": args.rollout_shared_first_input_source,
        "eval_n_episodes": int(args.eval_n_episodes),
        "eval_batch_size": int(args.eval_batch_size),
        "env": asdict(env_cfg),
        "effective_rename_map": dict(effective_rename_map),
        "run_numeric_summary_path": run_numeric_summary_path,
    }
    run_meta_path = output_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Paired overall metrics:")
    print(json.dumps(info["overall"], indent=2, ensure_ascii=False))
    print("\nPaired config:")
    print(json.dumps(summary["paired_rollout"], indent=2, ensure_ascii=False))

    logging.info("Saved eval info to %s", eval_info_path)
    logging.info("Saved eval summary json to %s", summary_json_path)
    logging.info("Saved eval summary text to %s", summary_txt_path)
    logging.info("Saved run meta to %s", run_meta_path)
    logging.info("Saved videos under %s", videos_dir)
    logging.info("Saved numeric diff tables under %s", numeric_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a policy on LIBERO dyn-mini using dataset observations for inference.

This script is a diagnostic evaluation mode for rollout-distribution analysis:

1. Reset the simulator to the dataset-corresponding initial scene.
2. Run policy inference from the offline dataset observation at frame t.
3. Execute only the first K predicted actions in the environment.
4. For the next inference, ignore the live environment observation and instead
   jump to the offline dataset observation at frame t + K.

The output format intentionally mirrors `lerobot_eval_dyn_mini_sync_fullvideo.py`:
videos, `eval_info.json`, `eval_info_summary.json`, and `eval_info_summary.txt`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from contextlib import nullcontext
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from lerobot.configs.types import FeatureType
from PIL import Image
from termcolor import colored
from torch import Tensor
from torch.utils.data._utils.collate import default_collate
from tqdm import trange

ROOT = Path(__file__).resolve().parents[4]
LIBERO_DYN_MINI_ROOT = ROOT / "LIBERO" / "libero_dyn_mini"
LEROBOT_SRC_ROOT = ROOT / "lerobot" / "src"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_DYN_MINI_ROOT / "config"))
sys.path.insert(0, str(LIBERO_DYN_MINI_ROOT / "py"))
sys.path.insert(0, str(LEROBOT_SRC_ROOT))

import libero_dyn_mini_v1  # noqa: F401

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.scripts.lerobot_eval_dyn_mini_sync_fullvideo import (
    _build_eval_info_summary,
    _format_eval_info_summary_text,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.libero_compat import (
    apply_rename_map_to_batch,
    apply_rename_map_to_preprocessor,
    resolve_libero_rename_map,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm


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


def build_obs_summary(per_records: list[dict[str, Any]]) -> dict[str, Any]:
    key_to_l2: dict[str, list[float]] = {}
    for record in per_records:
        for key, metrics in record.get("obs", {}).items():
            key_to_l2.setdefault(key, []).append(float(metrics["l2"]))
    return {key: summarize_metric_lists(values) for key, values in key_to_l2.items()}


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


DATASET_ROOT_DEFAULT = (
    LIBERO_DYN_MINI_ROOT / "datasets" / "libero_dyn_mini_balanced500_scripted_v2"
)
INIT_PLAN_DEFAULT_NAME = "rolling_ball_to_bowl.eval_from_dataset_balanced500_scripted_v2_first200.jsonl"
INIT_PLAN_DEFAULT_PATH = LIBERO_DYN_MINI_ROOT / "init_files" / "libero_dyn_mini" / INIT_PLAN_DEFAULT_NAME


@dataclass
class EpisodeBinding:
    dataset_episode_index: int
    plan_row_index: int
    dataset_from_index: int
    dataset_length: int


@dataclass(frozen=True)
class ImageNoiseConfig:
    enable: bool = False
    std: float = 0.0
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass(frozen=True)
class NoisyImageSaveConfig:
    enable: bool = False
    output_dir: Path | None = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_int_list(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_episode_selector(raw: str | None, *, max_episodes: int | None = None) -> list[int] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() == "all":
        return None
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid episode selector: {raw!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else max_episodes
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if stop is None:
            raise ValueError("Open-ended episode range requires a known dataset episode count.")
        return list(range(start, stop, step))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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
    execute_n_action_steps: int,
) -> Path:
    if raw_output_dir:
        path = Path(raw_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy_label = infer_policy_variant_label(policy_path)
        eval_label = f"dataset_obs_exec{int(execute_n_action_steps)}"
        path = Path("outputs/eval9") / f"{policy_label}_{eval_label}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def resolve_dataset_init_plan_path(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
    else:
        path = INIT_PLAN_DEFAULT_PATH.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Init plan not found: {path}")
    return path


def infer_effective_rename_map(
    *,
    dataset_meta: Any,
    env_cfg: LiberoEnvConfig,
    user_rename_map: dict[str, str],
    enable_legacy_compat: bool,
) -> dict[str, str]:
    return resolve_libero_rename_map(
        enable_legacy_compat=enable_legacy_compat,
        env_cfg=env_cfg,
        feature_keys=dataset_meta.features.keys(),
        user_rename_map=user_rename_map,
    )


def build_dataset(
    *,
    dataset_repo_id: str,
    dataset_root: Path,
    dataset_episodes: list[int] | None,
    env_cfg: LiberoEnvConfig,
    policy_cfg: PreTrainedConfig,
    tolerance_s: float,
    legacy_obs_compat: bool,
    rename_map: dict[str, str],
) -> tuple[LeRobotDataset, dict[str, str]]:
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset_repo_id,
            root=str(dataset_root),
            episodes=list(dataset_episodes) if dataset_episodes is not None else None,
        ),
        env=env_cfg,
        policy=policy_cfg,
        num_workers=1,
        tolerance_s=float(tolerance_s),
    )
    train_cfg.rename_map = dict(rename_map)
    train_cfg.libero_legacy_obs_compat = bool(legacy_obs_compat)
    dataset = make_dataset(train_cfg)
    if not isinstance(dataset, LeRobotDataset):
        raise TypeError(f"Expected a LeRobotDataset, got {type(dataset)}")
    effective_rename_map = infer_effective_rename_map(
        dataset_meta=dataset.meta,
        env_cfg=env_cfg,
        user_rename_map=rename_map,
        enable_legacy_compat=legacy_obs_compat,
    )
    return dataset, effective_rename_map


def build_processors(
    *,
    policy_cfg: PreTrainedConfig,
    policy: Any,
    dataset_meta: Any,
    effective_rename_map: dict[str, str],
) -> tuple[Any, Any]:
    dataset_stats_for_processor = (
        rename_stats(dataset_meta.stats, effective_rename_map) if effective_rename_map else dataset_meta.stats
    )
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": effective_rename_map},
        "normalizer_processor": {
            "stats": dataset_stats_for_processor,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": dataset_stats_for_processor,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        }
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    apply_rename_map_to_preprocessor(preprocessor, effective_rename_map)
    return preprocessor, postprocessor


def predict_action_chunk(policy: Any, batch: dict[str, Tensor]) -> Tensor:
    if getattr(policy, "diffusion", None) is not None:
        model_input = dict(batch)
        if getattr(policy.config, "image_features", None) and OBS_IMAGES not in model_input:
            model_input[OBS_IMAGES] = torch.stack(
                [model_input[key] for key in policy.config.image_features],
                dim=-4,
            )
        model_input.pop(ACTION, None)
        model_input.pop(f"{ACTION}_is_pad", None)
        return policy.diffusion.generate_actions(model_input)

    model_input = dict(batch)
    model_input.pop(ACTION, None)
    model_input.pop(f"{ACTION}_is_pad", None)
    return policy.predict_action_chunk(model_input)


def apply_image_noise_to_batch(
    batch: dict[str, Any],
    *,
    input_features: dict[str, Any],
    noise_cfg: ImageNoiseConfig,
) -> dict[str, Any]:
    if not noise_cfg.enable or noise_cfg.std <= 0:
        return batch

    visual_keys = [
        key
        for key, feature in input_features.items()
        if getattr(feature, "type", None) == FeatureType.VISUAL and key in batch and isinstance(batch[key], torch.Tensor)
    ]
    if not visual_keys:
        return batch

    noisy_batch = dict(batch)
    for key in visual_keys:
        image = noisy_batch[key]
        image_float = image.to(torch.float32)
        if image.dtype == torch.uint8:
            image_float = image_float / 255.0
        image_float = image_float + torch.randn_like(image_float) * noise_cfg.std
        image_float = image_float.clamp(noise_cfg.clip_min, noise_cfg.clip_max)
        noisy_batch[key] = image_float

    return noisy_batch


def get_visual_feature_keys(batch: dict[str, Any], *, input_features: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in input_features.items()
        if getattr(feature, "type", None) == FeatureType.VISUAL and key in batch and isinstance(batch[key], torch.Tensor)
    ]


def tensor_to_uint8_image(image: Tensor) -> np.ndarray:
    tensor = image.detach().to("cpu")
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor rank 3, got shape {tuple(tensor.shape)}")

    if tensor.shape[0] in {1, 3, 4} and tensor.shape[-1] not in {1, 3, 4}:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Unable to infer image layout from shape {tuple(tensor.shape)}")

    if tensor.dtype.is_floating_point:
        tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        tensor = tensor.clamp(0, 255).to(torch.uint8)

    image_np = tensor.numpy()
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    return image_np


def tensor_to_uint8_image_sequence(images: Tensor) -> list[np.ndarray]:
    tensor = images.detach().to("cpu")
    if tensor.ndim == 3:
        return [tensor_to_uint8_image(tensor)]
    if tensor.ndim != 4:
        raise ValueError(f"Expected image tensor rank 3 or 4, got shape {tuple(tensor.shape)}")

    # Support both [T, C, H, W] and [T, H, W, C].
    if tensor.shape[1] in {1, 3, 4} and tensor.shape[-1] not in {1, 3, 4}:
        return [tensor_to_uint8_image(frame) for frame in tensor]
    if tensor.shape[-1] in {1, 3, 4}:
        return [tensor_to_uint8_image(frame) for frame in tensor]

    raise ValueError(f"Unable to infer image sequence layout from shape {tuple(tensor.shape)}")


def save_noisy_images_for_inference(
    batch: dict[str, Any],
    *,
    input_features: dict[str, Any],
    save_cfg: NoisyImageSaveConfig,
    dataset_episode_indices: np.ndarray,
    episode_output_indices: list[int],
    observed_frame_indices: np.ndarray,
    dataset_lengths: np.ndarray,
    execute_n_action_steps: int,
    inference_call_indices: np.ndarray,
) -> None:
    if not save_cfg.enable or save_cfg.output_dir is None:
        return

    visual_keys = get_visual_feature_keys(batch, input_features=input_features)
    if not visual_keys:
        return

    for env_idx, episode_output_idx in enumerate(episode_output_indices):
        if episode_output_idx < 0 or inference_call_indices[env_idx] < 0:
            continue

        inference_idx = int(inference_call_indices[env_idx])

        sample_dir = (
            save_cfg.output_dir
            / f"episode_{episode_output_idx:04d}"
            / f"infer_{inference_idx:03d}_frame_{int(observed_frame_indices[env_idx]):04d}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "episode_output_index": int(episode_output_idx),
            "dataset_episode_index": int(dataset_episode_indices[env_idx]),
            "dataset_frame_index": int(observed_frame_indices[env_idx]),
            "inference_call_index": inference_idx,
            "dataset_length": int(dataset_lengths[env_idx]),
            "execute_n_action_steps": int(execute_n_action_steps),
            "visual_keys": list(visual_keys),
        }
        (sample_dir / "meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        for key in visual_keys:
            safe_key = _sanitize_path_token(key.replace("/", "_"))
            image_tensor = batch[key][env_idx]
            metadata[f"{key}_shape"] = list(image_tensor.shape)
            image_sequence = tensor_to_uint8_image_sequence(image_tensor)
            if len(image_sequence) == 1:
                Image.fromarray(image_sequence[0]).save(sample_dir / f"{safe_key}.png")
            else:
                for frame_idx, image_np in enumerate(image_sequence):
                    Image.fromarray(image_np).save(sample_dir / f"{safe_key}_t{frame_idx:02d}.png")

        (sample_dir / "meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def postprocess_action_chunk(
    *,
    normalized_actions: Tensor,
    postprocessor: Any,
    env_postprocessor: Any,
) -> Tensor:
    if normalized_actions.ndim == 2:
        normalized_actions = normalized_actions.unsqueeze(1)
    if normalized_actions.ndim != 3:
        raise RuntimeError(f"Expected action chunk rank 3, got shape {tuple(normalized_actions.shape)}")

    batch_size, chunk_size, action_dim = normalized_actions.shape
    flattened = normalized_actions.reshape(batch_size * chunk_size, action_dim)
    unnormalized = postprocessor(flattened)
    action_transition = {ACTION: unnormalized}
    action_transition = env_postprocessor(action_transition)
    env_actions = action_transition[ACTION]
    return env_actions.reshape(batch_size, chunk_size, action_dim)


def validate_episode_plan_alignment(
    *,
    dataset: LeRobotDataset,
    plan_rows: list[dict[str, Any]],
    selected_episode_indices: list[int],
    eval_n_episodes: int,
) -> list[EpisodeBinding]:
    if eval_n_episodes > len(selected_episode_indices):
        raise ValueError(
            f"eval.n_episodes={eval_n_episodes} exceeds the selected dataset episode count={len(selected_episode_indices)}."
        )
    if eval_n_episodes > len(plan_rows):
        raise ValueError(
            f"eval.n_episodes={eval_n_episodes} exceeds init-plan rows={len(plan_rows)}."
        )

    bindings: list[EpisodeBinding] = []
    for rollout_ix in range(eval_n_episodes):
        dataset_episode_index = int(selected_episode_indices[rollout_ix])
        plan_row = plan_rows[rollout_ix]
        source_accept_index = plan_row.get("source_accept_index")
        if source_accept_index is not None and int(source_accept_index) != dataset_episode_index:
            raise ValueError(
                "Dataset episode selection does not align with the init plan. "
                f"rollout_ix={rollout_ix} dataset_episode_index={dataset_episode_index} "
                f"but init plan source_accept_index={source_accept_index}. "
                "Pass a matching `--env.init_plan_path` or keep the default first-200 selection."
            )
        episode_meta = dataset.meta.episodes[dataset_episode_index]
        bindings.append(
            EpisodeBinding(
                dataset_episode_index=dataset_episode_index,
                plan_row_index=rollout_ix,
                dataset_from_index=int(episode_meta["dataset_from_index"]),
                dataset_length=int(episode_meta["length"]),
            )
        )
    return bindings


def get_absolute_to_relative_index(dataset: LeRobotDataset) -> dict[int, int] | None:
    return getattr(dataset, "_absolute_to_relative_idx", None)


def get_dataset_item_by_absolute_index(
    dataset: LeRobotDataset,
    abs_index: int,
    absolute_to_relative: dict[int, int] | None,
) -> dict[str, Any]:
    rel_index = abs_index if absolute_to_relative is None else int(absolute_to_relative[abs_index])
    return dataset[rel_index]


def zero_action_like(env: gym.vector.VectorEnv) -> np.ndarray:
    try:
        return np.zeros_like(env.action_space.sample())
    except Exception:  # noqa: BLE001
        shape = getattr(env.action_space, "shape", None)
        if shape is None:
            shape = (env.num_envs, 7)
        return np.zeros(shape, dtype=np.float32)


def rollout_dataset_observation(
    *,
    env: gym.vector.VectorEnv,
    policy: Any,
    dataset: LeRobotDataset,
    absolute_to_relative: dict[int, int] | None,
    episode_bindings: list[EpisodeBinding],
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    effective_rename_map: dict[str, str],
    execute_n_action_steps: int,
    image_noise_cfg: ImageNoiseConfig,
    noisy_image_save_cfg: NoisyImageSaveConfig,
    episode_output_indices: list[int],
    env_preprocessor: Any,
    rollout_fps: float | None,
    seeds: list[int] | None = None,
    render_callback: Any | None = None,
) -> dict[str, Any]:
    if execute_n_action_steps <= 0:
        raise ValueError(f"execute_n_action_steps must be > 0. Got {execute_n_action_steps}.")

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
    inference_calls = np.zeros(num_envs, dtype=np.int64)
    dataset_frame_trace: list[list[int]] = [[] for _ in range(num_envs)]
    dataset_exhausted = np.logical_not(valid_slots).copy()
    probe_records: list[dict[str, Any]] = []
    history = PerEnvHistoryBuffer(num_envs=num_envs, maxlen=int(policy.config.n_obs_steps))
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
        desc=f"Running dataset-observation rollout with at most {max_steps} steps",
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
                logging.info("[dataset-obs][grasp] enabled grasp event tracking.")
        except Exception:  # noqa: BLE001
            track_ball_grasp = False

    step = 0
    while not np.all(done) and step < max_steps:
        batch_items: list[dict[str, Any]] = []
        observed_frame_indices = np.zeros(num_envs, dtype=np.int64)
        inference_call_indices = np.full(num_envs, -1, dtype=np.int64)
        for env_idx in range(num_envs):
            if not valid_slots[env_idx]:
                abs_index = 0
                observed_frame_indices[env_idx] = 0
            else:
                safe_frame = int(min(current_frame_indices[env_idx], max(dataset_lengths[env_idx] - 1, 0)))
                abs_index = int(dataset_starts[env_idx] + safe_frame)
                observed_frame_indices[env_idx] = safe_frame
                if not done[env_idx]:
                    inference_call_indices[env_idx] = inference_calls[env_idx]
                    dataset_frame_trace[env_idx].append(safe_frame)
                    inference_calls[env_idx] += 1
            batch_items.append(get_dataset_item_by_absolute_index(dataset, abs_index, absolute_to_relative))

        raw_batch = default_collate(batch_items)
        raw_batch = apply_rename_map_to_batch(raw_batch, effective_rename_map)
        raw_batch = apply_image_noise_to_batch(
            raw_batch,
            input_features=policy.config.input_features,
            noise_cfg=image_noise_cfg,
        )
        save_noisy_images_for_inference(
            raw_batch,
            input_features=policy.config.input_features,
            save_cfg=noisy_image_save_cfg,
            dataset_episode_indices=dataset_episode_indices,
            episode_output_indices=episode_output_indices,
            observed_frame_indices=observed_frame_indices,
            dataset_lengths=dataset_lengths,
            execute_n_action_steps=execute_n_action_steps,
            inference_call_indices=inference_call_indices,
        )
        policy_batch = preprocessor(raw_batch)
        online_batch = history.build_batch()
        obs_by_env = compare_policy_batches_by_env(online_batch, policy_batch)

        with torch.inference_mode():
            online_pred_chunk_normalized = predict_action_chunk(policy, online_batch)
        online_pred_chunk = postprocess_action_chunk(
            normalized_actions=online_pred_chunk_normalized,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        ).detach().to("cpu")

        with torch.inference_mode():
            pred_chunk_normalized = predict_action_chunk(policy, policy_batch)
        pred_chunk = postprocess_action_chunk(
            normalized_actions=pred_chunk_normalized,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
        )
        pred_chunk_cpu = pred_chunk.detach().to("cpu")
        act_metrics = tensor_metrics_by_env(online_pred_chunk.to(torch.float32), pred_chunk_cpu.to(torch.float32))
        chunk_exec_steps = min(int(execute_n_action_steps), int(pred_chunk_cpu.shape[1]))

        for env_idx in range(num_envs):
            if not valid_slots[env_idx] or done[env_idx]:
                continue
            probe_records.append(
                {
                    "episode_ix": int(episode_output_indices[env_idx]),
                    "seed": int(seeds[env_idx]) if seeds is not None else None,
                    "reference_dataset_episode_index": int(dataset_episode_indices[env_idx]),
                    "chunk_ix": int(inference_calls[env_idx] - 1),
                    "step_ix": int(current_frame_indices[env_idx]),
                    "obs": obs_by_env[env_idx]["obs"],
                    "action_chunk": {
                        "l2": float(act_metrics["l2"][env_idx]),
                        "l1_mean": float(act_metrics["l1_mean"][env_idx]),
                        "mse": float(act_metrics["mse"][env_idx]),
                    },
                    "observation_source": "dataset",
                }
            )

        for chunk_step in range(chunk_exec_steps):
            action_step = pred_chunk_cpu[:, chunk_step].numpy()

            active_mask = valid_slots & np.logical_not(done)
            if not np.any(active_mask):
                break

            # For envs that already exhausted the aligned dataset action stream, send zeros and keep them masked.
            can_consume_dataset_action = active_mask & (current_frame_indices < dataset_lengths)
            action_step[~can_consume_dataset_action] = zero_action[~can_consume_dataset_action]

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
                        "[dataset-obs][grasp] env.call('is_ball_grasped') returned invalid shape "
                        f"{type(raw_grasp)} len={len(raw_grasp) if isinstance(raw_grasp, (list, tuple)) else 'NA'}."
                    )
                grasp_now = np.asarray(raw_grasp, dtype=bool)
                grasp_event_step = np.logical_and(np.logical_not(grasp_prev), grasp_now).astype(np.int32)
                grasp_prev = grasp_now

            current_frame_indices[can_consume_dataset_action] += 1
            newly_exhausted = can_consume_dataset_action & (current_frame_indices >= dataset_lengths)
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
            if not (np.all(done) or step >= max_steps):
                history.update_batched(
                    prepare_online_single_step_frame(
                        env=env,
                        observation=observation,
                        env_preprocessor=env_preprocessor,
                        preprocessor=preprocessor,
                        rollout_fps=rollout_fps,
                        step=step,
                    ),
                    active_mask=valid_slots & np.logical_not(done),
                )
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
        "dataset_episode_indices": dataset_episode_indices.tolist(),
        "dataset_lengths": dataset_lengths.tolist(),
        "executed_dataset_steps": current_frame_indices.tolist(),
        "inference_calls": inference_calls.tolist(),
        "dataset_frame_trace": dataset_frame_trace,
        "dataset_exhausted": dataset_exhausted.tolist(),
        "probe_records": probe_records,
    }


def eval_policy_dataset_observation(
    *,
    env: gym.vector.VectorEnv,
    policy: Any,
    dataset: LeRobotDataset,
    absolute_to_relative: dict[int, int] | None,
    episode_bindings: list[EpisodeBinding],
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
    env_preprocessor: Any,
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
    all_dataset_episode_indices: list[int] = []
    all_dataset_lengths: list[int] = []
    all_executed_dataset_steps: list[int] = []
    all_inference_calls: list[int] = []
    all_dataset_frame_traces: list[list[int]] = []
    all_dataset_exhausted: list[bool] = []
    all_seeds: list[int | None] = []
    all_probe_records: list[dict[str, Any]] = []
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

        rollout_data = rollout_dataset_observation(
            env=env,
            policy=policy,
            dataset=dataset,
            absolute_to_relative=absolute_to_relative,
            episode_bindings=batch_bindings,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            effective_rename_map=effective_rename_map,
            execute_n_action_steps=execute_n_action_steps,
            image_noise_cfg=image_noise_cfg,
            noisy_image_save_cfg=noisy_image_save_cfg,
            episode_output_indices=episode_output_indices,
            env_preprocessor=env_preprocessor,
            rollout_fps=rollout_fps,
            seeds=seeds,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
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

        all_dataset_episode_indices.extend(rollout_data["dataset_episode_indices"][0:active_count])
        all_dataset_lengths.extend(rollout_data["dataset_lengths"][0:active_count])
        all_executed_dataset_steps.extend(rollout_data["executed_dataset_steps"][0:active_count])
        all_inference_calls.extend(rollout_data["inference_calls"][0:active_count])
        all_dataset_frame_traces.extend(rollout_data["dataset_frame_trace"][0:active_count])
        all_dataset_exhausted.extend(rollout_data["dataset_exhausted"][0:active_count])
        all_probe_records.extend(rollout_data.get("probe_records", []))

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
                "dataset_episode_index": int(all_dataset_episode_indices[i]),
                "sum_reward": sum_rewards[i],
                "max_reward": max_rewards[i],
                "success": bool(all_successes[i]),
                "seed": all_seeds[i] if i < len(all_seeds) else None,
                "ball_grasp_count": int(ball_grasp_counts[i]),
                "ball_grasp_success": bool(ball_grasp_successes[i]),
                "dataset_length": int(all_dataset_lengths[i]),
                "executed_dataset_steps": int(all_executed_dataset_steps[i]),
                "inference_calls": int(all_inference_calls[i]),
                "dataset_exhausted": bool(all_dataset_exhausted[i]),
                "dataset_frame_trace": list(all_dataset_frame_traces[i]),
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
            "avg_executed_dataset_steps": float(np.nanmean(all_executed_dataset_steps)),
            "avg_inference_calls": float(np.nanmean(all_inference_calls)),
            "pc_dataset_exhausted": float(np.nanmean(all_dataset_exhausted) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / max(1, n_episodes),
        },
        "online_probe": {
            "per_chunk_record": all_probe_records,
            "summary": aggregate_chunk_records(all_probe_records),
            "obs_summary": build_obs_summary(all_probe_records),
            "action_chunk_l2": summarize_metric_lists([float(item["action_chunk"]["l2"]) for item in all_probe_records]),
        },
    }
    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths
    return info


def eval_policy_all_dataset_observation(
    *,
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy: Any,
    dataset: LeRobotDataset,
    absolute_to_relative: dict[int, int] | None,
    all_episode_bindings: list[EpisodeBinding],
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
    env_preprocessor: Any,
    rollout_fps: float | None,
) -> dict[str, Any]:
    start_t = time.time()

    overall_sum_rewards: list[float] = []
    overall_max_rewards: list[float] = []
    overall_successes: list[bool] = []
    overall_ball_grasp_counts: list[int] = []
    overall_ball_grasp_successes: list[bool] = []
    overall_video_paths: list[str] = []
    overall_executed_dataset_steps: list[int] = []
    overall_inference_calls: list[int] = []
    overall_dataset_exhausted: list[bool] = []
    ball_grasp_success_per_episode: list[dict[str, Any]] = []
    per_task_infos: list[dict[str, Any]] = []
    overall_probe_records: list[dict[str, Any]] = []
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
        group_executed_dataset_steps: list[int] = []
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

            task_result = eval_policy_dataset_observation(
                env=env,
                policy=policy,
                dataset=dataset,
                absolute_to_relative=absolute_to_relative,
                episode_bindings=task_bindings,
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
                env_preprocessor=env_preprocessor,
                rollout_fps=rollout_fps,
            )

            per_episode = task_result["per_episode"]
            task_sum_rewards = [float(ep["sum_reward"]) for ep in per_episode]
            task_max_rewards = [float(ep["max_reward"]) for ep in per_episode]
            task_successes = [bool(ep["success"]) for ep in per_episode]
            task_ball_grasp_counts = [int(ep.get("ball_grasp_count", 0)) for ep in per_episode]
            task_ball_grasp_successes = [bool(ep.get("ball_grasp_success", False)) for ep in per_episode]
            task_video_paths = list(task_result.get("video_paths", []) or [])
            task_executed_dataset_steps = [int(ep.get("executed_dataset_steps", 0)) for ep in per_episode]
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
            group_executed_dataset_steps.extend(task_executed_dataset_steps)
            group_inference_calls.extend(task_inference_calls)
            group_dataset_exhausted.extend(task_dataset_exhausted)

            overall_sum_rewards.extend(task_sum_rewards)
            overall_max_rewards.extend(task_max_rewards)
            overall_successes.extend(task_successes)
            overall_ball_grasp_counts.extend(task_ball_grasp_counts)
            overall_ball_grasp_successes.extend(task_ball_grasp_successes)
            overall_video_paths.extend(task_video_paths)
            overall_executed_dataset_steps.extend(task_executed_dataset_steps)
            overall_inference_calls.extend(task_inference_calls)
            overall_dataset_exhausted.extend(task_dataset_exhausted)
            overall_probe_records.extend(task_result.get("online_probe", {}).get("per_chunk_record", []))

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
                        "executed_dataset_steps": task_executed_dataset_steps,
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
            "executed_dataset_steps": group_executed_dataset_steps,
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
            "avg_executed_dataset_steps": _agg_from_list(acc["executed_dataset_steps"]),
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
        "avg_executed_dataset_steps": _agg_from_list(overall_executed_dataset_steps),
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
        "online_probe": {
            "per_chunk_record": overall_probe_records,
            "summary": aggregate_chunk_records(overall_probe_records),
            "obs_summary": build_obs_summary(overall_probe_records),
            "action_chunk_l2": summarize_metric_lists(
                [float(item["action_chunk"]["l2"]) for item in overall_probe_records]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LIBERO dyn-mini by using offline dataset observations for inference and simulator execution for actions."
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
    parser.add_argument("--env.episode_length", dest="env_episode_length", type=int, default=None)
    parser.add_argument("--env.control_mode", dest="env_control_mode", default="relative")
    parser.add_argument("--env.init_states", dest="env_init_states", type=str2bool, default=True)
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=None)
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
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="local/libero_dyn_mini_balanced500_scripted_v2")
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(DATASET_ROOT_DEFAULT))
    parser.add_argument("--dataset.episodes", dest="dataset_episodes", default="0:200")
    parser.add_argument("--dataset.tolerance_s", dest="dataset_tolerance_s", type=float, default=1e-4)
    parser.add_argument("--eval.n_episodes", dest="eval_n_episodes", type=int, default=200)
    parser.add_argument("--eval.batch_size", dest="eval_batch_size", type=int, default=2)
    parser.add_argument("--rollout.execute_n_action_steps", dest="rollout_execute_n_action_steps", type=int, default=8)
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
    plan_rows = load_jsonl(plan_path)
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
        execute_n_action_steps=int(args.rollout_execute_n_action_steps),
    )
    videos_dir = output_dir / "videos" / datetime.now().strftime("%Y%m%d_%H%M%S")
    noisy_image_save_cfg = NoisyImageSaveConfig(
        enable=bool(args.dataset_image_noise_save_images_enable and image_noise_cfg.enable),
        output_dir=output_dir / "noisy_dataset_inputs",
    )
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
    if noisy_image_save_cfg.enable:
        logging.info("Saving noisy dataset images to %s", noisy_image_save_cfg.output_dir)

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
            info = eval_policy_all_dataset_observation(
                envs=envs,
                policy=policy,
                dataset=dataset,
                absolute_to_relative=absolute_to_relative,
                all_episode_bindings=episode_bindings,
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
                env_preprocessor=env_preprocessor,
                rollout_fps=float(args.env_fps),
            )
    finally:
        close_envs(envs)

    info["dataset_observation_eval"] = {
        "mode": "dataset_observation_for_inference_environment_for_execution",
        "policy_path": str(Path(args.policy_path).resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "selected_dataset_episodes": list(selected_episodes[: args.eval_n_episodes]),
        "init_plan_path": str(plan_path),
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
    }

    eval_info_path = output_dir / "eval_info.json"
    eval_info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    online_probe_path = output_dir / "online_probe_result.json"
    online_probe_path.write_text(
        json.dumps(info.get("online_probe", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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

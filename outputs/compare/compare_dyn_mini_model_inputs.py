#!/usr/bin/env python3
"""Compare offline vs online model inputs for LIBERO dyn-mini DiffusionPolicy.

This script answers a narrow question:
for one dataset episode and one logical timestep, do the tensors that finally
reach the policy differ between the offline dataset path and the online
environment path?

It does not compare actions or rollout success. Instead it saves and compares:
1. Offline observation sample before policy preprocessor.
2. Online observation sample before policy preprocessor.
3. Offline sample after policy preprocessor.
4. Online sample after policy preprocessor.
5. Final model-relevant batch tensors (e.g. observation.state / observation.images).

The online path replays dataset actions from the same episode so the environment
is advanced to the same logical timestep before comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_DYN_MINI_ROOT = REPO_ROOT.parent / "LIBERO" / "libero_dyn_mini"
LEROBOT_SRC_ROOT = REPO_ROOT / "src"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_DYN_MINI_ROOT / "config"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf/datasets")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg")
if str(LEROBOT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC_ROOT))
if str(LIBERO_DYN_MINI_ROOT / "py") not in sys.path:
    sys.path.insert(0, str(LIBERO_DYN_MINI_ROOT / "py"))

import libero_dyn_mini_v1  # noqa: F401

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.libero import load_episode_start_states, save_episode_start_states
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.libero_compat import apply_rename_map_to_preprocessor, resolve_libero_rename_map
from lerobot.utils.random_utils import set_seed


DATASET_ROOT_DEFAULT = (
    LIBERO_DYN_MINI_ROOT / "datasets" / "libero_dyn_mini_balanced500_scripted_v2"
)
INIT_PLAN_DEFAULT = (
    LIBERO_DYN_MINI_ROOT
    / "init_files"
    / "libero_dyn_mini"
    / "rolling_ball_to_bowl.eval_from_dataset_balanced500_scripted_v2_first200.jsonl"
)


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def cpu_clone_dict(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    if tensor.numel() == 0:
        return summary
    if tensor.dtype == torch.bool:
        summary["true_count"] = int(tensor.sum().item())
        return summary
    if tensor.is_floating_point() or tensor.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        summary["min"] = float(tensor.min().item())
        summary["max"] = float(tensor.max().item())
        summary["mean"] = float(tensor.float().mean().item())
    return summary


def value_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if torch.is_tensor(value):
        summary = tensor_summary(value)
        summary["kind"] = "tensor"
        return summary
    return {
        "kind": type(value).__name__,
        "repr": repr(value),
    }


def compare_tensor_pair(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "dtype_equal": a.dtype == b.dtype,
    }
    if tuple(a.shape) != tuple(b.shape):
        return result

    if a.dtype == torch.bool and b.dtype == torch.bool:
        neq = (a != b)
        result["exact_equal"] = bool((~neq).all().item())
        result["num_mismatched"] = int(neq.sum().item())
        return result

    if not torch.is_floating_point(a):
        a = a.float()
    if not torch.is_floating_point(b):
        b = b.float()
    diff = (a - b).abs()
    result["max_abs_diff"] = float(diff.max().item()) if diff.numel() > 0 else 0.0
    result["mean_abs_diff"] = float(diff.mean().item()) if diff.numel() > 0 else 0.0
    result["allclose_atol_1e-6"] = bool(torch.allclose(a, b, atol=1e-6, rtol=0.0))
    result["allclose_atol_1e-5"] = bool(torch.allclose(a, b, atol=1e-5, rtol=0.0))
    result["allclose_atol_1e-4"] = bool(torch.allclose(a, b, atol=1e-4, rtol=0.0))
    return result


def compare_flat_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in sorted(set(a) | set(b)):
        if key not in a:
            summary[key] = {"status": "missing_in_a"}
            continue
        if key not in b:
            summary[key] = {"status": "missing_in_b"}
            continue
        value_a, value_b = a[key], b[key]
        if torch.is_tensor(value_a) and torch.is_tensor(value_b):
            summary[key] = compare_tensor_pair(value_a.detach().cpu(), value_b.detach().cpu())
        else:
            summary[key] = {
                "status": "non_tensor",
                "repr_a": repr(value_a),
                "repr_b": repr(value_b),
                "equal": value_a == value_b,
            }
    return summary


def sanitize_path_token(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "unknown"


def infer_output_dir(raw_output_dir: str | None, policy_path: Path, episode_index: int, frame_index: int) -> Path:
    if raw_output_dir:
        path = Path(raw_output_dir).expanduser().resolve()
    else:
        policy_label = sanitize_path_token(policy_path.parent.name or policy_path.name)
        path = REPO_ROOT / "outputs" / "compare" / f"{policy_label}_ep{episode_index:04d}_f{frame_index:04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", default=str(DATASET_ROOT_DEFAULT))
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", default="local/libero_dyn_mini_balanced500_scripted_v2")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=str(INIT_PLAN_DEFAULT))
    parser.add_argument("--env.episode_start_states_path", dest="env_episode_start_states_path", default=None)
    parser.add_argument("--libero_legacy_obs_compat", action="store_true", default=False)
    return parser.parse_args()


def load_policy_config(policy_path: Path, device: str) -> PreTrainedConfig:
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[f"--device={device}"])
    cfg.pretrained_path = policy_path
    return cfg


def infer_image_hw(policy_cfg: PreTrainedConfig) -> tuple[int, int]:
    image_features = getattr(policy_cfg, "image_features", None)
    if not image_features:
        raise ValueError("The selected policy does not expose image_features.")
    first_ft = next(iter(image_features.values()))
    if len(first_ft.shape) != 3:
        raise ValueError(f"Expected CHW image feature shape, got {first_ft.shape}")
    _, height, width = first_ft.shape
    return int(height), int(width)


def select_plan_row(plan_rows: list[dict[str, Any]], episode_index: int) -> tuple[int, dict[str, Any]]:
    for row_index, row in enumerate(plan_rows):
        source_accept_index = row.get("source_accept_index")
        if source_accept_index is not None and int(source_accept_index) == episode_index:
            return row_index, row
    if 0 <= episode_index < len(plan_rows):
        return episode_index, plan_rows[episode_index]
    raise ValueError(
        "Could not align the requested episode with the init plan. "
        f"episode_index={episode_index}, init_plan_rows={len(plan_rows)}"
    )


def write_single_row_plan(output_dir: Path, row: dict[str, Any]) -> Path:
    plan_path = output_dir / "_temp" / "single_episode_init_plan.jsonl"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return plan_path


def maybe_write_single_entry_cache(
    *,
    source_cache_path: str | None,
    row_index: int,
    output_dir: Path,
) -> Path | None:
    if not source_cache_path:
        return None

    loaded = load_episode_start_states(source_cache_path)
    sim_states = loaded["sim_states"]
    table_body_quats = loaded["table_body_quats"]
    metadata = dict(loaded["metadata"])
    if row_index < 0 or row_index >= int(sim_states.shape[0]):
        raise ValueError(
            f"Requested row_index={row_index} is outside episode-start cache length={int(sim_states.shape[0])}."
        )
    single_state = sim_states[row_index : row_index + 1].copy()
    single_quat = None if table_body_quats is None else table_body_quats[row_index : row_index + 1].copy()
    metadata["selected_row_index"] = int(row_index)
    metadata["selected_from_cache_path"] = str(Path(source_cache_path).expanduser().resolve())
    cache_path = output_dir / "_temp" / "single_episode_start_state.npz"
    saved = save_episode_start_states(
        cache_path,
        sim_states=single_state,
        table_body_quats=single_quat,
        metadata=metadata,
    )
    return Path(saved)


def build_env_cfg(
    *,
    plan_path: Path,
    episode_start_cache_path: Path | None,
    image_height: int,
    image_width: int,
) -> LiberoEnvConfig:
    return LiberoEnvConfig(
        task="libero_dyn_mini",
        task_ids=[0],
        fps=30,
        episode_length=260,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        init_plan_path=str(plan_path),
        episode_start_states_path=str(episode_start_cache_path) if episode_start_cache_path else None,
        init_plan_loop=True,
        init_plan_launch_settle_steps=6,
        init_plan_launch_ramp_steps=8,
        init_plan_warmup_steps=0,
        ball_grasp_eval_mode="strict",
        ball_grasp_strict_lift_multiplier=1.2,
        ball_grasp_strict_grip_center_max_dist=0.045,
        ball_grasp_strict_require_pad_contact=True,
        observation_height=image_height,
        observation_width=image_width,
        control_mode="relative",
    )


def build_dataset_pair(
    *,
    dataset_repo_id: str,
    dataset_root: Path,
    episode_index: int,
    env_cfg: LiberoEnvConfig,
    policy_cfg: PreTrainedConfig,
    legacy_obs_compat: bool,
) -> tuple[LeRobotDataset, LeRobotDataset, dict[str, str]]:
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset_repo_id,
            root=str(dataset_root),
            episodes=[int(episode_index)],
        ),
        env=env_cfg,
        policy=policy_cfg,
        num_workers=1,
        tolerance_s=1e-4,
    )
    train_cfg.libero_legacy_obs_compat = bool(legacy_obs_compat)
    train_cfg.rename_map = {}
    dataset_with_deltas = make_dataset(train_cfg)
    if not isinstance(dataset_with_deltas, LeRobotDataset):
        raise TypeError(f"Expected LeRobotDataset, got {type(dataset_with_deltas)}")

    effective_rename_map = resolve_libero_rename_map(
        enable_legacy_compat=legacy_obs_compat,
        env_cfg=env_cfg,
        feature_keys=dataset_with_deltas.meta.features.keys(),
        user_rename_map={},
    )

    dataset_raw = LeRobotDataset(
        dataset_repo_id,
        root=str(dataset_root),
        episodes=[int(episode_index)],
    )
    return dataset_with_deltas, dataset_raw, effective_rename_map


def build_preprocessor(
    *,
    policy_cfg: PreTrainedConfig,
    dataset_meta: Any,
    effective_rename_map: dict[str, str],
) -> Any:
    dataset_stats_for_processor = (
        rename_stats(dataset_meta.stats, effective_rename_map) if effective_rename_map else dataset_meta.stats
    )
    preprocessor_overrides = {
        "device_processor": {"device": str(policy_cfg.device)},
        "rename_observations_processor": {"rename_map": effective_rename_map},
        "normalizer_processor": {
            "stats": dataset_stats_for_processor,
            "features": {**policy_cfg.input_features, **policy_cfg.output_features},
            "norm_map": policy_cfg.normalization_mapping,
        },
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    apply_rename_map_to_preprocessor(preprocessor, effective_rename_map)
    return preprocessor


def get_task_env(envs: dict[str, dict[int, Any]], env_cfg: LiberoEnvConfig) -> Any:
    task_group = env_cfg.task.split(",")[0].strip()
    task_id = int(env_cfg.task_ids[0]) if env_cfg.task_ids else 0
    return envs[task_group][task_id]


def process_single_online_observation(raw_observation: dict[str, Any], env_preprocessor: Any) -> dict[str, torch.Tensor]:
    observation = preprocess_observation(raw_observation)
    processed = env_preprocessor(observation)
    return processed


def maybe_inject_ball_pos(
    *,
    raw_observation: dict[str, Any],
    env: Any,
    policy_cfg: PreTrainedConfig,
) -> dict[str, Any]:
    future_ball_pos_key = str(getattr(policy_cfg, "future_ball_pos_key", "observation.ball_pos"))
    input_features = getattr(policy_cfg, "input_features", None) or {}
    need_scene_ball_pos = future_ball_pos_key in input_features
    if not need_scene_ball_pos:
        return raw_observation
    if "ball_pos" in raw_observation or "observation.ball_pos" in raw_observation:
        return raw_observation

    feature = input_features.get(future_ball_pos_key)
    if feature is None or len(feature.shape) != 1:
        return raw_observation
    ball_pos_dim = int(np.prod(feature.shape))

    raw_ball_pos = env.call("get_ball_pos")
    if not isinstance(raw_ball_pos, (list, tuple)) or len(raw_ball_pos) == 0:
        raise RuntimeError("env.call('get_ball_pos') returned no values while ball_pos is required.")

    ball_rows: list[np.ndarray] = []
    for env_idx, item in enumerate(raw_ball_pos):
        if item is None:
            raise RuntimeError(f"Missing ball position for env_idx={env_idx}.")
        row = np.asarray(item, dtype=np.float32).reshape(-1)
        if row.size < ball_pos_dim:
            raise RuntimeError(
                f"Invalid ball position shape for env_idx={env_idx}: dim={row.size} < expected {ball_pos_dim}."
            )
        if row.size > ball_pos_dim:
            row = row[:ball_pos_dim]
        if not np.isfinite(row).all():
            raise RuntimeError(f"Ball position has NaN/Inf for env_idx={env_idx}.")
        ball_rows.append(row.astype(np.float32, copy=False))

    augmented_observation = dict(raw_observation)
    augmented_observation["ball_pos"] = np.stack(ball_rows, axis=0)
    return augmented_observation


def extract_model_input_keys(policy_cfg: PreTrainedConfig) -> list[str]:
    keys: list[str] = []
    if OBS_STATE in policy_cfg.input_features:
        keys.append(OBS_STATE)
    image_features = getattr(policy_cfg, "image_features", None)
    if image_features:
        keys.extend(list(image_features.keys()))
    future_ball_pos_key = getattr(policy_cfg, "future_ball_pos_key", None)
    if future_ball_pos_key and future_ball_pos_key in policy_cfg.input_features:
        keys.append(future_ball_pos_key)
    if OBS_ENV_STATE in policy_cfg.input_features:
        keys.append(OBS_ENV_STATE)
    return keys


def build_online_history_sample(
    *,
    processed_history: list[dict[str, torch.Tensor]],
    target_frame_index: int,
    policy_cfg: PreTrainedConfig,
) -> dict[str, torch.Tensor]:
    if target_frame_index >= len(processed_history):
        raise ValueError(
            f"target_frame_index={target_frame_index} is outside processed history length={len(processed_history)}."
        )
    sample: dict[str, torch.Tensor] = {}
    model_input_keys = extract_model_input_keys(policy_cfg)
    deltas = list(policy_cfg.observation_delta_indices or [0])

    for key in model_input_keys:
        stacked: list[torch.Tensor] = []
        for delta in deltas:
            history_index = min(max(target_frame_index + int(delta), 0), len(processed_history) - 1)
            if key not in processed_history[history_index]:
                raise KeyError(f"Online processed observation missing key: {key}")
            value = processed_history[history_index][key]
            if not torch.is_tensor(value):
                raise TypeError(f"Expected tensor for online key {key}, got {type(value)}")
            if value.shape[0] != 1:
                raise ValueError(f"Expected batched online tensor with batch=1 for key {key}, got {tuple(value.shape)}")
            stacked.append(value[0].detach().cpu().clone())
        sample[key] = torch.stack(stacked, dim=0)
    return sample


def build_offline_observation_sample(
    dataset_item: dict[str, Any],
    policy_cfg: PreTrainedConfig,
) -> dict[str, torch.Tensor]:
    sample: dict[str, torch.Tensor] = {}
    for key in extract_model_input_keys(policy_cfg):
        if key not in dataset_item:
            raise KeyError(f"Offline dataset item missing key: {key}")
        value = dataset_item[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Expected tensor for offline key {key}, got {type(value)}")
        sample[key] = value.detach().cpu().clone()
    return sample


def assemble_final_model_batch(
    *,
    preprocessed_sample: dict[str, torch.Tensor],
    policy_cfg: PreTrainedConfig,
) -> dict[str, torch.Tensor]:
    batch = cpu_clone_dict(preprocessed_sample)
    final_batch: dict[str, torch.Tensor] = {}
    if OBS_STATE in batch:
        final_batch[OBS_STATE] = batch[OBS_STATE]
    if OBS_ENV_STATE in batch:
        final_batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE]

    image_features = getattr(policy_cfg, "image_features", None)
    if image_features:
        image_keys = list(image_features.keys())
        final_batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_keys], dim=-4)

    future_ball_pos_key = getattr(policy_cfg, "future_ball_pos_key", None)
    if future_ball_pos_key and future_ball_pos_key in batch:
        final_batch[future_ball_pos_key] = batch[future_ball_pos_key]
    if "timestamp" in batch:
        final_batch["timestamp"] = batch["timestamp"]
    return final_batch


def replay_online_history(
    *,
    env: Any,
    dataset_raw: LeRobotDataset,
    target_frame_index: int,
    env_preprocessor: Any,
    policy_cfg: PreTrainedConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]]]:
    raw_history: list[dict[str, Any]] = []
    processed_history: list[dict[str, torch.Tensor]] = []

    observation, _ = env.reset(seed=[int(seed)])
    observation = maybe_inject_ball_pos(raw_observation=observation, env=env, policy_cfg=policy_cfg)
    raw_history.append(copy.deepcopy(observation))
    processed_history.append(process_single_online_observation(observation, env_preprocessor))

    for step_index in range(target_frame_index):
        action_tensor = dataset_raw[step_index]["action"]
        if not torch.is_tensor(action_tensor):
            raise TypeError(f"Expected raw dataset action tensor, got {type(action_tensor)}")
        action_np = action_tensor.detach().cpu().numpy()[None, :]
        observation, _, terminated, truncated, _ = env.step(action_np)
        observation = maybe_inject_ball_pos(raw_observation=observation, env=env, policy_cfg=policy_cfg)
        raw_history.append(copy.deepcopy(observation))
        processed_history.append(process_single_online_observation(observation, env_preprocessor))
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            raise RuntimeError(
                f"Environment terminated/truncated during replay at step_index={step_index}, "
                "cannot align the requested timestep."
            )
    return raw_history, processed_history


def write_tensor_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_clone_dict(payload), path)


def write_python_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = infer_output_dir(args.output_dir, policy_path, args.episode_index, args.frame_index)

    set_seed(args.seed)
    policy_cfg = load_policy_config(policy_path, device=args.device)
    if getattr(policy_cfg, "type", None) != "diffusion":
        raise ValueError(f"Expected a diffusion policy, got {getattr(policy_cfg, 'type', None)!r}")
    if getattr(policy_cfg, "model", None) != "orig":
        raise ValueError(f"Expected diffusion model='orig', got {getattr(policy_cfg, 'model', None)!r}")

    image_height, image_width = infer_image_hw(policy_cfg)
    init_plan_path = Path(args.env_init_plan_path).expanduser().resolve()
    if not init_plan_path.exists():
        raise FileNotFoundError(f"Init plan not found: {init_plan_path}")
    plan_rows = load_jsonl(init_plan_path)
    plan_row_index, plan_row = select_plan_row(plan_rows, args.episode_index)
    single_plan_path = write_single_row_plan(output_dir, plan_row)
    single_cache_path = maybe_write_single_entry_cache(
        source_cache_path=args.env_episode_start_states_path,
        row_index=plan_row_index,
        output_dir=output_dir,
    )

    env_cfg = build_env_cfg(
        plan_path=single_plan_path,
        episode_start_cache_path=single_cache_path,
        image_height=image_height,
        image_width=image_width,
    )

    dataset_with_deltas, dataset_raw, effective_rename_map = build_dataset_pair(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=dataset_root,
        episode_index=args.episode_index,
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
        legacy_obs_compat=bool(args.libero_legacy_obs_compat),
    )

    episode_length = int(dataset_raw.meta.episodes[args.episode_index]["length"])
    if not (0 <= args.frame_index < episode_length):
        raise ValueError(
            f"frame_index={args.frame_index} is outside episode length={episode_length} for episode={args.episode_index}."
        )

    preprocessor = build_preprocessor(
        policy_cfg=policy_cfg,
        dataset_meta=dataset_with_deltas.meta,
        effective_rename_map=effective_rename_map,
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    offline_item = dataset_with_deltas[args.frame_index]
    offline_before = build_offline_observation_sample(offline_item, policy_cfg)
    offline_after = preprocessor(cpu_clone_dict(offline_before))
    offline_final = assemble_final_model_batch(preprocessed_sample=offline_after, policy_cfg=policy_cfg)

    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = get_task_env(envs, env_cfg)
    try:
        online_raw_history, online_processed_history = replay_online_history(
            env=env,
            dataset_raw=dataset_raw,
            target_frame_index=args.frame_index,
            env_preprocessor=env_preprocessor,
            policy_cfg=policy_cfg,
            seed=args.seed,
        )
    finally:
        env.close()

    online_before = build_online_history_sample(
        processed_history=online_processed_history,
        target_frame_index=args.frame_index,
        policy_cfg=policy_cfg,
    )
    online_after = preprocessor(cpu_clone_dict(online_before))
    online_final = assemble_final_model_batch(preprocessed_sample=online_after, policy_cfg=policy_cfg)

    write_python_dump(output_dir / "offline_dataset_item_full.pt", cpu_clone_dict(offline_item))
    write_python_dump(output_dir / "online_raw_observation_t.pt", copy.deepcopy(online_raw_history[args.frame_index]))
    write_tensor_dump(output_dir / "online_env_pre_observation_t.pt", online_processed_history[args.frame_index])
    write_tensor_dump(output_dir / "offline_before_policy_pre.pt", offline_before)
    write_tensor_dump(output_dir / "online_before_policy_pre.pt", online_before)
    write_tensor_dump(output_dir / "offline_after_policy_pre.pt", offline_after)
    write_tensor_dump(output_dir / "online_after_policy_pre.pt", online_after)
    write_tensor_dump(output_dir / "offline_final_model_batch.pt", offline_final)
    write_tensor_dump(output_dir / "online_final_model_batch.pt", online_final)

    metadata = {
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "episode_index": int(args.episode_index),
        "frame_index": int(args.frame_index),
        "episode_length": int(episode_length),
        "seed": int(args.seed),
        "policy_type": getattr(policy_cfg, "type", None),
        "policy_model_mode": getattr(policy_cfg, "model", None),
        "policy_device": str(policy_cfg.device),
        "effective_rename_map": effective_rename_map,
        "policy_observation_delta_indices": list(policy_cfg.observation_delta_indices or []),
        "policy_input_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in policy_cfg.input_features.items()
        },
        "plan_row_index": int(plan_row_index),
        "plan_row": plan_row,
        "source_init_plan_path": str(init_plan_path),
        "single_init_plan_path": str(single_plan_path),
        "source_episode_start_states_path": (
            str(Path(args.env_episode_start_states_path).expanduser().resolve())
            if args.env_episode_start_states_path
            else None
        ),
        "single_episode_start_states_path": str(single_cache_path) if single_cache_path else None,
        "env_cfg": asdict(env_cfg),
        "online_history_length": len(online_raw_history),
        "offline_before_summary": {key: value_summary(value) for key, value in offline_before.items()},
        "online_before_summary": {key: value_summary(value) for key, value in online_before.items()},
        "offline_after_summary": {key: value_summary(value) for key, value in offline_after.items()},
        "online_after_summary": {key: value_summary(value) for key, value in online_after.items()},
        "offline_final_summary": {key: value_summary(value) for key, value in offline_final.items()},
        "online_final_summary": {key: value_summary(value) for key, value in online_final.items()},
    }
    save_json(output_dir / "metadata.json", metadata)

    diff_summary = {
        "before_policy_pre": compare_flat_dicts(offline_before, online_before),
        "after_policy_pre": compare_flat_dicts(offline_after, online_after),
        "final_model_batch": compare_flat_dicts(offline_final, online_final),
    }
    save_json(output_dir / "diff_summary.json", diff_summary)

    print(f"Saved comparison artifacts to: {output_dir}")
    print(f"Plan row index used: {plan_row_index}")
    if single_cache_path is not None:
        print(f"Using single-entry episode-start cache: {single_cache_path}")
    else:
        print("No episode-start cache was provided; reset alignment relies on the selected init-plan row only.")


if __name__ == "__main__":
    main()

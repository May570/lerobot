#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export per-frame policy action chunks from an offline LeRobot dataset.

This script is designed for the LIBERO dyn-mini workflow used in `outputs/eval7/run_eval_seq.sh`.
It loads the training dataset, runs inference for every frame (using the policy's observation history
window, e.g. previous frame + current frame for diffusion), and saves the predicted future action chunk
aligned to the current frame.

By default it inherits dataset settings from `<policy.path>/train_config.json` and aligns inference
hyperparameters with `run_eval_seq.sh`:
  - policy.device=cuda
  - policy.use_amp=true
  - policy.n_action_steps=15
  - policy.num_inference_steps=20
  - policy.future_condition_delta=4
  - batch_size=2
  - seed=1000
"""

import copy
import dataclasses
import datetime as dt
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.libero_compat import (
    apply_rename_map_to_batch,
    apply_rename_map_to_preprocessor,
    resolve_libero_rename_map,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def _extract_override_keys(overrides: list[str] | None) -> set[str]:
    keys: set[str] = set()
    for arg in overrides or []:
        if not arg.startswith("--"):
            continue
        key = arg[2:]
        if "=" in key:
            key = key.split("=", 1)[0]
        keys.add(key)
    return keys


def _apply_run_eval_seq_defaults(policy_cfg: PreTrainedConfig, override_keys: set[str]) -> None:
    aligned_defaults = {
        "device": "cuda",
        "use_amp": True,
        "n_action_steps": 15,
        "num_inference_steps": 20,
        "future_condition_delta": 4,
    }
    for key, value in aligned_defaults.items():
        if key in override_keys or not hasattr(policy_cfg, key):
            continue
        setattr(policy_cfg, key, value)


def _to_python(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if dataclasses.is_dataclass(value):
        return _to_python(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ExportDatasetConfig:
    repo_id: str = ""
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False


@dataclass
class ExportActionChunksConfig:
    policy: PreTrainedConfig | None = None
    dataset: ExportDatasetConfig = field(default_factory=ExportDatasetConfig)
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    batch_size: int = 2
    num_workers: int = 0
    max_episodes: int | None = 200
    tolerance_s: float = 1e-4
    save_ground_truth: bool = True
    save_normalized_predictions: bool = False
    rename_map: dict[str, str] = field(default_factory=dict)
    libero_legacy_obs_compat: bool = False
    use_eval_alignment_defaults: bool = True
    resolved_env_type: str | None = field(init=False, default=None)
    resolved_env_task: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Policy path is required. Please pass `--policy.path=...`.")

        cli_overrides = parser.get_cli_overrides("policy")
        override_keys = _extract_override_keys(cli_overrides)

        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = Path(policy_path)
        if self.use_eval_alignment_defaults:
            _apply_run_eval_seq_defaults(self.policy, override_keys)

        train_cfg = self._load_train_config(policy_path)
        if train_cfg is not None and getattr(train_cfg, "env", None) is not None:
            self.resolved_env_type = getattr(train_cfg.env, "type", None)
            self.resolved_env_task = getattr(train_cfg.env, "task", None)
        self._inherit_dataset_defaults(train_cfg)
        self._inherit_rename_defaults(train_cfg)

        if not self.dataset.repo_id:
            raise ValueError(
                "Dataset repo_id is empty after resolving defaults. "
                "Please pass `--dataset.repo_id=...` or ensure train_config.json exists."
            )

        if not self.job_name:
            policy_root = Path(policy_path)
            if policy_root.name == "pretrained_model" and len(policy_root.parents) >= 3:
                policy_name = policy_root.parents[2].name
            else:
                policy_name = policy_root.name
            self.job_name = f"dataset_action_chunks_{policy_name}"

        if not self.output_dir:
            now = dt.datetime.now()
            export_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/dataset_action_chunks") / export_dir

    def _load_train_config(self, policy_path: str) -> TrainPipelineConfig | None:
        train_cfg_path = Path(policy_path) / TRAIN_CONFIG_NAME
        if not train_cfg_path.exists():
            logging.warning("No %s found under %s. Dataset defaults must come from CLI.", TRAIN_CONFIG_NAME, policy_path)
            return None
        train_cfg = TrainPipelineConfig.from_pretrained(train_cfg_path)
        return train_cfg

    def _inherit_dataset_defaults(self, train_cfg: TrainPipelineConfig | None) -> None:
        if train_cfg is None:
            return

        user_dataset = copy.deepcopy(self.dataset)
        default_dataset = ExportDatasetConfig()
        merged_dataset = ExportDatasetConfig(
            repo_id=train_cfg.dataset.repo_id,
            root=train_cfg.dataset.root,
            episodes=list(train_cfg.dataset.episodes) if train_cfg.dataset.episodes is not None else None,
            image_transforms=copy.deepcopy(train_cfg.dataset.image_transforms),
            revision=train_cfg.dataset.revision,
            use_imagenet_stats=train_cfg.dataset.use_imagenet_stats,
            video_backend=train_cfg.dataset.video_backend,
            streaming=train_cfg.dataset.streaming,
        )

        if user_dataset.repo_id:
            merged_dataset.repo_id = user_dataset.repo_id
        if user_dataset.root is not None:
            merged_dataset.root = user_dataset.root
        if user_dataset.episodes is not None:
            merged_dataset.episodes = list(user_dataset.episodes)
        if user_dataset.revision is not None:
            merged_dataset.revision = user_dataset.revision
        if user_dataset.use_imagenet_stats != default_dataset.use_imagenet_stats:
            merged_dataset.use_imagenet_stats = user_dataset.use_imagenet_stats
        if user_dataset.video_backend != default_dataset.video_backend:
            merged_dataset.video_backend = user_dataset.video_backend
        if user_dataset.streaming != default_dataset.streaming:
            merged_dataset.streaming = user_dataset.streaming
        if user_dataset.image_transforms != default_dataset.image_transforms:
            merged_dataset.image_transforms = user_dataset.image_transforms

        inherited_root = merged_dataset.root
        if inherited_root and user_dataset.root is None and not Path(inherited_root).exists():
            logging.warning(
                "Inherited dataset root %s does not exist on this machine. Falling back to the default LeRobot dataset root.",
                inherited_root,
            )
            merged_dataset.root = None

        if self.max_episodes is not None and self.max_episodes > 0:
            if merged_dataset.episodes is None:
                ds_meta = LeRobotDatasetMetadata(
                    merged_dataset.repo_id,
                    root=merged_dataset.root,
                    revision=merged_dataset.revision,
                )
                merged_dataset.episodes = list(range(min(self.max_episodes, ds_meta.total_episodes)))
            else:
                merged_dataset.episodes = list(merged_dataset.episodes)[: self.max_episodes]

        self.dataset = merged_dataset

    def _inherit_rename_defaults(self, train_cfg: TrainPipelineConfig | None) -> None:
        if train_cfg is None:
            return
        if not self.rename_map and getattr(train_cfg, "rename_map", None):
            self.rename_map = dict(train_cfg.rename_map)
        if not self.libero_legacy_obs_compat and getattr(train_cfg, "libero_legacy_obs_compat", False):
            self.libero_legacy_obs_compat = True

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _make_dataset_config(cfg: ExportActionChunksConfig) -> TrainPipelineConfig:
    env_cfg: Any = None
    if cfg.resolved_env_type is not None:
        env_cfg = type(
            "EnvCfg",
            (),
            {
                "type": cfg.resolved_env_type,
                "task": cfg.resolved_env_task,
            },
        )()
    elif cfg.libero_legacy_obs_compat and cfg.dataset.repo_id.startswith("libero"):
        env_cfg = type(
            "EnvCfg",
            (),
            {
                "type": "libero",
                "task": cfg.resolved_env_task or cfg.dataset.repo_id,
            },
        )()
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=list(cfg.dataset.episodes) if cfg.dataset.episodes is not None else None,
            image_transforms=copy.deepcopy(cfg.dataset.image_transforms),
            revision=cfg.dataset.revision,
            use_imagenet_stats=cfg.dataset.use_imagenet_stats,
            video_backend=cfg.dataset.video_backend,
            streaming=cfg.dataset.streaming,
        ),
        env=env_cfg,
        policy=cfg.policy,
        num_workers=max(int(cfg.num_workers), 1),
        tolerance_s=float(cfg.tolerance_s),
    )
    train_cfg.rename_map = dict(cfg.rename_map)
    train_cfg.libero_legacy_obs_compat = bool(cfg.libero_legacy_obs_compat)
    return train_cfg


def _infer_effective_rename_map(cfg: ExportActionChunksConfig, dataset_meta: LeRobotDatasetMetadata) -> dict[str, str]:
    env_cfg: Any = None
    if cfg.resolved_env_type is not None:
        env_cfg = type(
            "EnvCfg",
            (),
            {
                "type": cfg.resolved_env_type,
                "task": cfg.resolved_env_task,
            },
        )()
    elif cfg.dataset.repo_id == "libero_dyn_mini":
        # Reuse the same compatibility path used by the LIBERO eval/train scripts.
        env_cfg = type("EnvCfg", (), {"type": "libero", "task": "libero_dyn_mini"})()
    return resolve_libero_rename_map(
        enable_legacy_compat=cfg.libero_legacy_obs_compat,
        env_cfg=env_cfg,
        feature_keys=dataset_meta.features.keys(),
        user_rename_map=cfg.rename_map,
    )


def _build_processors(
    cfg: ExportActionChunksConfig,
    policy,
    dataset_meta,
    effective_rename_map: dict[str, str],
):
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
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    apply_rename_map_to_preprocessor(preprocessor, effective_rename_map)
    return preprocessor, postprocessor


def _predict_action_chunk(policy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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


def _postprocess_action_chunk(postprocessor, normalized_actions: torch.Tensor) -> torch.Tensor:
    batch_size, chunk_size, action_dim = normalized_actions.shape
    flattened = normalized_actions.reshape(batch_size * chunk_size, action_dim)
    unnormalized = postprocessor(flattened)
    return unnormalized.reshape(batch_size, chunk_size, action_dim)


def _aligned_ground_truth_chunk(
    raw_batch: dict[str, torch.Tensor],
    pred_chunk_len: int,
    *,
    n_obs_steps: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if ACTION not in raw_batch:
        return None, None

    gt_actions = raw_batch[ACTION]
    gt_pad = raw_batch.get(f"{ACTION}_is_pad")
    if gt_actions.ndim < 3:
        return None, None

    # Diffusion training windows are aligned as [-1, 0, 1, ...], so at inference we keep
    # the slice that starts at the current frame (offset n_obs_steps - 1).
    start = max(int(n_obs_steps) - 1, 0)
    end = min(start + int(pred_chunk_len), gt_actions.shape[1])
    gt_actions = gt_actions[:, start:end]
    if gt_pad is not None and gt_pad.ndim >= 2:
        gt_pad = gt_pad[:, start:end]
    else:
        gt_pad = None
    return gt_actions, gt_pad


def _episode_start_index(dataset_meta, episode_index: int) -> int:
    episode = dataset_meta.episodes[episode_index]
    return int(episode["dataset_from_index"])


def _make_fixed_size_list_array(
    values: np.ndarray,
    *,
    value_type: pa.DataType,
) -> pa.Array:
    if values.ndim == 1:
        return pa.array(values, type=value_type)
    if values.ndim == 2:
        flat = pa.array(values.reshape(-1), type=value_type)
        return pa.FixedSizeListArray.from_arrays(flat, int(values.shape[1]))
    if values.ndim == 3:
        flat = pa.array(values.reshape(-1), type=value_type)
        inner = pa.FixedSizeListArray.from_arrays(flat, int(values.shape[2]))
        return pa.FixedSizeListArray.from_arrays(inner, int(values.shape[1]))
    raise ValueError(f"Unsupported sidecar array rank {values.ndim}; expected 1, 2, or 3.")


def _write_sidecar_parquet(
    path: Path,
    *,
    buffers: dict[str, list[Any]],
    save_ground_truth: bool,
    save_normalized_predictions: bool,
) -> dict[str, Any]:
    arrays = [
        pa.array(np.asarray(buffers["episode_index"], dtype=np.int64)),
        pa.array(np.asarray(buffers["frame_index"], dtype=np.int64)),
        pa.array(np.asarray(buffers["dataset_index"], dtype=np.int64)),
        pa.array(np.asarray(buffers["timestamp"], dtype=np.float64)),
        _make_fixed_size_list_array(
            np.stack(buffers["pred_action_chunk"]).astype(np.float32, copy=False),
            value_type=pa.float32(),
        ),
    ]
    names = [
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "pred_action_chunk",
    ]

    if save_normalized_predictions:
        arrays.append(
            _make_fixed_size_list_array(
                np.stack(buffers["pred_action_chunk_normalized"]).astype(np.float32, copy=False),
                value_type=pa.float32(),
            )
        )
        names.append("pred_action_chunk_normalized")

    if save_ground_truth and buffers["gt_action_chunk"]:
        arrays.append(
            _make_fixed_size_list_array(
                np.stack(buffers["gt_action_chunk"]).astype(np.float32, copy=False),
                value_type=pa.float32(),
            )
        )
        names.append("gt_action_chunk")

    if save_ground_truth and buffers["gt_action_is_pad"]:
        arrays.append(
            _make_fixed_size_list_array(
                np.stack(buffers["gt_action_is_pad"]).astype(np.bool_, copy=False),
                value_type=pa.bool_(),
            )
        )
        names.append("gt_action_is_pad")

    table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(table, path, compression="zstd")
    return {
        "path": str(path),
        "num_rows": int(len(buffers["frame_index"])),
        "columns": names,
    }


@parser.wrap()
def export_main(cfg: ExportActionChunksConfig) -> None:
    init_logging()
    logging.info(pformat(_to_python(asdict(cfg))))

    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("A pretrained policy is required.")

    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if cfg.seed is not None:
        set_seed(cfg.seed)

    dataset_cfg = _make_dataset_config(cfg)
    try:
        dataset = make_dataset(dataset_cfg)
    except Exception as exc:  # noqa: BLE001
        root_hint = cfg.dataset.root if cfg.dataset.root is not None else "<default LeRobot cache>"
        raise RuntimeError(
            "Failed to load the offline dataset. "
            f"Tried root={root_hint!r} for repo_id={cfg.dataset.repo_id!r}. "
            "If the inherited path from train_config.json is stale on this machine, please pass "
            "`--dataset.root=/path/to/your/local/dataset`."
        ) from exc
    effective_rename_map = _infer_effective_rename_map(cfg, dataset.meta)
    logging.info("Effective rename map: %s", effective_rename_map)
    logging.info("Dataset loaded: %d samples across %d episodes", dataset.num_frames, dataset.num_episodes)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=effective_rename_map,
    )
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = _build_processors(cfg, policy, dataset.meta, effective_rename_map)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if int(cfg.num_workers) > 0 else None,
    )

    autocast_ctx = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    episode_summaries: list[dict[str, Any]] = []
    current_episode_index: int | None = None
    current_episode_num_frames = 0
    current_buffers = {
        "episode_index": [],
        "dataset_index": [],
        "frame_index": [],
        "timestamp": [],
        "pred_action_chunk": [],
        "pred_action_chunk_normalized": [],
        "gt_action_chunk": [],
        "gt_action_is_pad": [],
    }

    def flush_episode_summary() -> None:
        nonlocal current_episode_index, current_episode_num_frames
        if current_episode_index is None:
            return
        episode_summaries.append(
            {
                "episode_index": int(current_episode_index),
                "num_frames": int(current_episode_num_frames),
            }
        )
        current_episode_index = None
        current_episode_num_frames = 0

    progress = tqdm(dataloader, desc="Exporting action chunks", unit="batch")
    for raw_batch in progress:
        raw_batch = apply_rename_map_to_batch(raw_batch, effective_rename_map)
        preprocessed_batch = preprocessor(raw_batch)
        with torch.inference_mode(), autocast_ctx:
            pred_action_chunk_normalized = _predict_action_chunk(policy, preprocessed_batch)
        pred_action_chunk = _postprocess_action_chunk(postprocessor, pred_action_chunk_normalized)

        gt_action_chunk, gt_action_is_pad = _aligned_ground_truth_chunk(
            raw_batch,
            pred_chunk_len=pred_action_chunk.shape[1],
            n_obs_steps=int(cfg.policy.n_obs_steps),
        )

        pred_action_chunk_cpu = pred_action_chunk.detach().to("cpu", dtype=torch.float32).numpy()
        pred_action_chunk_normalized_cpu = (
            pred_action_chunk_normalized.detach().to("cpu", dtype=torch.float32).numpy()
            if cfg.save_normalized_predictions
            else None
        )
        gt_action_chunk_cpu = (
            gt_action_chunk.detach().to("cpu", dtype=torch.float32).numpy() if gt_action_chunk is not None else None
        )
        gt_action_is_pad_cpu = (
            gt_action_is_pad.detach().to("cpu", dtype=torch.bool).numpy() if gt_action_is_pad is not None else None
        )

        batch_size = pred_action_chunk_cpu.shape[0]
        for row_idx in range(batch_size):
            episode_index = int(raw_batch["episode_index"][row_idx].item())
            dataset_index = int(raw_batch["index"][row_idx].item())
            ep_start = _episode_start_index(dataset.meta, episode_index)
            frame_index = dataset_index - ep_start
            timestamp = float(raw_batch["timestamp"][row_idx].item())

            if current_episode_index is None:
                current_episode_index = episode_index
            elif episode_index != current_episode_index:
                flush_episode_summary()
                current_episode_index = episode_index

            current_episode_num_frames += 1
            current_buffers["episode_index"].append(episode_index)
            current_buffers["dataset_index"].append(dataset_index)
            current_buffers["frame_index"].append(frame_index)
            current_buffers["timestamp"].append(timestamp)
            current_buffers["pred_action_chunk"].append(pred_action_chunk_cpu[row_idx])
            if cfg.save_normalized_predictions and pred_action_chunk_normalized_cpu is not None:
                current_buffers["pred_action_chunk_normalized"].append(pred_action_chunk_normalized_cpu[row_idx])
            if cfg.save_ground_truth and gt_action_chunk_cpu is not None:
                current_buffers["gt_action_chunk"].append(gt_action_chunk_cpu[row_idx])
            if cfg.save_ground_truth and gt_action_is_pad_cpu is not None:
                current_buffers["gt_action_is_pad"].append(gt_action_is_pad_cpu[row_idx])

    flush_episode_summary()

    parquet_path = output_dir / "pred_action_chunks.parquet"
    parquet_summary = _write_sidecar_parquet(
        parquet_path,
        buffers=current_buffers,
        save_ground_truth=cfg.save_ground_truth,
        save_normalized_predictions=cfg.save_normalized_predictions,
    )

    summary = {
        "policy_path": str(cfg.policy.pretrained_path),
        "policy_type": str(cfg.policy.type),
        "output_dir": str(output_dir),
        "parquet_path": str(parquet_path),
        "seed": cfg.seed,
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "max_episodes": cfg.max_episodes,
        "dataset": {
            "repo_id": cfg.dataset.repo_id,
            "root": cfg.dataset.root,
            "episodes": list(cfg.dataset.episodes) if cfg.dataset.episodes is not None else None,
            "num_frames": int(dataset.num_frames),
            "num_episodes": int(dataset.num_episodes),
        },
        "effective_rename_map": dict(effective_rename_map),
        "inference_alignment": {
            "policy.device": str(cfg.policy.device),
            "policy.use_amp": bool(cfg.policy.use_amp),
            "policy.n_obs_steps": int(cfg.policy.n_obs_steps),
            "policy.n_action_steps": int(getattr(cfg.policy, "n_action_steps", 0)),
            "policy.num_inference_steps": getattr(cfg.policy, "num_inference_steps", None),
            "policy.future_condition_delta": getattr(cfg.policy, "future_condition_delta", None),
        },
        "sidecar": parquet_summary,
        "episode_summaries": episode_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(_to_python(summary), indent=2, ensure_ascii=False))
    (output_dir / "config.json").write_text(json.dumps(_to_python(asdict(cfg)), indent=2, ensure_ascii=False))

    logging.info("Saved sidecar parquet to %s", parquet_path)
    print(json.dumps(_to_python(summary), indent=2, ensure_ascii=False))


def main() -> None:
    export_main()


if __name__ == "__main__":
    main()

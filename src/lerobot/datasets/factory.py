#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from pprint import pformat
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_PREFIX, REWARD
from lerobot.utils.libero_compat import resolve_libero_rename_map

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


class LabelsEnvironmentStateDataset(torch.utils.data.Dataset):
    """Augment a LeRobot dataset with environment_state loaded from meta/labels.parquet."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        labels_path: str | Path,
        history_delta_indices: list[int],
    ):
        self.dataset = dataset
        self.labels_path = Path(labels_path)
        self.history_delta_indices = list(history_delta_indices)
        self._environment_state = self._load_environment_state(self.labels_path)
        self.env_state_dim = int(self._environment_state.shape[1])
        self.meta = dataset.meta
        self._validate_labels_coverage()
        self._augment_meta_features()

    def _load_environment_state(self, labels_path: Path) -> np.ndarray:
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Expected labels parquet at {labels_path}, but it does not exist."
            )

        labels = pd.read_parquet(labels_path, columns=["index", OBS_ENV_STATE])
        if "index" not in labels or OBS_ENV_STATE not in labels:
            raise ValueError(
                f"{labels_path} must contain 'index' and '{OBS_ENV_STATE}' columns."
            )
        if not labels["index"].is_unique:
            raise ValueError(f"{labels_path} must have unique values in the 'index' column.")

        labels = labels.sort_values("index").reset_index(drop=True)
        row_indices = labels["index"].to_numpy(dtype=np.int64)
        max_index = int(row_indices[-1])
        expected_indices = np.arange(max_index + 1, dtype=np.int64)
        if row_indices.shape[0] != expected_indices.shape[0] or not np.array_equal(row_indices, expected_indices):
            raise ValueError(
                f"{labels_path} must cover a contiguous [0, N] index range for labels-backed conditioning."
            )

        env_values = [np.asarray(value, dtype=np.float32) for value in labels[OBS_ENV_STATE]]
        env_dims = {value.shape for value in env_values}
        if len(env_dims) != 1:
            raise ValueError(
                f"{labels_path} contains inconsistent '{OBS_ENV_STATE}' shapes: {sorted(env_dims)}."
            )

        env_array = np.stack(env_values, axis=0)
        if env_array.ndim != 2:
            raise ValueError(
                f"{labels_path} must store 1D environment_state vectors per row. Got array shape {env_array.shape}."
            )
        if env_array.shape[1] <= 0:
            raise ValueError(
                f"{labels_path} must store non-empty environment_state vectors. Got shape {env_array.shape}."
            )
        if not np.isfinite(env_array).all():
            raise ValueError(f"{labels_path} contains non-finite values in '{OBS_ENV_STATE}'.")
        return env_array

    def _validate_labels_coverage(self) -> None:
        selected_episode_indices = self.dataset.episodes
        if selected_episode_indices is None:
            selected_episode_indices = range(len(self.meta.episodes))
        if len(selected_episode_indices) == 0:
            raise ValueError("Labels-backed environment_state requires at least one selected episode.")

        max_required_index = max(
            int(self.meta.episodes[ep_idx]["dataset_to_index"]) - 1 for ep_idx in selected_episode_indices
        )
        max_available_index = len(self._environment_state) - 1
        if max_required_index > max_available_index:
            raise ValueError(
                f"{self.labels_path} only covers indices up to {max_available_index}, "
                f"but the selected dataset episodes require labels through index {max_required_index}."
            )

    def _augment_meta_features(self) -> None:
        env_feature = {
            "dtype": "float32",
            "shape": [self.env_state_dim],
            "names": None,
        }
        existing_feature = self.meta.info["features"].get(OBS_ENV_STATE)
        if existing_feature is not None and existing_feature != env_feature:
            raise ValueError(
                f"Dataset already defines {OBS_ENV_STATE} with incompatible feature metadata: "
                f"{existing_feature} vs {env_feature}."
            )
        self.meta.info["features"][OBS_ENV_STATE] = env_feature

    def _get_environment_state_history(self, abs_idx: int, ep_idx: int) -> torch.Tensor:
        if abs_idx >= len(self._environment_state):
            raise IndexError(
                f"Sample index {abs_idx} is outside labels-backed environment_state range "
                f"[0, {len(self._environment_state) - 1}]."
            )

        episode = self.meta.episodes[ep_idx]
        ep_start = episode["dataset_from_index"]
        ep_end = episode["dataset_to_index"]
        env_indices = [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in self.history_delta_indices]
        return torch.from_numpy(self._environment_state[env_indices].copy())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        item = self.dataset[idx]

        abs_idx = item["index"]
        if torch.is_tensor(abs_idx):
            abs_idx = int(abs_idx.item())
        else:
            abs_idx = int(abs_idx)

        ep_idx = item["episode_index"]
        if torch.is_tensor(ep_idx):
            ep_idx = int(ep_idx.item())
        else:
            ep_idx = int(ep_idx)

        env_state_history = self._get_environment_state_history(abs_idx, ep_idx)
        expected_shape = (len(self.history_delta_indices), self.env_state_dim)
        if tuple(env_state_history.shape) != expected_shape:
            raise ValueError(
                f"Labels-backed `{OBS_ENV_STATE}` history for sample index {abs_idx} has shape "
                f"{tuple(env_state_history.shape)}, expected {expected_shape}."
            )
        if not torch.isfinite(env_state_history).all():
            raise ValueError(
                f"Labels-backed `{OBS_ENV_STATE}` history for sample index {abs_idx} contains non-finite values."
            )
        item[OBS_ENV_STATE] = env_state_history
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _maybe_add_labels_environment_state(
    dataset: LeRobotDataset | StreamingLeRobotDataset,
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | LabelsEnvironmentStateDataset:
    use_labels_environment_state = getattr(cfg.policy, "use_labels_environment_state", False)
    use_env_state_to_mask_future = getattr(cfg.policy, "use_env_state_to_mask_future", False)
    if not (use_labels_environment_state or use_env_state_to_mask_future):
        return dataset

    if isinstance(dataset, StreamingLeRobotDataset):
        raise ValueError(
            "Labels-backed environment_state requires a non-streaming dataset."
        )
    if not hasattr(cfg.policy, "observation_delta_indices"):
        raise ValueError(
            "Labels-backed environment_state requires a policy config with "
            "`observation_delta_indices` so labels-backed history can be aligned with observations."
        )

    labels_path = Path(dataset.root) / "meta" / "labels.parquet"
    wrapped_dataset = LabelsEnvironmentStateDataset(
        dataset=dataset,
        labels_path=labels_path,
        history_delta_indices=cfg.policy.observation_delta_indices,
    )

    if cfg.policy.input_features:
        non_obs_env_keys = [
            key
            for key, feature in cfg.policy.input_features.items()
            if feature.type is FeatureType.ENV and key != OBS_ENV_STATE
        ]
        if non_obs_env_keys:
            raise ValueError(
                "Labels-backed environment_state requires the only ENV input feature to be "
                f"`{OBS_ENV_STATE}`. Found additional ENV keys: {non_obs_env_keys}."
            )
        expected_feature = PolicyFeature(type=FeatureType.ENV, shape=(wrapped_dataset.env_state_dim,))
        existing_feature = cfg.policy.input_features.get(OBS_ENV_STATE)
        if existing_feature is not None and existing_feature != expected_feature:
            raise ValueError(
                f"`{OBS_ENV_STATE}` already exists in policy input_features with shape {existing_feature.shape}, "
                f"but labels-backed environment_state requires shape {expected_feature.shape}."
            )
        cfg.policy.input_features[OBS_ENV_STATE] = expected_feature

    return wrapped_dataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata,
    rename_map: dict[str, str] | None = None,
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    observation_delta_indices_by_key = getattr(cfg, "observation_delta_indices_by_key", None)
    for key in ds_meta.features:
        canonical_key = rename_map.get(key, key) if rename_map else key

        if canonical_key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if canonical_key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if canonical_key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            obs_delta_indices = (
                observation_delta_indices_by_key.get(canonical_key)
                if isinstance(observation_delta_indices_by_key, dict)
                else None
            )
            if obs_delta_indices is None:
                obs_delta_indices = cfg.observation_delta_indices
            delta_timestamps[key] = [i / ds_meta.fps for i in obs_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        effective_rename_map = resolve_libero_rename_map(
            enable_legacy_compat=cfg.libero_legacy_obs_compat,
            env_cfg=cfg.env,
            feature_keys=ds_meta.features.keys(),
            user_rename_map=cfg.rename_map,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta, rename_map=effective_rename_map)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    dataset = _maybe_add_labels_environment_state(dataset, cfg)

    return dataset

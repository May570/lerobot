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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _make_config(*, kalman_feature_mode: str = "posvel6") -> DiffusionConfig:
    return DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        enable_kalman_condition=True,
        kalman_feature_mode=kalman_feature_mode,
        kalman_force_zero_input=True,
        precomputed_kalman_root=None,
        kalman_use_dataset_stats_norm=False,
        down_dims=(64, 128),
    )


def test_kalman_force_zero_input_bypasses_other_paths(monkeypatch):
    cfg = _make_config(kalman_feature_mode="posvel6")
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    def _fail(*args, **kwargs):
        raise AssertionError("Expected kalman_force_zero_input to bypass this path")

    monkeypatch.setattr(model, "_get_precomputed_kalman_features_from_batch", _fail)
    monkeypatch.setattr(model, "_compute_online_kalman_from_state", _fail)
    if model.precomputed_kalman_reader is not None:
        monkeypatch.setattr(model.precomputed_kalman_reader, "get_kalman_features", _fail)

    batch_size = 3
    seq_len = cfg.n_obs_steps
    batch = {
        OBS_STATE: torch.randn(batch_size, seq_len, 8),
    }
    kalman = model._get_kalman_features(batch)

    assert kalman.shape == (batch_size, seq_len, 6)
    assert torch.count_nonzero(kalman).item() == 0

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
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STATE_RAW


def _make_config(
    *,
    kalman_feature_mode: str = "posvel6",
    enable_kalman_condition: bool = True,
    enable_kalman_posvel6_direct_condition: bool = False,
    kalman_force_zero_input: bool = True,
) -> DiffusionConfig:
    return DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        enable_kalman_condition=enable_kalman_condition,
        enable_kalman_posvel6_direct_condition=enable_kalman_posvel6_direct_condition,
        kalman_feature_mode=kalman_feature_mode,
        kalman_force_zero_input=kalman_force_zero_input,
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


def test_kalman_posvel6_direct_condition_is_online_from_processed_state():
    cfg = _make_config(
        enable_kalman_condition=False,
        enable_kalman_posvel6_direct_condition=True,
        kalman_force_zero_input=False,
    )
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    # Processed observation.state (what the policy actually consumes).
    state = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0]],
            [[2.0, 1.0, 0.5, 7.0, 7.0, 7.0, 7.0, 7.0], [2.0, 1.0, 0.5, 7.0, 7.0, 7.0, 7.0, 7.0]],
        ],
        dtype=torch.float32,
    )

    # Provide raw state with very different values; direct branch should ignore it.
    batch = {
        OBS_STATE: state,
        OBS_STATE_RAW: torch.full_like(state, 1234.0),
        OBS_IMAGES: torch.zeros((state.shape[0], cfg.n_obs_steps, 1, 3, 32, 32), dtype=torch.float32),
        "timestamp": torch.tensor([[0.0, 0.1], [0.0, 0.1]], dtype=torch.float32),
    }

    direct = model._compute_online_kalman_posvel6_direct_from_processed_state(batch)
    assert direct.shape == (2, cfg.n_obs_steps, 6)
    # Pos part is direct passthrough from processed state.
    assert torch.allclose(direct[..., :3], state[..., :3], atol=1e-6)
    # Changing position across time should induce non-zero filtered velocity at the next step.
    assert torch.count_nonzero(direct[0, 1, 3:]).item() > 0
    # Constant position should keep near-zero velocity.
    assert torch.allclose(direct[1, 1, 3:], torch.zeros(3), atol=1e-6)

    # Global condition includes direct 6D Kalman features by raw concatenation (as the tail).
    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(state.shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -6:], direct, atol=1e-6)

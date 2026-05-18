#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _make_config(
    *,
    model: str,
    delay_random: bool = False,
    delay_random_deltas: tuple[int, ...] = (2, 3, 4, 5, 6),
    delay_random_probs: tuple[float, ...] = (0.08, 0.17, 0.29, 0.29, 0.17),
    future_condition_delta: int = 2,
    disable_future_condition_gate: bool = False,
) -> ACTConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
    }
    if model == "scene_only":
        input_features["observation.ball_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(3,))

    return ACTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        use_vae=False,
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        dropout=0.0,
        pretrained_backbone_weights=None,
        model=model,
        delay_random=delay_random,
        delay_random_deltas=delay_random_deltas,
        delay_random_probs=delay_random_probs,
        future_condition_delta=future_condition_delta,
        disable_future_condition_gate=disable_future_condition_gate,
    )


def _make_batch(
    *,
    include_ball_pos: bool,
    ball_pos: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    batch = {
        OBS_STATE: torch.arange(2 * 1 * 8, dtype=torch.float32).view(2, 1, 8),
        OBS_ENV_STATE: torch.arange(2 * 1 * 4, dtype=torch.float32).view(2, 1, 4) / 10.0,
        ACTION: torch.arange(2 * 4 * 7, dtype=torch.float32).view(2, 4, 7) / 10.0,
        "action_is_pad": torch.zeros((2, 4), dtype=torch.bool),
    }
    if include_ball_pos:
        batch["observation.ball_pos"] = (
            ball_pos
            if ball_pos is not None
            else torch.tensor([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], dtype=torch.float32)
        )
    return batch


def test_act_orig_keeps_observation_delta_indices_none():
    cfg = _make_config(model="orig")
    assert cfg.observation_delta_indices is None
    assert cfg.observation_delta_indices_by_key is None


def test_act_scene_only_observation_delta_indices_by_key():
    cfg = _make_config(model="scene_only", future_condition_delta=4)
    assert cfg.observation_delta_indices == [0]
    assert cfg.observation_delta_indices_by_key == {
        OBS_STATE: [0],
        "observation.ball_pos": [4],
    }


def test_act_scene_only_future_token_uses_mlp_and_gate():
    cfg = _make_config(model="scene_only")
    policy = ACTPolicy(cfg)
    model = policy.model
    batch = _make_batch(include_ball_pos=True)

    scene_token = model._prepare_future_scene_token(
        batch,
        batch_size=2,
        device=batch[ACTION].device,
        dtype=batch[ACTION].dtype,
    )

    ball_pos = batch["observation.ball_pos"][:, 0]
    expected_cond = model.future_ball_pos_mlp(ball_pos)
    expected_cond = expected_cond * model.future_ball_pos_gate(expected_cond)
    expected_token = model.future_ball_pos_input_proj(expected_cond)
    assert torch.allclose(scene_token, expected_token, atol=1e-6)


def test_act_scene_only_future_token_can_bypass_gate():
    cfg = _make_config(model="scene_only", disable_future_condition_gate=True)
    policy = ACTPolicy(cfg)
    model = policy.model
    batch = _make_batch(include_ball_pos=True)

    scene_token = model._prepare_future_scene_token(
        batch,
        batch_size=2,
        device=batch[ACTION].device,
        dtype=batch[ACTION].dtype,
    )

    ball_pos = batch["observation.ball_pos"][:, 0]
    expected_cond = model.future_ball_pos_mlp(ball_pos)
    expected_token = model.future_ball_pos_input_proj(expected_cond)
    assert model.future_ball_pos_gate is None
    assert torch.allclose(scene_token, expected_token, atol=1e-6)


def test_act_scene_only_delay_random_training_samples_future_delta_and_eval_uses_fixed_delta():
    cfg = _make_config(
        model="scene_only",
        delay_random=True,
        future_condition_delta=2,
        delay_random_deltas=(2, 3, 4, 5, 6),
        delay_random_probs=(0.0, 0.0, 0.0, 1.0, 0.0),
    )
    policy = ACTPolicy(cfg)
    model = policy.model
    ball_future_candidates = torch.tensor(
        [
            [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]],
            [[0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9], [1.0, 1.0, 1.0], [1.1, 1.1, 1.1]],
        ],
        dtype=torch.float32,
    )
    batch = _make_batch(include_ball_pos=True, ball_pos=ball_future_candidates)

    model.train()
    scene_token_train = model._prepare_future_scene_token(
        batch,
        batch_size=2,
        device=batch[ACTION].device,
        dtype=batch[ACTION].dtype,
    )
    expected_train_cond = model.future_ball_pos_mlp(ball_future_candidates[:, 3])
    expected_train_cond = expected_train_cond * model.future_ball_pos_gate(expected_train_cond)
    expected_train_token = model.future_ball_pos_input_proj(expected_train_cond)
    assert torch.allclose(scene_token_train, expected_train_token, atol=1e-6)

    model.eval()
    scene_token_eval = model._prepare_future_scene_token(
        batch,
        batch_size=2,
        device=batch[ACTION].device,
        dtype=batch[ACTION].dtype,
    )
    expected_eval_cond = model.future_ball_pos_mlp(ball_future_candidates[:, 0])
    expected_eval_cond = expected_eval_cond * model.future_ball_pos_gate(expected_eval_cond)
    expected_eval_token = model.future_ball_pos_input_proj(expected_eval_cond)
    assert torch.allclose(scene_token_eval, expected_eval_token, atol=1e-6)


def test_act_scene_only_training_requires_future_ball_pos():
    cfg = _make_config(model="scene_only")
    policy = ACTPolicy(cfg)
    batch = _make_batch(include_ball_pos=False)

    with pytest.raises(ValueError, match="Missing required dataset feature for ACT scene_only"):
        policy.forward(batch)


def test_act_scene_only_forward_accepts_singleton_observation_axis():
    cfg = _make_config(model="scene_only")
    policy = ACTPolicy(cfg)
    batch = _make_batch(include_ball_pos=True)

    loss, loss_dict = policy.forward(batch)

    assert loss.ndim == 0
    assert "l1_loss" in loss_dict

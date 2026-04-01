#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

IMAGE_KEY = "observation.images.image"


def _make_config(
    *,
    enable_future_predictor: bool = True,
    future_predictor_type: str = "mlp",
    future_target_type: str = "future_feature",
    future_training_stage: str = "joint",
    future_condition_fusion: str = "concat",
    future_condition_proj_dim: int | None = None,
    lambda_future: float = 1.0,
    future_freeze_encoder: bool = False,
) -> DiffusionConfig:
    return DiffusionConfig(
        n_obs_steps=2,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        enable_future_predictor=enable_future_predictor,
        future_predictor_type=future_predictor_type,
        future_target_type=future_target_type,
        future_training_stage=future_training_stage,
        future_condition_fusion=future_condition_fusion,
        future_condition_proj_dim=future_condition_proj_dim,
        lambda_future=lambda_future,
        future_freeze_encoder=future_freeze_encoder,
        down_dims=(64, 128),
    )


def _make_batch(config: DiffusionConfig, *, with_actions: bool = True) -> dict[str, torch.Tensor]:
    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 8, dtype=torch.float32),
        IMAGE_KEY: torch.randn(batch_size, config.n_obs_steps, 3, 32, 32, dtype=torch.float32),
    }
    if with_actions:
        batch[ACTION] = torch.randn(batch_size, config.horizon, 7, dtype=torch.float32)
        batch["action_is_pad"] = torch.zeros(batch_size, config.horizon, dtype=torch.bool)
    return batch


def test_future_condition_concat_adds_expected_width():
    cfg_base = _make_config(enable_future_predictor=False)
    cfg_future = _make_config(enable_future_predictor=True, future_condition_fusion="concat")

    model_base = DiffusionPolicy(cfg_base).diffusion
    model_future = DiffusionPolicy(cfg_future).diffusion
    batch = _make_batch(cfg_future, with_actions=False)

    cond_batch = {
        OBS_STATE: batch[OBS_STATE],
        OBS_IMAGES: batch[IMAGE_KEY].unsqueeze(2),
    }
    cond_base = model_base._prepare_global_conditioning(cond_batch)
    cond_future = model_future._prepare_global_conditioning(cond_batch)

    assert cond_future.shape[1] - cond_base.shape[1] == model_future._future_condition_dim


def test_future_condition_project_concat_uses_projection_width():
    cfg_base = _make_config(enable_future_predictor=False)
    cfg_future = _make_config(
        enable_future_predictor=True,
        future_condition_fusion="project_concat",
        future_condition_proj_dim=11,
    )

    model_base = DiffusionPolicy(cfg_base).diffusion
    model_future = DiffusionPolicy(cfg_future).diffusion
    batch = _make_batch(cfg_future, with_actions=False)

    cond_batch = {
        OBS_STATE: batch[OBS_STATE],
        OBS_IMAGES: batch[IMAGE_KEY].unsqueeze(2),
    }
    cond_base = model_base._prepare_global_conditioning(cond_batch)
    cond_future = model_future._prepare_global_conditioning(cond_batch)

    assert cond_future.shape[1] - cond_base.shape[1] == 11


def test_joint_training_reports_policy_and_future_losses():
    cfg = _make_config(
        future_target_type="future_feature",
        future_training_stage="joint",
        lambda_future=0.5,
    )
    policy = DiffusionPolicy(cfg)
    batch = _make_batch(cfg, with_actions=True)

    loss, output_dict = policy.forward(batch)

    assert "policy_loss" in output_dict
    assert "future_loss" in output_dict
    assert "total_loss" in output_dict
    assert output_dict["future_stage"] == "joint"
    assert output_dict["policy_loss"] > 0
    assert output_dict["future_loss"] >= 0
    assert torch.isfinite(loss)
    assert abs(loss.item() - output_dict["total_loss"]) < 1e-6


def test_pretrain_stage_optimizes_future_loss_only():
    cfg = _make_config(
        future_target_type="future_delta",
        future_training_stage="pretrain",
        lambda_future=2.0,
    )
    policy = DiffusionPolicy(cfg)
    batch = _make_batch(cfg, with_actions=False)

    trainable_names = [name for name, param in policy.diffusion.named_parameters() if param.requires_grad]
    assert len(trainable_names) > 0
    assert all(name.startswith("future_predictor.") for name in trainable_names)

    loss, output_dict = policy.forward(batch)
    assert output_dict["future_stage"] == "pretrain"
    assert output_dict["policy_loss"] == 0.0
    assert output_dict["future_loss"] >= 0
    assert abs(loss.item() - cfg.lambda_future * output_dict["future_loss"]) < 1e-6


def test_future_freeze_encoder_freezes_rgb_encoder_only():
    cfg = _make_config(
        future_training_stage="joint",
        future_freeze_encoder=True,
    )
    policy = DiffusionPolicy(cfg).diffusion

    assert all(not p.requires_grad for p in policy.rgb_encoder.parameters())
    assert any(p.requires_grad for p in policy.unet.parameters())
    assert any(p.requires_grad for p in policy.future_predictor.parameters())

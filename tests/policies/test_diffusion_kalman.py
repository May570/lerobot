#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _make_config(
    *,
    kalman_feature_mode: str = "posvel6",
    enable_kalman_condition: bool = True,
    enable_kalman_feature_mlp: bool = False,
    kalman_feature_mlp_dim: int | None = None,
    enable_kalman_feature_gate: bool = False,
    enable_kalman_mid_only_condition: bool = False,
    kalman_force_zero_global_condition: bool = False,
) -> DiffusionConfig:
    return DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        enable_kalman_condition=enable_kalman_condition,
        kalman_feature_mode=kalman_feature_mode,
        enable_kalman_feature_mlp=enable_kalman_feature_mlp,
        kalman_feature_mlp_dim=kalman_feature_mlp_dim,
        enable_kalman_feature_gate=enable_kalman_feature_gate,
        enable_kalman_mid_only_condition=enable_kalman_mid_only_condition,
        kalman_force_zero_global_condition=kalman_force_zero_global_condition,
        down_dims=(64, 128),
    )


def _make_batch(n_obs_steps: int = 2) -> dict[str, torch.Tensor]:
    state = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0], [1.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0]],
            [[2.0, 1.0, 0.5, 7.0, 7.0, 7.0, 7.0, 7.0], [2.0, 1.0, 0.5, 7.0, 7.0, 7.0, 7.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    return {
        OBS_STATE: state,
        OBS_IMAGES: torch.zeros((state.shape[0], n_obs_steps, 1, 3, 32, 32), dtype=torch.float32),
        "timestamp": torch.tensor([[0.0, 0.1], [0.0, 0.1]], dtype=torch.float32),
    }


def test_kalman_posvel6_direct_concat_to_global_condition():
    cfg = _make_config(kalman_feature_mode="posvel6")
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman = model._compute_online_kalman_from_state(batch)

    assert kalman.shape == (2, cfg.n_obs_steps, 6)
    # Motion in sample 0 should induce non-zero velocity at step 1.
    assert torch.count_nonzero(kalman[0, 1, 3:]).item() > 0
    # Constant position in sample 1 should keep near-zero velocity.
    assert torch.allclose(kalman[1, 1, 3:], torch.zeros(3), atol=1e-6)

    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -6:], kalman, atol=1e-6)


def test_kalman_full10_direct_concat_to_global_condition():
    cfg = _make_config(kalman_feature_mode="full10")
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman = model._compute_online_kalman_from_state(batch)

    assert kalman.shape == (2, cfg.n_obs_steps, 10)
    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -10:], kalman, atol=1e-6)


def test_kalman_force_zero_global_condition():
    cfg = _make_config(kalman_feature_mode="posvel6", kalman_force_zero_global_condition=True)
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)

    assert torch.allclose(global_cond_steps[..., :8], batch[OBS_STATE], atol=1e-6)
    assert torch.count_nonzero(global_cond_steps[..., -6:]).item() == 0


def test_kalman_vel3_direct_concat_to_global_condition():
    cfg = _make_config(kalman_feature_mode="vel3")
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman = model._compute_online_kalman_from_state(batch)

    assert kalman.shape == (2, cfg.n_obs_steps, 3)
    assert torch.count_nonzero(kalman[0, 1, :]).item() > 0
    assert torch.allclose(kalman[1, 1, :], torch.zeros(3), atol=1e-6)

    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -3:], kalman, atol=1e-6)


def test_kalman_velpred6_direct_concat_to_global_condition():
    cfg = _make_config(kalman_feature_mode="velpred6")
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman = model._compute_online_kalman_from_state(batch)

    assert kalman.shape == (2, cfg.n_obs_steps, 6)
    # vel block responds to motion.
    assert torch.count_nonzero(kalman[0, 1, :3]).item() > 0
    # pred_exec block also responds for moving sample.
    assert torch.count_nonzero(kalman[0, 1, 3:]).item() > 0
    # stationary sample keeps near-zero velocity.
    assert torch.allclose(kalman[1, 1, :3], torch.zeros(3), atol=1e-6)

    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -6:], kalman, atol=1e-6)


def test_kalman_feature_mlp_enabled_applies_before_concat():
    cfg = _make_config(kalman_feature_mode="vel3", enable_kalman_feature_mlp=True)
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman_raw = model._compute_online_kalman_from_state(batch)
    kalman_proj = model.kalman_feature_mlp(kalman_raw)

    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -3:], kalman_proj, atol=1e-6)


def test_kalman_feature_mlp_dim_changes_concat_width():
    cfg = _make_config(
        kalman_feature_mode="vel3",
        enable_kalman_feature_mlp=True,
        kalman_feature_mlp_dim=5,
    )
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman_raw = model._compute_online_kalman_from_state(batch)
    kalman_proj = model.kalman_feature_mlp(kalman_raw)

    assert kalman_proj.shape[-1] == 5
    cfg_ref = _make_config(kalman_feature_mode="vel3", enable_kalman_feature_mlp=False)
    policy_ref = DiffusionPolicy(cfg_ref)
    model_ref = policy_ref.diffusion
    global_cond = model._prepare_global_conditioning(batch)
    global_cond_ref = model_ref._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    global_cond_ref_steps = global_cond_ref.view(batch[OBS_STATE].shape[0], cfg_ref.n_obs_steps, -1)
    assert global_cond_steps.shape[-1] == global_cond_ref_steps.shape[-1] + 2
    assert torch.allclose(global_cond_steps[..., -5:], kalman_proj, atol=1e-6)


def test_kalman_feature_gate_enabled_applies_before_concat():
    cfg = _make_config(kalman_feature_mode="vel3", enable_kalman_feature_gate=True)
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman_raw = model._compute_online_kalman_from_state(batch)
    kalman_gated = kalman_raw * model.kalman_feature_gate(kalman_raw)

    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -3:], kalman_gated, atol=1e-6)


def test_kalman_mid_only_condition_uses_split_global_conditions():
    cfg_mid = _make_config(
        kalman_feature_mode="vel3",
        enable_kalman_mid_only_condition=True,
    )
    policy_mid = DiffusionPolicy(cfg_mid)
    model_mid = policy_mid.diffusion

    batch = _make_batch(cfg_mid.n_obs_steps)
    downup_cond, mid_cond = model_mid._prepare_unet_conditioning(batch)

    assert mid_cond is not None
    bsz = batch[OBS_STATE].shape[0]
    downup_steps = downup_cond.view(bsz, cfg_mid.n_obs_steps, -1)
    mid_steps = mid_cond.view(bsz, cfg_mid.n_obs_steps, -1)
    assert torch.allclose(mid_steps[..., : downup_steps.shape[-1]], downup_steps, atol=1e-6)
    assert mid_steps.shape[-1] == downup_steps.shape[-1] + 3


def test_kalman_force_zero_overrides_mlp_and_gate():
    cfg = _make_config(
        kalman_feature_mode="vel3",
        enable_kalman_feature_mlp=True,
        enable_kalman_feature_gate=True,
        kalman_force_zero_global_condition=True,
    )
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)

    assert torch.count_nonzero(global_cond_steps[..., -3:]).item() == 0

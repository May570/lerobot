#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


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


def _make_future_mode_config(
    *,
    model: str,
    include_ball_pos: bool = False,
    disable_future_condition_gate: bool = False,
    delay_random: bool = False,
    delay_random_deltas: tuple[int, ...] = (2, 3, 4, 5, 6),
    delay_random_probs: tuple[float, ...] = (0.08, 0.17, 0.29, 0.29, 0.17),
    future_condition_delta: int = 2,
) -> DiffusionConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
    }
    if include_ball_pos:
        input_features["observation.ball_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(3,))

    return DiffusionConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        model=model,
        disable_future_condition_gate=disable_future_condition_gate,
        delay_random=delay_random,
        delay_random_deltas=delay_random_deltas,
        delay_random_probs=delay_random_probs,
        future_condition_delta=future_condition_delta,
        enable_kalman_condition=False,
        down_dims=(64, 128),
    )


def _make_future_mode_batch(
    *,
    include_future_state: bool,
    include_ball_pos: bool,
) -> dict[str, torch.Tensor]:
    state_steps = 3 if include_future_state else 2
    batch = {
        OBS_STATE: torch.arange(2 * state_steps * 8, dtype=torch.float32).view(2, state_steps, 8),
        OBS_ENV_STATE: torch.arange(2 * 2 * 4, dtype=torch.float32).view(2, 2, 4) / 10.0,
    }
    if include_ball_pos:
        batch["observation.ball_pos"] = torch.tensor(
            [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], dtype=torch.float32
        )
    return batch


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




def test_kalman_pred3_direct_concat_to_global_condition():
    cfg = _make_config(kalman_feature_mode="pred3")
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_batch(cfg.n_obs_steps)
    kalman = model._compute_online_kalman_from_state(batch)

    assert kalman.shape == (2, cfg.n_obs_steps, 3)
    # pred_exec should respond for moving sample.
    assert torch.count_nonzero(kalman[0, 1, :]).item() > 0

    global_cond = model._prepare_global_conditioning(batch)
    global_cond_steps = global_cond.view(batch[OBS_STATE].shape[0], cfg.n_obs_steps, -1)
    assert torch.allclose(global_cond_steps[..., -3:], kalman, atol=1e-6)


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


def test_future_mode_observation_delta_indices_by_key():
    cfg = _make_future_mode_config(model="robot_scene", include_ball_pos=True)
    assert cfg.observation_delta_indices_by_key == {
        OBS_STATE: [-1, 0, 2],
        "observation.ball_pos": [2],
    }


def test_delay_random_observation_delta_indices_by_key():
    cfg = _make_future_mode_config(model="robot_scene", include_ball_pos=True, delay_random=True)
    assert cfg.observation_delta_indices_by_key == {
        OBS_STATE: [-1, 0, 2, 3, 4, 5, 6],
        "observation.ball_pos": [2, 3, 4, 5, 6],
    }


def test_disable_future_condition_gate_requires_non_orig_model():
    with pytest.raises(ValueError, match="requires a non-`orig` model"):
        _make_future_mode_config(model="orig", disable_future_condition_gate=True)


def test_robot_only_future_state_gate_is_appended_to_global_condition():
    cfg_orig = _make_future_mode_config(model="orig")
    policy_orig = DiffusionPolicy(cfg_orig)
    model_orig = policy_orig.diffusion

    cfg_robot = _make_future_mode_config(model="robot_only")
    policy_robot = DiffusionPolicy(cfg_robot)
    model_robot = policy_robot.diffusion

    batch_robot = _make_future_mode_batch(include_future_state=True, include_ball_pos=False)
    batch_orig = {
        OBS_STATE: batch_robot[OBS_STATE][:, :2],
        OBS_ENV_STATE: batch_robot[OBS_ENV_STATE],
    }

    cond_orig = model_orig._prepare_global_conditioning(batch_orig)
    cond_robot = model_robot._prepare_global_conditioning(batch_robot)

    future_state = batch_robot[OBS_STATE][:, 2]
    expected_future = future_state * model_robot.future_state_gate(future_state)
    assert torch.allclose(cond_robot[:, : cond_orig.shape[-1]], cond_orig, atol=1e-6)
    assert torch.allclose(cond_robot[:, -8:], expected_future, atol=1e-6)


def test_robot_only_future_state_can_bypass_gate():
    cfg_orig = _make_future_mode_config(model="orig")
    policy_orig = DiffusionPolicy(cfg_orig)
    model_orig = policy_orig.diffusion

    cfg_robot = _make_future_mode_config(model="robot_only", disable_future_condition_gate=True)
    policy_robot = DiffusionPolicy(cfg_robot)
    model_robot = policy_robot.diffusion

    batch_robot = _make_future_mode_batch(include_future_state=True, include_ball_pos=False)
    batch_orig = {
        OBS_STATE: batch_robot[OBS_STATE][:, :2],
        OBS_ENV_STATE: batch_robot[OBS_ENV_STATE],
    }

    cond_orig = model_orig._prepare_global_conditioning(batch_orig)
    cond_robot = model_robot._prepare_global_conditioning(batch_robot)

    future_state = batch_robot[OBS_STATE][:, 2]
    assert model_robot.future_state_gate is None
    assert torch.allclose(cond_robot[:, : cond_orig.shape[-1]], cond_orig, atol=1e-6)
    assert torch.allclose(cond_robot[:, -8:], future_state, atol=1e-6)


def test_scene_only_future_ball_pos_uses_mlp_and_gate():
    cfg_orig = _make_future_mode_config(model="orig")
    policy_orig = DiffusionPolicy(cfg_orig)
    model_orig = policy_orig.diffusion

    cfg_scene = _make_future_mode_config(model="scene_only", include_ball_pos=True)
    policy_scene = DiffusionPolicy(cfg_scene)
    model_scene = policy_scene.diffusion

    batch_orig = _make_future_mode_batch(include_future_state=False, include_ball_pos=False)
    batch_scene = _make_future_mode_batch(include_future_state=False, include_ball_pos=True)

    cond_orig = model_orig._prepare_global_conditioning(batch_orig)
    cond_scene = model_scene._prepare_global_conditioning(batch_scene)

    ball_pos = batch_scene["observation.ball_pos"][:, -1]
    expected_scene = model_scene.future_ball_pos_mlp(ball_pos)
    expected_scene = expected_scene * model_scene.future_ball_pos_gate(expected_scene)
    assert torch.allclose(cond_scene[:, : cond_orig.shape[-1]], cond_orig, atol=1e-6)
    assert torch.allclose(cond_scene[:, -8:], expected_scene, atol=1e-6)


def test_scene_only_future_ball_pos_can_bypass_gate():
    cfg_orig = _make_future_mode_config(model="orig")
    policy_orig = DiffusionPolicy(cfg_orig)
    model_orig = policy_orig.diffusion

    cfg_scene = _make_future_mode_config(model="scene_only", include_ball_pos=True, disable_future_condition_gate=True)
    policy_scene = DiffusionPolicy(cfg_scene)
    model_scene = policy_scene.diffusion

    batch_orig = _make_future_mode_batch(include_future_state=False, include_ball_pos=False)
    batch_scene = _make_future_mode_batch(include_future_state=False, include_ball_pos=True)

    cond_orig = model_orig._prepare_global_conditioning(batch_orig)
    cond_scene = model_scene._prepare_global_conditioning(batch_scene)

    ball_pos = batch_scene["observation.ball_pos"][:, -1]
    expected_scene = model_scene.future_ball_pos_mlp(ball_pos)
    assert model_scene.future_ball_pos_gate is None
    assert torch.allclose(cond_scene[:, : cond_orig.shape[-1]], cond_orig, atol=1e-6)
    assert torch.allclose(cond_scene[:, -8:], expected_scene, atol=1e-6)


def test_robot_scene_appends_robot_and_scene_future_conditions():
    cfg = _make_future_mode_config(model="robot_scene", include_ball_pos=True)
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion
    batch = _make_future_mode_batch(include_future_state=True, include_ball_pos=True)

    cond = model._prepare_global_conditioning(batch)
    future_state = batch[OBS_STATE][:, 2]
    expected_robot = future_state * model.future_state_gate(future_state)
    ball_pos = batch["observation.ball_pos"][:, -1]
    expected_scene = model.future_ball_pos_mlp(ball_pos)
    expected_scene = expected_scene * model.future_ball_pos_gate(expected_scene)
    expected = torch.cat([expected_robot, expected_scene], dim=-1)
    assert torch.allclose(cond[:, -16:], expected, atol=1e-6)


def test_robot_only_online_history_uses_kalman_predicted_future_state():
    cfg = _make_future_mode_config(model="robot_only")
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    batch = _make_future_mode_batch(include_future_state=False, include_ball_pos=False)
    batch["timestamp"] = torch.tensor([[0.0, 0.1], [0.0, 0.1]], dtype=torch.float32)

    cond = model._prepare_global_conditioning(batch)
    predicted_future = model._predict_future_observation_with_kalman(batch[OBS_STATE], batch)
    expected_future = predicted_future * model.future_state_gate(predicted_future)
    assert torch.allclose(cond[:, -8:], expected_future, atol=1e-6)


def test_scene_only_online_history_uses_kalman_predicted_ball_pos():
    cfg = _make_future_mode_config(model="scene_only", include_ball_pos=True)
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion
    model.eval()

    batch = _make_future_mode_batch(include_future_state=False, include_ball_pos=False)
    batch["observation.ball_pos"] = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            [[0.5, -0.5, 1.0], [1.0, 0.0, 1.5]],
        ],
        dtype=torch.float32,
    )
    batch["timestamp"] = torch.tensor([[0.0, 0.1], [0.0, 0.1]], dtype=torch.float32)

    cond = model._prepare_global_conditioning(batch)
    predicted_ball = model._predict_future_observation_with_kalman(batch["observation.ball_pos"], batch)
    expected_scene = model.future_ball_pos_mlp(predicted_ball)
    expected_scene = expected_scene * model.future_ball_pos_gate(expected_scene)
    assert torch.allclose(cond[:, -8:], expected_scene, atol=1e-6)


def test_delay_random_training_samples_future_delta_and_eval_uses_fixed_delta():
    cfg = _make_future_mode_config(
        model="robot_scene",
        include_ball_pos=True,
        delay_random=True,
        future_condition_delta=2,
        delay_random_deltas=(2, 3, 4, 5, 6),
        # Always choose delta=5 (index=3) during training for deterministic assertions.
        delay_random_probs=(0.0, 0.0, 0.0, 1.0, 0.0),
    )
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion

    state_history = torch.tensor(
        [
            [[0.0] * 8, [1.0] * 8],
            [[2.0] * 8, [3.0] * 8],
        ],
        dtype=torch.float32,
    )
    state_future_candidates = torch.tensor(
        [
            [[10.0] * 8, [20.0] * 8, [30.0] * 8, [40.0] * 8, [50.0] * 8],
            [[11.0] * 8, [21.0] * 8, [31.0] * 8, [41.0] * 8, [51.0] * 8],
        ],
        dtype=torch.float32,
    )
    ball_future_candidates = torch.tensor(
        [
            [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]],
            [[0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9], [1.0, 1.0, 1.0], [1.1, 1.1, 1.1]],
        ],
        dtype=torch.float32,
    )
    batch = {
        OBS_STATE: torch.cat([state_history, state_future_candidates], dim=1),
        OBS_ENV_STATE: torch.zeros((2, 2, 4), dtype=torch.float32),
        "observation.ball_pos": ball_future_candidates,
    }

    # Training mode: choose delta=5 (candidate index 3).
    model.train()
    cond_train = model._prepare_global_conditioning(batch)
    expected_train_robot = state_future_candidates[:, 3] * model.future_state_gate(state_future_candidates[:, 3])
    expected_train_scene = model.future_ball_pos_mlp(ball_future_candidates[:, 3])
    expected_train_scene = expected_train_scene * model.future_ball_pos_gate(expected_train_scene)
    expected_train = torch.cat([expected_train_robot, expected_train_scene], dim=-1)
    assert torch.allclose(cond_train[:, -16:], expected_train, atol=1e-6)

    # Eval mode: use fixed future_condition_delta=2 (candidate index 0).
    model.eval()
    cond_eval = model._prepare_global_conditioning(batch)
    expected_eval_robot = state_future_candidates[:, 0] * model.future_state_gate(state_future_candidates[:, 0])
    expected_eval_scene = model.future_ball_pos_mlp(ball_future_candidates[:, 0])
    expected_eval_scene = expected_eval_scene * model.future_ball_pos_gate(expected_eval_scene)
    expected_eval = torch.cat([expected_eval_robot, expected_eval_scene], dim=-1)
    assert torch.allclose(cond_eval[:, -16:], expected_eval, atol=1e-6)


def test_training_raises_when_scene_branch_missing_dataset_ball_pos():
    cfg = _make_future_mode_config(model="scene_only", include_ball_pos=True)
    policy = DiffusionPolicy(cfg)
    model = policy.diffusion
    model.train()

    batch = _make_future_mode_batch(include_future_state=False, include_ball_pos=False)

    with pytest.raises(ValueError, match="Missing required dataset feature for training"):
        model._prepare_global_conditioning(batch)

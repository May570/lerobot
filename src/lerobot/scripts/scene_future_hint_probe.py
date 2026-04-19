#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval_realtime3 import _resolve_dyn_mini_default_plan_path
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.libero_compat import (
    apply_rename_map_to_preprocessor,
    resolve_libero_rename_map,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


CompactObs = dict[str, Tensor]


@dataclass
class EpisodeStartContext:
    start_abs_step: int
    start_state: np.ndarray
    obs_history: list[CompactObs]


@dataclass
class ProbeSnapshot:
    probe_step: int
    abs_step: int
    sim_state: np.ndarray
    obs_history: list[CompactObs]


@dataclass
class ProbeCandidate:
    probe_step: int
    score: float
    history_gap: float
    donor_gap: float
    min_wrong_horizon_gap: float
    mean_wrong_horizon_gap: float
    correct_hint_ball_pos: list[float]
    wrong_horizon_gaps: dict[int, float]


@dataclass
class EpisodeTrace:
    episode_index: int
    seed: int | None
    start_context: EpisodeStartContext
    ball_pos_tape: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    ball_grasp_count: int = 0
    probe_snapshots: dict[int, ProbeSnapshot] = field(default_factory=dict)


@dataclass
class EvalContext:
    args: argparse.Namespace
    vec_env: Any
    single_env: Any
    underlying_env: Any
    policy: Any
    env_preprocessor: Any
    env_postprocessor: Any
    preprocessor: Any
    postprocessor: Any
    image_feature_keys: list[str]
    future_ball_pos_key: str
    ball_pos_dim: int
    need_scene_ball_pos: bool
    rollout_fps: float | None
    zero_action: np.ndarray
    device: torch.device
    device_type: str
    use_amp: bool
    dense_progress_handles: dict[str, Any] | None = None


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe scene-only future-ball conditioning sensitivity on LIBERO dyn-mini policies."
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--policy.device", dest="policy_device", default="cuda")
    parser.add_argument("--policy.use_amp", dest="policy_use_amp", type=str2bool, default=True)
    parser.add_argument("--policy.n_action_steps", dest="policy_n_action_steps", type=int, default=None)
    parser.add_argument("--policy.num_inference_steps", dest="policy_num_inference_steps", type=int, default=None)
    parser.add_argument("--env.type", dest="env_type", default="libero")
    parser.add_argument("--env.task", dest="env_task", default="libero_dyn_mini")
    parser.add_argument("--env.task_ids", dest="env_task_ids", default="0")
    parser.add_argument("--env.fps", dest="env_fps", type=float, default=30.0)
    parser.add_argument("--env.episode_length", dest="env_episode_length", type=int, default=None)
    parser.add_argument("--env.control_mode", dest="env_control_mode", default="relative")
    parser.add_argument("--env.init_states", dest="env_init_states", type=str2bool, default=True)
    parser.add_argument("--env.init_plan_path", dest="env_init_plan_path", default=None)
    parser.add_argument("--env.init_plan_loop", dest="env_init_plan_loop", type=str2bool, default=True)
    parser.add_argument("--env.ball_grasp_eval_mode", dest="ball_grasp_eval_mode", default="strict")
    parser.add_argument(
        "--env.ball_grasp_strict_require_pad_contact",
        dest="ball_grasp_strict_require_pad_contact",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--env.ball_grasp_strict_lift_multiplier",
        dest="ball_grasp_strict_lift_multiplier",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--env.ball_grasp_strict_grip_center_max_dist",
        dest="ball_grasp_strict_grip_center_max_dist",
        type=float,
        default=0.045,
    )
    parser.add_argument("--libero_legacy_obs_compat", dest="libero_legacy_obs_compat", type=str2bool, default=False)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--episode_start_seed", dest="episode_start_seed", type=int, default=None)
    parser.add_argument("--reference_episodes", dest="reference_episodes", type=int, default=2)
    parser.add_argument("--reference_max_attempts", dest="reference_max_attempts", type=int, default=None)
    parser.add_argument("--reference_require_success", dest="reference_require_success", action="store_true")
    parser.add_argument("--reference_min_reward_sum", dest="reference_min_reward_sum", type=float, default=None)
    parser.add_argument("--reference_min_grasp_count", dest="reference_min_grasp_count", type=int, default=None)
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int, default=5)
    parser.add_argument("--reference_max_steps", dest="reference_max_steps", type=int, default=None)
    parser.add_argument("--probe_steps", dest="probe_steps", default=None)
    parser.add_argument("--auto_probes_per_episode", dest="auto_probes_per_episode", type=int, default=1)
    parser.add_argument(
        "--probe_selection_mode",
        dest="probe_selection_mode",
        choices=("motion", "uniform"),
        default="motion",
    )
    parser.add_argument("--probe_min_step", dest="probe_min_step", type=int, default=10)
    parser.add_argument("--probe_max_step", dest="probe_max_step", type=int, default=None)
    parser.add_argument("--probe_min_gap", dest="probe_min_gap", type=int, default=25)
    parser.add_argument("--probe_min_motion_score", dest="probe_min_motion_score", type=float, default=0.0)
    parser.add_argument("--alt_deltas", dest="alt_deltas", default="2,8")
    parser.add_argument("--compare_first_steps", dest="compare_first_steps", type=int, default=3)
    parser.add_argument("--branch_max_steps", dest="branch_max_steps", type=int, default=60)
    parser.add_argument("--skip_rollouts", dest="skip_rollouts", action="store_true")
    parser.add_argument("--include_history_only", dest="include_history_only", action="store_true")
    parser.add_argument("--full_episode_rollouts", dest="full_episode_rollouts", action="store_true")
    parser.add_argument("--full_episode_max_steps", dest="full_episode_max_steps", type=int, default=None)
    parser.add_argument("--output_dir", dest="output_dir", default=None)
    return parser


def make_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        path = Path(raw_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs/future_hint_probe") / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def clone_compact_obs(obs: CompactObs) -> CompactObs:
    return {key: value.detach().clone().to("cpu") for key, value in obs.items()}


def clone_obs_history(obs_history: list[CompactObs]) -> list[CompactObs]:
    return [clone_compact_obs(obs) for obs in obs_history]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def init_plugin_for_task(env_task: str) -> None:
    if env_task != "libero_dyn_mini":
        return
    try:
        import libero_dyn_mini_v1  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import `libero_dyn_mini_v1`. Make sure "
            "`PYTHONPATH` contains `/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/py`."
        ) from exc


def build_cli_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [f"--device={args.policy_device}", f"--use_amp={'true' if args.policy_use_amp else 'false'}"]
    if args.policy_n_action_steps is not None:
        overrides.append(f"--n_action_steps={int(args.policy_n_action_steps)}")
    if args.policy_num_inference_steps is not None:
        overrides.append(f"--num_inference_steps={int(args.policy_num_inference_steps)}")
    return overrides


def resolve_policy_and_env(args: argparse.Namespace) -> tuple[EvalContext, dict[str, Any]]:
    init_plugin_for_task(args.env_task)

    safe_device = get_safe_torch_device(args.policy_device, log=True)
    policy_cfg = PreTrainedConfig.from_pretrained(
        args.policy_path,
        cli_overrides=build_cli_overrides(args),
    )
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = str(safe_device)
    policy_cfg.use_amp = bool(args.policy_use_amp and policy_cfg.use_amp)

    task_ids = parse_int_list(args.env_task_ids)
    init_plan_path = args.env_init_plan_path
    if args.env_task == "libero_dyn_mini" and not init_plan_path:
        default_plan = _resolve_dyn_mini_default_plan_path()
        if default_plan is not None:
            init_plan_path = str(default_plan)
            logging.info("Using dyn-mini init plan: %s", init_plan_path)

    env_cfg = make_env_config(
        args.env_type,
        task=args.env_task,
        task_ids=task_ids if task_ids else None,
        fps=int(args.env_fps),
        episode_length=args.env_episode_length,
        control_mode=args.env_control_mode,
        init_states=bool(args.env_init_states),
        init_plan_path=init_plan_path,
        init_plan_loop=bool(args.env_init_plan_loop),
        ball_grasp_eval_mode=args.ball_grasp_eval_mode,
        ball_grasp_strict_require_pad_contact=bool(args.ball_grasp_strict_require_pad_contact),
        ball_grasp_strict_lift_multiplier=float(args.ball_grasp_strict_lift_multiplier),
        ball_grasp_strict_grip_center_max_dist=float(args.ball_grasp_strict_grip_center_max_dist),
    )

    policy_feature_keys = list(policy_cfg.input_features.keys()) if policy_cfg.input_features else []
    rename_map = resolve_libero_rename_map(
        enable_legacy_compat=args.libero_legacy_obs_compat,
        env_cfg=env_cfg,
        feature_keys=policy_feature_keys,
        user_rename_map=None,
    )

    envs = make_env(
        env_cfg,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=False,
    )
    vec_env = next(iter(next(iter(envs.values())).values()))
    single_env = vec_env.envs[0]
    underlying_env = single_env._env

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy_cfg.device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    apply_rename_map_to_preprocessor(preprocessor, rename_map)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    input_features = getattr(policy_cfg, "input_features", None) or {}
    future_ball_pos_key = str(getattr(policy_cfg, "future_ball_pos_key", "observation.ball_pos"))
    need_scene_ball_pos = str(getattr(policy_cfg, "model", "orig")) in {"scene_only", "robot_scene"}
    if not need_scene_ball_pos and future_ball_pos_key in input_features:
        need_scene_ball_pos = True
    ball_pos_dim = 3
    if future_ball_pos_key in input_features:
        feature = input_features[future_ball_pos_key]
        shape = getattr(feature, "shape", None)
        if shape:
            ball_pos_dim = int(np.prod(shape))

    single_action_space = getattr(vec_env, "single_action_space", None)
    if single_action_space is not None and getattr(single_action_space, "shape", None) is not None:
        zero_action = np.zeros(single_action_space.shape, dtype=np.float32)
    else:
        zero_action = np.zeros((7,), dtype=np.float32)

    ctx = EvalContext(
        args=args,
        vec_env=vec_env,
        single_env=single_env,
        underlying_env=underlying_env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        image_feature_keys=list(policy_cfg.image_features.keys()),
        future_ball_pos_key=future_ball_pos_key,
        ball_pos_dim=ball_pos_dim,
        need_scene_ball_pos=need_scene_ball_pos,
        rollout_fps=float(args.env_fps),
        zero_action=zero_action,
        device=safe_device,
        device_type="cuda" if safe_device.type == "cuda" else safe_device.type,
        use_amp=bool(policy_cfg.use_amp),
    )

    meta = {
        "policy_type": policy_cfg.type,
        "policy_model": getattr(policy_cfg, "model", None),
        "future_condition_delta": int(getattr(policy_cfg, "future_condition_delta", 0)),
        "future_condition_deltas": list(getattr(policy_cfg, "future_condition_deltas", [])),
        "n_obs_steps": int(policy_cfg.n_obs_steps),
        "n_action_steps": int(policy_cfg.n_action_steps),
        "num_inference_steps": int(policy_cfg.num_inference_steps)
        if getattr(policy_cfg, "num_inference_steps", None) is not None
        else None,
        "future_ball_pos_key": future_ball_pos_key,
        "ball_pos_dim": ball_pos_dim,
        "rename_map": rename_map,
        "init_plan_path": init_plan_path,
    }
    return ctx, meta


def compact_policy_observation(ctx: EvalContext, formatted_obs: dict[str, Any], abs_step: int) -> tuple[CompactObs, np.ndarray]:
    augmented_obs = maybe_batchify_formatted_obs(formatted_obs)
    if ctx.need_scene_ball_pos:
        if "ball_pos" in augmented_obs:
            raw_ball_pos = np.asarray(augmented_obs["ball_pos"], dtype=np.float32).reshape(-1)
        elif ctx.future_ball_pos_key in augmented_obs:
            raw_ball_pos = np.asarray(augmented_obs[ctx.future_ball_pos_key], dtype=np.float32).reshape(-1)
        else:
            raw = ctx.single_env.get_ball_pos()
            if raw is None:
                raise RuntimeError("Environment did not return `ball_pos` while scene conditioning is required.")
            raw_ball_pos = np.asarray(raw, dtype=np.float32).reshape(-1)
            augmented_obs["ball_pos"] = raw_ball_pos[: ctx.ball_pos_dim]
        raw_ball_pos = raw_ball_pos[: ctx.ball_pos_dim]
        augmented_obs["ball_pos"] = raw_ball_pos.astype(np.float32, copy=False)
    else:
        raw_ball_pos = np.zeros((ctx.ball_pos_dim,), dtype=np.float32)

    base_obs = preprocess_observation(augmented_obs)
    policy_obs = add_envs_task(ctx.vec_env, deepcopy(base_obs))
    policy_obs = ctx.env_preprocessor(policy_obs)
    policy_obs = ctx.preprocessor(policy_obs)

    if ctx.rollout_fps is not None and ctx.rollout_fps > 0 and OBS_STATE in policy_obs:
        obs_state = policy_obs[OBS_STATE]
        ts = torch.full(
            (int(obs_state.shape[0]),),
            float(abs_step) / float(ctx.rollout_fps),
            device=obs_state.device,
            dtype=obs_state.dtype,
        )
        policy_obs["timestamp"] = ts

    keep_keys = {OBS_STATE, ctx.future_ball_pos_key, "timestamp", *ctx.image_feature_keys}
    compact = {}
    for key in keep_keys:
        if key in policy_obs:
            compact[key] = policy_obs[key].detach().clone().to("cpu")
    return compact, raw_ball_pos


def maybe_batchify_formatted_obs(formatted_obs: dict[str, Any]) -> dict[str, Any]:
    needs_batch = False
    robot_state = formatted_obs.get("robot_state")
    if isinstance(robot_state, dict):
        try:
            quat = robot_state["eef"]["quat"]
            if isinstance(quat, np.ndarray) and quat.ndim == 1:
                needs_batch = True
        except Exception:  # noqa: BLE001
            needs_batch = False
    if not needs_batch:
        pixels = formatted_obs.get("pixels")
        if isinstance(pixels, dict) and len(pixels) > 0:
            first_image = next(iter(pixels.values()))
            if isinstance(first_image, np.ndarray) and first_image.ndim == 3:
                needs_batch = True

    if not needs_batch:
        return dict(formatted_obs)

    def _add_batch(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _add_batch(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return np.expand_dims(value, axis=0)
        return value

    return _add_batch(dict(formatted_obs))


def restore_snapshot_context(ctx: EvalContext, snapshot_state: np.ndarray, seed: int | None = None) -> None:
    ctx.vec_env.reset(seed=[seed] if seed is not None else None)
    ctx.underlying_env.set_init_state(np.asarray(snapshot_state, dtype=np.float64).copy())


def collect_episode_start_context(ctx: EvalContext, seed: int | None) -> EpisodeStartContext:
    observation, _ = ctx.vec_env.reset(seed=[seed] if seed is not None else None)
    obs_history: deque[CompactObs] = deque(maxlen=int(ctx.policy.config.n_obs_steps))
    initial_obs, _ = compact_policy_observation(ctx, observation, abs_step=0)
    obs_history.append(initial_obs)
    current_abs_step = 0

    for _ in range(max(0, int(ctx.args.warmup_steps))):
        raw_obs, _, done, _ = ctx.underlying_env.step(ctx.zero_action)
        current_abs_step += 1
        success = bool(ctx.underlying_env.check_success())
        if done or success:
            raise RuntimeError("Episode terminated during warmup. Try a different seed or fewer warmup steps.")
        formatted = ctx.single_env._format_raw_obs(raw_obs)
        compact_obs, _ = compact_policy_observation(ctx, formatted, abs_step=current_abs_step)
        obs_history.append(compact_obs)

    if len(obs_history) < int(ctx.policy.config.n_obs_steps):
        raise RuntimeError(
            f"Need {ctx.policy.config.n_obs_steps} observations after warmup, got {len(obs_history)}."
        )

    start_state = np.asarray(ctx.underlying_env.sim.get_state().flatten().copy(), dtype=np.float64)
    return EpisodeStartContext(
        start_abs_step=current_abs_step,
        start_state=start_state,
        obs_history=clone_obs_history(list(obs_history)),
    )


def make_noise(policy: Any, seed: int, batch_size: int = 1) -> Tensor:
    cfg = policy.config
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(
        size=(batch_size, int(cfg.horizon), int(cfg.action_feature.shape[0])),
        dtype=np.float32,
    )
    return torch.from_numpy(noise).to(device=device, dtype=dtype)


class BallPosNormalizer:
    def __init__(self, preprocessor: Any, future_ball_pos_key: str):
        self.preprocessor = preprocessor
        self.future_ball_pos_key = future_ball_pos_key
        self._cache: dict[tuple[float, ...], Tensor] = {}

    def __call__(self, raw_ball_pos: np.ndarray | list[float]) -> Tensor:
        arr = np.asarray(raw_ball_pos, dtype=np.float32).reshape(1, -1)
        key = tuple(float(x) for x in arr.reshape(-1).tolist())
        if key not in self._cache:
            processed = self.preprocessor({self.future_ball_pos_key: torch.from_numpy(arr.copy())})
            self._cache[key] = processed[self.future_ball_pos_key].detach().clone().to("cpu")
        return self._cache[key].detach().clone()


def build_policy_batch(
    ctx: EvalContext,
    obs_history: list[CompactObs],
    *,
    ball_pos_override: Tensor | None,
) -> dict[str, Tensor]:
    device = next(ctx.policy.parameters()).device
    batch: dict[str, Tensor] = {}

    state_seq = [obs[OBS_STATE].to(device=device) for obs in obs_history]
    batch[OBS_STATE] = torch.stack(state_seq, dim=1)

    if ctx.image_feature_keys:
        image_seq = []
        for obs in obs_history:
            stacked = torch.stack([obs[key].to(device=device) for key in ctx.image_feature_keys], dim=-4)
            image_seq.append(stacked)
        batch[OBS_IMAGES] = torch.stack(image_seq, dim=1)

    if "timestamp" in obs_history[-1]:
        batch["timestamp"] = torch.stack([obs["timestamp"].to(device=device) for obs in obs_history], dim=1)

    if ball_pos_override is None:
        batch[ctx.future_ball_pos_key] = torch.stack(
            [obs[ctx.future_ball_pos_key].to(device=device) for obs in obs_history],
            dim=1,
        )
    else:
        batch[ctx.future_ball_pos_key] = ball_pos_override.to(device=device).unsqueeze(1)

    return batch


def read_future_gate_debug(policy: Any) -> dict[str, float | None] | None:
    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None:
        return None
    getter = getattr(diffusion, "get_last_future_gate_debug", None)
    if not callable(getter):
        return None
    raw = getter(clear=True)
    if not isinstance(raw, dict):
        return None

    result: dict[str, float | None] = {}
    for key, values in raw.items():
        if isinstance(values, (list, tuple)) and len(values) > 0:
            try:
                value = float(values[0])
                result[key] = value if np.isfinite(value) else None
            except Exception:  # noqa: BLE001
                result[key] = None
        else:
            result[key] = None
    return result


def generate_action_chunk(
    ctx: EvalContext,
    obs_history: list[CompactObs],
    *,
    noise_seed: int,
    ball_pos_override: Tensor | None,
) -> tuple[Tensor, dict[str, float | None] | None]:
    batch = build_policy_batch(ctx, obs_history, ball_pos_override=ball_pos_override)
    noise = make_noise(ctx.policy, noise_seed)
    autocast_ctx = torch.autocast(device_type=ctx.device_type) if ctx.use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        actions = ctx.policy.diffusion.generate_actions(batch, noise=noise)

    gate_debug = read_future_gate_debug(ctx.policy)
    if actions.ndim != 3 or actions.shape[0] != 1:
        raise RuntimeError(f"Expected action chunk shape (1, T, D), got {tuple(actions.shape)}.")

    env_actions: list[Tensor] = []
    for chunk_idx in range(actions.shape[1]):
        action_t = actions[:, chunk_idx]
        action_t = ctx.postprocessor(action_t)
        action_transition = {ACTION: action_t}
        action_transition = ctx.env_postprocessor(action_transition)
        env_actions.append(action_transition[ACTION][0].detach().clone().to("cpu"))
    return torch.stack(env_actions, dim=0), gate_debug


def reference_noise_seed(base_seed: int, episode_index: int, policy_step: int) -> int:
    return int(base_seed + 1_000_000 + episode_index * 100_000 + policy_step)


def branch_noise_seed(base_seed: int, episode_index: int, probe_step: int, policy_step: int) -> int:
    return int(base_seed + 9_000_000 + episode_index * 100_000 + probe_step * 1_000 + policy_step)


def run_reference_episode(
    ctx: EvalContext,
    *,
    episode_index: int,
    seed: int | None,
    start_context: EpisodeStartContext,
    capture_probe_steps: set[int] | None = None,
) -> EpisodeTrace:
    restore_snapshot_context(ctx, start_context.start_state, seed=seed)
    obs_history: deque[CompactObs] = deque(clone_obs_history(start_context.obs_history), maxlen=int(ctx.policy.config.n_obs_steps))
    action_queue: deque[Tensor] = deque()
    trace = EpisodeTrace(
        episode_index=episode_index,
        seed=seed,
        start_context=start_context,
    )

    current_abs_step = int(start_context.start_abs_step)
    current_policy_step = 0
    max_abs_steps = int(ctx.single_env._max_episode_steps)
    max_policy_steps = (
        int(ctx.args.reference_max_steps)
        if ctx.args.reference_max_steps is not None
        else max(0, max_abs_steps - int(start_context.start_abs_step))
    )
    capture_probe_steps = capture_probe_steps or set()
    grasp_prev = bool(ctx.single_env.is_ball_grasped())
    grasp_events = 0

    while current_abs_step < max_abs_steps and current_policy_step < max_policy_steps:
        ball_pos = ctx.single_env.get_ball_pos()
        if ball_pos is None:
            raise RuntimeError("Failed to read ball position from env during reference rollout.")
        trace.ball_pos_tape.append(np.asarray(ball_pos, dtype=np.float32).reshape(-1)[: ctx.ball_pos_dim].tolist())

        if current_policy_step in capture_probe_steps:
            trace.probe_snapshots[current_policy_step] = ProbeSnapshot(
                probe_step=current_policy_step,
                abs_step=current_abs_step,
                sim_state=np.asarray(ctx.underlying_env.sim.get_state().flatten().copy(), dtype=np.float64),
                obs_history=clone_obs_history(list(obs_history)),
            )

        if len(action_queue) == 0:
            chunk, _ = generate_action_chunk(
                ctx,
                list(obs_history),
                noise_seed=reference_noise_seed(ctx.args.seed, episode_index, current_policy_step),
                ball_pos_override=None,
            )
            action_queue.extend(chunk)

        action = action_queue.popleft().numpy().astype(np.float32, copy=False)
        raw_obs, reward, done_env, _ = ctx.underlying_env.step(action)
        success = bool(ctx.underlying_env.check_success())
        done = bool(done_env or success or (current_abs_step + 1 >= max_abs_steps))
        trace.rewards.append(float(reward))
        trace.successes.append(success)
        trace.dones.append(done)

        grasp_now = bool(ctx.single_env.is_ball_grasped())
        if (not grasp_prev) and grasp_now:
            grasp_events += 1
        grasp_prev = grasp_now

        if done:
            break

        current_abs_step += 1
        current_policy_step += 1
        formatted = ctx.single_env._format_raw_obs(raw_obs)
        compact_obs, _ = compact_policy_observation(ctx, formatted, abs_step=current_abs_step)
        obs_history.append(compact_obs)

    trace.ball_grasp_count = int(grasp_events)
    return trace


def trace_metrics(trace: EpisodeTrace) -> dict[str, Any]:
    rewards = np.asarray(trace.rewards, dtype=np.float64)
    return {
        "len": int(len(trace.ball_pos_tape)),
        "sum_reward": float(np.sum(rewards)) if rewards.size > 0 else 0.0,
        "max_reward": float(np.max(rewards)) if rewards.size > 0 else 0.0,
        "success": bool(any(trace.successes)),
        "ball_grasp_count": int(trace.ball_grasp_count),
    }


def reference_trace_qualifies(trace: EpisodeTrace, args: argparse.Namespace) -> bool:
    metrics = trace_metrics(trace)
    if bool(args.reference_require_success) and not metrics["success"]:
        return False
    if args.reference_min_reward_sum is not None and metrics["sum_reward"] < float(args.reference_min_reward_sum):
        return False
    if args.reference_min_grasp_count is not None and metrics["ball_grasp_count"] < int(args.reference_min_grasp_count):
        return False
    return True


def reference_trace_sort_key(trace: EpisodeTrace) -> tuple[float, float, float, float]:
    metrics = trace_metrics(trace)
    return (
        1.0 if metrics["success"] else 0.0,
        float(metrics["sum_reward"]),
        float(metrics["ball_grasp_count"]),
        float(metrics["max_reward"]),
    )


def compute_l2(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    arr_a = np.asarray(a, dtype=np.float32).reshape(-1)
    arr_b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(arr_a - arr_b))


def compute_probe_candidates(
    *,
    trace: EpisodeTrace,
    donor_trace: EpisodeTrace,
    probe_min_step: int,
    probe_max_step: int | None,
    required_same_episode_delta: int,
    required_other_episode_delta: int,
    correct_delta: int,
    alt_deltas: list[int],
) -> list[ProbeCandidate]:
    valid_max = min(
        len(trace.ball_pos_tape) - 1 - required_same_episode_delta,
        len(donor_trace.ball_pos_tape) - 1 - required_other_episode_delta,
    )
    if probe_max_step is not None:
        valid_max = min(valid_max, int(probe_max_step))
    if valid_max < 0:
        return []

    lo = min(max(0, int(probe_min_step)), valid_max)
    candidates: list[ProbeCandidate] = []
    for probe_step in range(lo, valid_max + 1):
        current_ball = np.asarray(trace.ball_pos_tape[probe_step], dtype=np.float32)
        correct_ball = np.asarray(trace.ball_pos_tape[probe_step + correct_delta], dtype=np.float32)
        donor_ball = np.asarray(donor_trace.ball_pos_tape[probe_step + correct_delta], dtype=np.float32)
        history_gap = compute_l2(correct_ball, current_ball)
        donor_gap = compute_l2(correct_ball, donor_ball)

        wrong_horizon_gaps: dict[int, float] = {}
        for delta in alt_deltas:
            wrong_ball = np.asarray(trace.ball_pos_tape[probe_step + delta], dtype=np.float32)
            wrong_horizon_gaps[int(delta)] = compute_l2(correct_ball, wrong_ball)

        min_wrong_gap = min(wrong_horizon_gaps.values()) if wrong_horizon_gaps else 0.0
        mean_wrong_gap = float(np.mean(list(wrong_horizon_gaps.values()))) if wrong_horizon_gaps else 0.0
        conservative_score = min([history_gap, donor_gap, *wrong_horizon_gaps.values()]) if wrong_horizon_gaps else min(
            history_gap, donor_gap
        )
        candidates.append(
            ProbeCandidate(
                probe_step=int(probe_step),
                score=float(conservative_score),
                history_gap=float(history_gap),
                donor_gap=float(donor_gap),
                min_wrong_horizon_gap=float(min_wrong_gap),
                mean_wrong_horizon_gap=float(mean_wrong_gap),
                correct_hint_ball_pos=correct_ball.tolist(),
                wrong_horizon_gaps=wrong_horizon_gaps,
            )
        )
    return candidates


def select_top_probe_candidates(
    *,
    candidates: list[ProbeCandidate],
    num_probes: int,
    min_gap: int,
    min_motion_score: float,
) -> list[ProbeCandidate]:
    if len(candidates) == 0 or num_probes <= 0:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (item.score, item.history_gap, item.donor_gap, item.min_wrong_horizon_gap, item.probe_step),
        reverse=True,
    )

    def _greedy_pick(pool: list[ProbeCandidate]) -> list[ProbeCandidate]:
        selected: list[ProbeCandidate] = []
        for candidate in pool:
            if any(abs(candidate.probe_step - picked.probe_step) < min_gap for picked in selected):
                continue
            selected.append(candidate)
            if len(selected) >= num_probes:
                break
        return selected

    filtered = [candidate for candidate in sorted_candidates if candidate.score >= float(min_motion_score)]
    selected = _greedy_pick(filtered)
    if len(selected) < num_probes:
        selected = _greedy_pick(sorted_candidates)
    return sorted(selected, key=lambda item: item.probe_step)


def choose_probe_steps(
    *,
    requested_probe_steps: list[int],
    auto_probes_per_episode: int,
    probe_selection_mode: str,
    probe_min_step: int,
    probe_max_step: int | None,
    probe_min_gap: int,
    probe_min_motion_score: float,
    trace: EpisodeTrace,
    donor_trace: EpisodeTrace,
    correct_delta: int,
    alt_deltas: list[int],
    required_same_episode_delta: int,
    required_other_episode_delta: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    candidates = compute_probe_candidates(
        trace=trace,
        donor_trace=donor_trace,
        probe_min_step=probe_min_step,
        probe_max_step=probe_max_step,
        required_same_episode_delta=required_same_episode_delta,
        required_other_episode_delta=required_other_episode_delta,
        correct_delta=correct_delta,
        alt_deltas=alt_deltas,
    )
    candidate_by_step = {candidate.probe_step: candidate for candidate in candidates}
    candidate_max_step = max(candidate_by_step) if candidate_by_step else -1

    if requested_probe_steps:
        selected_candidates = [candidate_by_step[step] for step in requested_probe_steps if step in candidate_by_step]
        selected_candidates = sorted(selected_candidates, key=lambda item: item.probe_step)
        return [candidate.probe_step for candidate in selected_candidates], [
            probe_candidate_to_dict(candidate) for candidate in selected_candidates
        ]

    if len(candidates) == 0:
        return [], []

    if probe_selection_mode == "uniform":
        lo = min(max(0, int(probe_min_step)), candidate_max_step)
        hi = candidate_max_step
        if probe_max_step is not None:
            hi = min(hi, int(probe_max_step))
        if hi < lo:
            return [], []
        if auto_probes_per_episode <= 1:
            picked_steps = [int((lo + hi) // 2)]
        elif lo == hi:
            picked_steps = [lo]
        else:
            raw = np.linspace(lo, hi, num=auto_probes_per_episode)
            picked_steps = []
            seen = set()
            for value in raw:
                step = int(round(float(value)))
                step = max(lo, min(hi, step))
                if step not in seen and step in candidate_by_step:
                    seen.add(step)
                    picked_steps.append(step)
        selected_candidates = [candidate_by_step[step] for step in picked_steps if step in candidate_by_step]
        return [candidate.probe_step for candidate in selected_candidates], [
            probe_candidate_to_dict(candidate) for candidate in selected_candidates
        ]

    selected_candidates = select_top_probe_candidates(
        candidates=candidates,
        num_probes=auto_probes_per_episode,
        min_gap=max(1, int(probe_min_gap)),
        min_motion_score=float(probe_min_motion_score),
    )
    return [candidate.probe_step for candidate in selected_candidates], [
        probe_candidate_to_dict(candidate) for candidate in selected_candidates
    ]


def probe_candidate_to_dict(candidate: ProbeCandidate) -> dict[str, Any]:
    return {
        "probe_step": int(candidate.probe_step),
        "score": float(candidate.score),
        "history_gap": float(candidate.history_gap),
        "donor_gap": float(candidate.donor_gap),
        "min_wrong_horizon_gap": float(candidate.min_wrong_horizon_gap),
        "mean_wrong_horizon_gap": float(candidate.mean_wrong_horizon_gap),
        "correct_hint_ball_pos": list(candidate.correct_hint_ball_pos),
        "wrong_horizon_gaps": {str(delta): float(value) for delta, value in candidate.wrong_horizon_gaps.items()},
    }


def make_condition_specs(correct_delta: int, alt_deltas: list[int], include_history_only: bool) -> list[dict[str, Any]]:
    specs = [
        {"name": "correct", "mode": "trace", "source": "same_episode", "delta": correct_delta},
        {"name": "zeros", "mode": "zeros", "source": "none", "delta": None},
        {"name": "other_episode", "mode": "trace", "source": "other_episode", "delta": correct_delta},
    ]
    for delta in alt_deltas:
        specs.append(
            {
                "name": f"same_episode_tplus{delta}",
                "mode": "trace",
                "source": "same_episode",
                "delta": int(delta),
            }
        )
    if include_history_only:
        specs.append({"name": "history_only", "mode": "history_only", "source": "history", "delta": None})
    return specs


def query_ball_pos_from_trace(trace: EpisodeTrace, step_index: int) -> tuple[np.ndarray, int, bool]:
    if len(trace.ball_pos_tape) == 0:
        raise RuntimeError(f"Trace for episode {trace.episode_index} has no ball_pos entries.")
    used_index = max(0, min(int(step_index), len(trace.ball_pos_tape) - 1))
    clamped = used_index != int(step_index)
    value = np.asarray(trace.ball_pos_tape[used_index], dtype=np.float32)
    return value, used_index, clamped


def resolve_condition_hint(
    condition: dict[str, Any],
    *,
    episode_trace: EpisodeTrace,
    donor_trace: EpisodeTrace,
    policy_step: int,
) -> dict[str, Any]:
    mode = str(condition["mode"])
    if mode == "zeros":
        zeros = np.zeros((len(episode_trace.ball_pos_tape[0]),), dtype=np.float32)
        return {
            "mode": "zeros",
            "source_episode_index": None,
            "requested_step": None,
            "used_step": None,
            "clamped": False,
            "raw_ball_pos": zeros,
        }
    if mode == "history_only":
        return {
            "mode": "history_only",
            "source_episode_index": episode_trace.episode_index,
            "requested_step": None,
            "used_step": None,
            "clamped": False,
            "raw_ball_pos": None,
        }

    delta = int(condition["delta"])
    requested_step = int(policy_step + delta)
    source = episode_trace if condition["source"] == "same_episode" else donor_trace
    raw_ball_pos, used_step, clamped = query_ball_pos_from_trace(source, requested_step)
    return {
        "mode": "trace",
        "source_episode_index": source.episode_index,
        "requested_step": requested_step,
        "used_step": used_step,
        "clamped": clamped,
        "raw_ball_pos": raw_ball_pos,
    }


def compute_hint_metrics(
    *,
    correct_hint_raw: np.ndarray | list[float] | None,
    candidate_hint_raw: np.ndarray | list[float] | None,
) -> dict[str, Any]:
    if correct_hint_raw is None or candidate_hint_raw is None:
        return {
            "hint_l2_vs_correct": None,
            "hint_abs_mean_vs_correct": None,
        }
    correct = np.asarray(correct_hint_raw, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate_hint_raw, dtype=np.float32).reshape(-1)
    diff = candidate - correct
    return {
        "hint_l2_vs_correct": float(np.linalg.norm(diff)),
        "hint_abs_mean_vs_correct": float(np.mean(np.abs(diff))),
    }


def compute_action_diff_metrics(
    correct_chunk: Tensor,
    candidate_chunk: Tensor,
    compare_first_steps: int,
) -> dict[str, Any]:
    diff = candidate_chunk - correct_chunk
    per_step_l2 = torch.linalg.norm(diff, dim=-1)
    first_k = max(1, min(int(compare_first_steps), int(candidate_chunk.shape[0])))
    return {
        "chunk_l2_vs_correct": float(torch.linalg.norm(diff).item()),
        "per_step_l2_vs_correct": per_step_l2.tolist(),
        "first_k_step_l2_vs_correct": per_step_l2[:first_k].tolist(),
        "first_k_mean_l2_vs_correct": float(per_step_l2[:first_k].mean().item()),
        "chunk_abs_mean_vs_correct": float(diff.abs().mean().item()),
    }


def _find_body_id_contains(model: Any, fragments: list[str]) -> int | None:
    lowered_fragments = [str(fragment).lower() for fragment in fragments if str(fragment)]
    for gid in range(model.nbody):
        name = model.body_id2name(gid)
        if not name:
            continue
        lowered = str(name).lower()
        if any(fragment in lowered for fragment in lowered_fragments):
            return int(gid)
    return None


def resolve_dense_progress_handles(ctx: EvalContext) -> dict[str, Any]:
    if ctx.dense_progress_handles is not None:
        return ctx.dense_progress_handles

    single_env = ctx.single_env
    try:
        resolver = getattr(single_env, "_resolve_dyn_handles", None)
        if callable(resolver):
            resolver()
    except Exception:  # noqa: BLE001
        pass

    model = ctx.underlying_env.sim.model
    handles = {
        "ball_body_id": getattr(single_env, "_dyn_ball_body_id", None),
        "table_collision_geom_id": getattr(single_env, "_dyn_table_collision_geom_id", None),
        "grip_site_id": getattr(single_env, "_dyn_grip_site_id", None),
        "ball_radius": getattr(single_env, "_dyn_ball_radius", None),
        "bowl_body_id": _find_body_id_contains(model, ["fixed_bowl_1", "bowl_1", "fixed_bowl", "bowl"]),
    }
    ctx.dense_progress_handles = handles
    return handles


def read_dense_progress_state(ctx: EvalContext) -> dict[str, Any]:
    handles = resolve_dense_progress_handles(ctx)
    data = ctx.underlying_env.sim.data
    model = ctx.underlying_env.sim.model

    ball_pos = None
    ball_body_id = handles.get("ball_body_id")
    if ball_body_id is not None:
        try:
            ball_pos = np.asarray(data.body_xpos[int(ball_body_id)], dtype=np.float64).reshape(3).copy()
        except Exception:  # noqa: BLE001
            ball_pos = None

    bowl_pos = None
    bowl_body_id = handles.get("bowl_body_id")
    if bowl_body_id is not None:
        try:
            bowl_pos = np.asarray(data.body_xpos[int(bowl_body_id)], dtype=np.float64).reshape(3).copy()
        except Exception:  # noqa: BLE001
            bowl_pos = None

    grip_pos = None
    grip_site_id = handles.get("grip_site_id")
    if grip_site_id is not None:
        try:
            grip_pos = np.asarray(data.site_xpos[int(grip_site_id)], dtype=np.float64).reshape(3).copy()
        except Exception:  # noqa: BLE001
            grip_pos = None

    table_top_z = None
    table_collision_geom_id = handles.get("table_collision_geom_id")
    if table_collision_geom_id is not None:
        try:
            table_center_z = float(data.geom_xpos[int(table_collision_geom_id)][2])
            table_half_z = float(model.geom_size[int(table_collision_geom_id)][2])
            table_top_z = table_center_z + table_half_z
        except Exception:  # noqa: BLE001
            table_top_z = None

    ball_to_bowl_dist = None
    ball_to_bowl_xy_dist = None
    if ball_pos is not None and bowl_pos is not None:
        ball_to_bowl_dist = float(np.linalg.norm(ball_pos - bowl_pos))
        ball_to_bowl_xy_dist = float(np.linalg.norm(ball_pos[:2] - bowl_pos[:2]))

    gripper_to_ball_dist = None
    if ball_pos is not None and grip_pos is not None:
        gripper_to_ball_dist = float(np.linalg.norm(ball_pos - grip_pos))

    ball_height_above_table = None
    if ball_pos is not None and table_top_z is not None:
        ball_height_above_table = float(ball_pos[2] - table_top_z)

    return {
        "ball_pos": None if ball_pos is None else ball_pos.tolist(),
        "bowl_pos": None if bowl_pos is None else bowl_pos.tolist(),
        "grip_pos": None if grip_pos is None else grip_pos.tolist(),
        "table_top_z": table_top_z,
        "ball_to_bowl_dist": ball_to_bowl_dist,
        "ball_to_bowl_xy_dist": ball_to_bowl_xy_dist,
        "gripper_to_ball_dist": gripper_to_ball_dist,
        "ball_height_above_table": ball_height_above_table,
    }


def init_dense_progress_tracker(ctx: EvalContext, *, grasp_active: bool) -> dict[str, Any]:
    state = read_dense_progress_state(ctx)
    return {
        "initial_state": state,
        "final_state": state,
        "min_ball_to_bowl_dist": state["ball_to_bowl_dist"],
        "min_ball_to_bowl_xy_dist": state["ball_to_bowl_xy_dist"],
        "min_gripper_to_ball_dist": state["gripper_to_ball_dist"],
        "max_ball_height_above_table": state["ball_height_above_table"],
        "grasp_active_steps": int(bool(grasp_active)),
        "ever_grasped": bool(grasp_active),
    }


def update_dense_progress_tracker(tracker: dict[str, Any], state: dict[str, Any], *, grasp_active: bool) -> None:
    tracker["final_state"] = state
    for key in ["ball_to_bowl_dist", "ball_to_bowl_xy_dist", "gripper_to_ball_dist"]:
        current = state.get(key)
        min_key = f"min_{key}"
        existing = tracker.get(min_key)
        if current is not None and (existing is None or float(current) < float(existing)):
            tracker[min_key] = float(current)

    current_height = state.get("ball_height_above_table")
    existing_height = tracker.get("max_ball_height_above_table")
    if current_height is not None and (existing_height is None or float(current_height) > float(existing_height)):
        tracker["max_ball_height_above_table"] = float(current_height)

    if grasp_active:
        tracker["grasp_active_steps"] = int(tracker.get("grasp_active_steps", 0)) + 1
        tracker["ever_grasped"] = True


def finalize_dense_progress_tracker(tracker: dict[str, Any]) -> dict[str, Any]:
    initial = tracker.get("initial_state") or {}
    final = tracker.get("final_state") or {}

    def _improvement(start_key: str, end_value: float | None) -> float | None:
        start_value = initial.get(start_key)
        if start_value is None or end_value is None:
            return None
        return float(start_value) - float(end_value)

    return {
        "initial_state": initial,
        "final_state": final,
        "min_ball_to_bowl_dist": tracker.get("min_ball_to_bowl_dist"),
        "min_ball_to_bowl_xy_dist": tracker.get("min_ball_to_bowl_xy_dist"),
        "min_gripper_to_ball_dist": tracker.get("min_gripper_to_ball_dist"),
        "max_ball_height_above_table": tracker.get("max_ball_height_above_table"),
        "grasp_active_steps": int(tracker.get("grasp_active_steps", 0)),
        "ever_grasped": bool(tracker.get("ever_grasped", False)),
        "final_ball_to_bowl_dist": final.get("ball_to_bowl_dist"),
        "final_ball_to_bowl_xy_dist": final.get("ball_to_bowl_xy_dist"),
        "final_gripper_to_ball_dist": final.get("gripper_to_ball_dist"),
        "final_ball_height_above_table": final.get("ball_height_above_table"),
        "ball_to_bowl_dist_improvement": _improvement("ball_to_bowl_dist", final.get("ball_to_bowl_dist")),
        "ball_to_bowl_xy_dist_improvement": _improvement("ball_to_bowl_xy_dist", final.get("ball_to_bowl_xy_dist")),
        "best_ball_to_bowl_dist_improvement": _improvement("ball_to_bowl_dist", tracker.get("min_ball_to_bowl_dist")),
        "best_ball_to_bowl_xy_dist_improvement": _improvement(
            "ball_to_bowl_xy_dist", tracker.get("min_ball_to_bowl_xy_dist")
        ),
        "gripper_to_ball_dist_improvement": _improvement("gripper_to_ball_dist", final.get("gripper_to_ball_dist")),
        "best_gripper_to_ball_dist_improvement": _improvement(
            "gripper_to_ball_dist", tracker.get("min_gripper_to_ball_dist")
        ),
        "ball_height_gain": None
        if initial.get("ball_height_above_table") is None or tracker.get("max_ball_height_above_table") is None
        else float(tracker["max_ball_height_above_table"]) - float(initial["ball_height_above_table"]),
    }


def branch_rollout(
    ctx: EvalContext,
    *,
    episode_trace: EpisodeTrace,
    donor_trace: EpisodeTrace,
    probe_snapshot: ProbeSnapshot,
    condition: dict[str, Any],
    normalizer: BallPosNormalizer,
    initial_chunk: Tensor,
    initial_gate_debug: dict[str, float | None] | None,
    initial_hint_info: dict[str, Any],
) -> dict[str, Any]:
    restore_snapshot_context(ctx, probe_snapshot.sim_state, seed=episode_trace.seed)
    obs_history: deque[CompactObs] = deque(
        clone_obs_history(probe_snapshot.obs_history),
        maxlen=int(ctx.policy.config.n_obs_steps),
    )
    action_queue: deque[Tensor] = deque(initial_chunk)
    current_abs_step = int(probe_snapshot.abs_step)
    current_policy_step = int(probe_snapshot.probe_step)
    max_abs_steps = int(ctx.single_env._max_episode_steps)
    max_branch_steps = max(1, int(ctx.args.branch_max_steps))
    grasp_prev = bool(ctx.single_env.is_ball_grasped())
    replan_records = [
        {
            "policy_step": current_policy_step,
            "hint": {
                key: json_ready(value) for key, value in initial_hint_info.items() if key != "raw_ball_pos"
            },
            "future_gate_debug": json_ready(initial_gate_debug),
        }
    ]

    rewards: list[float] = []
    grasp_events = 0
    success = False
    done = False
    progress_tracker = init_dense_progress_tracker(ctx, grasp_active=grasp_prev)

    for local_step in range(max_branch_steps):
        if local_step > 0 and len(action_queue) == 0:
            hint_info = resolve_condition_hint(
                condition,
                episode_trace=episode_trace,
                donor_trace=donor_trace,
                policy_step=current_policy_step,
            )
            override = None
            if hint_info["raw_ball_pos"] is not None:
                override = normalizer(hint_info["raw_ball_pos"])
            chunk, gate_debug = generate_action_chunk(
                ctx,
                list(obs_history),
                noise_seed=branch_noise_seed(
                    ctx.args.seed, episode_trace.episode_index, probe_snapshot.probe_step, current_policy_step
                ),
                ball_pos_override=override,
            )
            action_queue.extend(chunk)
            replan_records.append(
                {
                    "policy_step": current_policy_step,
                    "hint": {key: json_ready(value) for key, value in hint_info.items() if key != "raw_ball_pos"},
                    "future_gate_debug": json_ready(gate_debug),
                }
            )

        action = action_queue.popleft().numpy().astype(np.float32, copy=False)
        raw_obs, reward, done_env, _ = ctx.underlying_env.step(action)
        success = bool(ctx.underlying_env.check_success())
        done = bool(done_env or success or (current_abs_step + 1 >= max_abs_steps))
        rewards.append(float(reward))

        grasp_now = bool(ctx.single_env.is_ball_grasped())
        if (not grasp_prev) and grasp_now:
            grasp_events += 1
        update_dense_progress_tracker(
            progress_tracker,
            read_dense_progress_state(ctx),
            grasp_active=grasp_now,
        )
        grasp_prev = grasp_now

        if done:
            break

        current_abs_step += 1
        current_policy_step += 1
        formatted = ctx.single_env._format_raw_obs(raw_obs)
        compact_obs, _ = compact_policy_observation(ctx, formatted, abs_step=current_abs_step)
        obs_history.append(compact_obs)

    return {
        "num_steps": len(rewards),
        "sum_reward": float(np.sum(rewards, dtype=np.float64)),
        "avg_reward": float(np.mean(rewards, dtype=np.float64)) if rewards else 0.0,
        "success": bool(success),
        "done": bool(done),
        "ball_grasp_count": int(grasp_events),
        "progress": finalize_dense_progress_tracker(progress_tracker),
        "replans": replan_records,
    }


def full_episode_rollout(
    ctx: EvalContext,
    *,
    episode_trace: EpisodeTrace,
    donor_trace: EpisodeTrace,
    condition: dict[str, Any],
    normalizer: BallPosNormalizer,
) -> dict[str, Any]:
    restore_snapshot_context(ctx, episode_trace.start_context.start_state, seed=episode_trace.seed)
    obs_history: deque[CompactObs] = deque(
        clone_obs_history(episode_trace.start_context.obs_history),
        maxlen=int(ctx.policy.config.n_obs_steps),
    )
    action_queue: deque[Tensor] = deque()
    current_abs_step = int(episode_trace.start_context.start_abs_step)
    current_policy_step = 0
    max_abs_steps = int(ctx.single_env._max_episode_steps)
    max_policy_steps = (
        int(ctx.args.full_episode_max_steps)
        if ctx.args.full_episode_max_steps is not None
        else max(0, max_abs_steps - int(episode_trace.start_context.start_abs_step))
    )
    grasp_prev = bool(ctx.single_env.is_ball_grasped())

    rewards: list[float] = []
    grasp_events = 0
    success = False
    done = False
    gate_means: list[float] = []
    initial_hint_info: dict[str, Any] | None = None
    initial_gate_debug: dict[str, float | None] | None = None
    initial_chunk: Tensor | None = None
    num_replans = 0
    progress_tracker = init_dense_progress_tracker(ctx, grasp_active=grasp_prev)

    while current_abs_step < max_abs_steps and current_policy_step < max_policy_steps:
        if len(action_queue) == 0:
            hint_info = resolve_condition_hint(
                condition,
                episode_trace=episode_trace,
                donor_trace=donor_trace,
                policy_step=current_policy_step,
            )
            override = None
            if hint_info["raw_ball_pos"] is not None:
                override = normalizer(hint_info["raw_ball_pos"])
            chunk, gate_debug = generate_action_chunk(
                ctx,
                list(obs_history),
                noise_seed=reference_noise_seed(ctx.args.seed, episode_trace.episode_index, current_policy_step),
                ball_pos_override=override,
            )
            action_queue.extend(chunk)
            num_replans += 1
            gate_value = None if gate_debug is None else gate_debug.get("future_ball_gate_mean")
            if gate_value is not None:
                gate_means.append(float(gate_value))
            if initial_hint_info is None:
                initial_hint_info = hint_info
                initial_gate_debug = gate_debug
                initial_chunk = chunk

        action = action_queue.popleft().numpy().astype(np.float32, copy=False)
        raw_obs, reward, done_env, _ = ctx.underlying_env.step(action)
        success = bool(ctx.underlying_env.check_success())
        done = bool(done_env or success or (current_abs_step + 1 >= max_abs_steps))
        rewards.append(float(reward))

        grasp_now = bool(ctx.single_env.is_ball_grasped())
        if (not grasp_prev) and grasp_now:
            grasp_events += 1
        update_dense_progress_tracker(
            progress_tracker,
            read_dense_progress_state(ctx),
            grasp_active=grasp_now,
        )
        grasp_prev = grasp_now

        if done:
            break

        current_abs_step += 1
        current_policy_step += 1
        formatted = ctx.single_env._format_raw_obs(raw_obs)
        compact_obs, _ = compact_policy_observation(ctx, formatted, abs_step=current_abs_step)
        obs_history.append(compact_obs)

    return {
        "num_steps": len(rewards),
        "sum_reward": float(np.sum(rewards, dtype=np.float64)),
        "avg_reward": float(np.mean(rewards, dtype=np.float64)) if rewards else 0.0,
        "success": bool(success),
        "done": bool(done),
        "ball_grasp_count": int(grasp_events),
        "num_replans": int(num_replans),
        "mean_future_ball_gate_mean": float(np.mean(gate_means)) if gate_means else None,
        "progress": finalize_dense_progress_tracker(progress_tracker),
        "initial_hint": None
        if initial_hint_info is None
        else {key: value for key, value in initial_hint_info.items() if key != "raw_ball_pos"},
        "initial_raw_hint_ball_pos": None if initial_hint_info is None else initial_hint_info["raw_ball_pos"],
        "initial_future_gate_debug": initial_gate_debug,
        "initial_action_chunk": initial_chunk,
    }


def summarize_progress_metrics(progress_records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "final_ball_to_bowl_dist",
        "final_ball_to_bowl_xy_dist",
        "min_ball_to_bowl_dist",
        "min_ball_to_bowl_xy_dist",
        "final_gripper_to_ball_dist",
        "min_gripper_to_ball_dist",
        "ball_to_bowl_dist_improvement",
        "ball_to_bowl_xy_dist_improvement",
        "best_ball_to_bowl_dist_improvement",
        "best_ball_to_bowl_xy_dist_improvement",
        "gripper_to_ball_dist_improvement",
        "best_gripper_to_ball_dist_improvement",
        "ball_height_gain",
        "grasp_active_steps",
    ]
    summary: dict[str, Any] = {}
    for key in metric_keys:
        values = [float(progress[key]) for progress in progress_records if progress.get(key) is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None

    ever_grasped = [bool(progress.get("ever_grasped", False)) for progress in progress_records]
    summary["ever_grasped_rate"] = float(np.mean(ever_grasped)) if ever_grasped else None
    return summary


def summarize_results(probe_results: list[dict[str, Any]], condition_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_probes": len(probe_results),
        "probe_quality": summarize_probe_quality(probe_results),
        "conditions": {},
    }

    for condition_name in condition_names:
        hint_l2 = []
        hint_abs_mean = []
        action_chunk_l2 = []
        first_k_mean_l2 = []
        gate_means = []
        rollout_sum_rewards = []
        rollout_successes = []
        rollout_reward_deltas = []
        rollout_progress_records = []

        for probe in probe_results:
            condition_record = probe["conditions"].get(condition_name)
            if condition_record is None:
                continue
            hint_metrics = condition_record.get("hint_metrics") or {}
            hint_l2_value = hint_metrics.get("hint_l2_vs_correct")
            hint_abs_mean_value = hint_metrics.get("hint_abs_mean_vs_correct")
            if hint_l2_value is not None:
                hint_l2.append(float(hint_l2_value))
            if hint_abs_mean_value is not None:
                hint_abs_mean.append(float(hint_abs_mean_value))
            action_metrics = condition_record["action_metrics"]
            action_chunk_l2.append(float(action_metrics["chunk_l2_vs_correct"]))
            first_k_mean_l2.append(float(action_metrics["first_k_mean_l2_vs_correct"]))
            gate_debug = condition_record.get("future_gate_debug") or {}
            gate_value = gate_debug.get("future_ball_gate_mean")
            if gate_value is not None:
                gate_means.append(float(gate_value))

            rollout_record = condition_record.get("rollout")
            if rollout_record is not None:
                rollout_sum_rewards.append(float(rollout_record["sum_reward"]))
                rollout_successes.append(bool(rollout_record["success"]))
                correct_reward = float(probe["conditions"]["correct"]["rollout"]["sum_reward"])
                rollout_reward_deltas.append(float(rollout_record["sum_reward"]) - correct_reward)
                progress = rollout_record.get("progress")
                if isinstance(progress, dict):
                    rollout_progress_records.append(progress)

        condition_summary = {
            "mean_hint_l2_vs_correct": float(np.mean(hint_l2)) if hint_l2 else None,
            "mean_hint_abs_mean_vs_correct": float(np.mean(hint_abs_mean)) if hint_abs_mean else None,
            "mean_chunk_l2_vs_correct": float(np.mean(action_chunk_l2)) if action_chunk_l2 else None,
            "mean_first_k_l2_vs_correct": float(np.mean(first_k_mean_l2)) if first_k_mean_l2 else None,
            "mean_future_ball_gate_mean": float(np.mean(gate_means)) if gate_means else None,
            "mean_rollout_sum_reward": float(np.mean(rollout_sum_rewards)) if rollout_sum_rewards else None,
            "mean_rollout_reward_delta_vs_correct": float(np.mean(rollout_reward_deltas))
            if rollout_reward_deltas
            else None,
            "rollout_success_rate": float(np.mean(rollout_successes)) if rollout_successes else None,
        }
        condition_summary.update(summarize_progress_metrics(rollout_progress_records))
        summary["conditions"][condition_name] = condition_summary
    summary["future_effectiveness"] = summarize_future_effectiveness(summary["conditions"])
    return summary


def summarize_full_episode_results(full_episode_results: list[dict[str, Any]], condition_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_episodes": len(full_episode_results),
        "conditions": {},
    }

    for condition_name in condition_names:
        initial_hint_l2 = []
        initial_chunk_l2 = []
        sum_rewards = []
        reward_deltas = []
        successes = []
        grasp_counts = []
        num_steps = []
        gate_means = []
        progress_records = []

        for episode in full_episode_results:
            condition_record = episode["conditions"].get(condition_name)
            if condition_record is None:
                continue
            hint_metrics = condition_record.get("initial_hint_metrics") or {}
            hint_l2_value = hint_metrics.get("hint_l2_vs_correct")
            if hint_l2_value is not None:
                initial_hint_l2.append(float(hint_l2_value))

            action_metrics = condition_record.get("initial_action_metrics")
            if action_metrics is not None:
                initial_chunk_l2.append(float(action_metrics["chunk_l2_vs_correct"]))

            rollout = condition_record["rollout"]
            sum_rewards.append(float(rollout["sum_reward"]))
            successes.append(bool(rollout["success"]))
            grasp_counts.append(int(rollout["ball_grasp_count"]))
            num_steps.append(int(rollout["num_steps"]))
            gate_value = rollout.get("mean_future_ball_gate_mean")
            if gate_value is not None:
                gate_means.append(float(gate_value))
            progress = rollout.get("progress")
            if isinstance(progress, dict):
                progress_records.append(progress)

            correct_reward = float(episode["conditions"]["correct"]["rollout"]["sum_reward"])
            reward_deltas.append(float(rollout["sum_reward"]) - correct_reward)

        condition_summary = {
            "mean_initial_hint_l2_vs_correct": float(np.mean(initial_hint_l2)) if initial_hint_l2 else None,
            "mean_initial_chunk_l2_vs_correct": float(np.mean(initial_chunk_l2)) if initial_chunk_l2 else None,
            "mean_sum_reward": float(np.mean(sum_rewards)) if sum_rewards else None,
            "mean_reward_delta_vs_correct": float(np.mean(reward_deltas)) if reward_deltas else None,
            "success_rate": float(np.mean(successes)) if successes else None,
            "mean_ball_grasp_count": float(np.mean(grasp_counts)) if grasp_counts else None,
            "mean_num_steps": float(np.mean(num_steps)) if num_steps else None,
            "mean_future_ball_gate_mean": float(np.mean(gate_means)) if gate_means else None,
        }
        condition_summary.update(summarize_progress_metrics(progress_records))
        summary["conditions"][condition_name] = condition_summary

    return summary


def summarize_probe_quality(probe_results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = []
    history_gaps = []
    donor_gaps = []
    min_wrong_gaps = []
    mean_wrong_gaps = []
    for probe in probe_results:
        selection = probe.get("probe_selection")
        if not isinstance(selection, dict):
            continue
        for key, bucket in [
            ("score", scores),
            ("history_gap", history_gaps),
            ("donor_gap", donor_gaps),
            ("min_wrong_horizon_gap", min_wrong_gaps),
            ("mean_wrong_horizon_gap", mean_wrong_gaps),
        ]:
            value = selection.get(key)
            if value is not None:
                bucket.append(float(value))
    return {
        "mean_score": float(np.mean(scores)) if scores else None,
        "mean_history_gap": float(np.mean(history_gaps)) if history_gaps else None,
        "mean_donor_gap": float(np.mean(donor_gaps)) if donor_gaps else None,
        "mean_min_wrong_horizon_gap": float(np.mean(min_wrong_gaps)) if min_wrong_gaps else None,
        "mean_mean_wrong_horizon_gap": float(np.mean(mean_wrong_gaps)) if mean_wrong_gaps else None,
    }


def summarize_future_effectiveness(condition_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def _safe_ratio(numerator_key: str, denominator_key: str) -> float | None:
        numerator = condition_summary.get(numerator_key, {}).get("mean_chunk_l2_vs_correct")
        denominator = condition_summary.get(denominator_key, {}).get("mean_chunk_l2_vs_correct")
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator) / float(denominator)

    return {
        "history_only_vs_zero_action_gap_ratio": _safe_ratio("history_only", "zeros"),
        "other_episode_vs_zero_action_gap_ratio": _safe_ratio("other_episode", "zeros"),
        "other_episode_vs_history_only_action_gap_ratio": _safe_ratio("other_episode", "history_only"),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = make_output_dir(args.output_dir)
    init_logging(log_file=output_dir / "run.log")
    set_seed(args.seed)

    requested_probe_steps = parse_int_list(args.probe_steps)
    alt_deltas = parse_int_list(args.alt_deltas)
    if args.reference_episodes < 2:
        raise ValueError("`--reference_episodes` must be at least 2 so `other_episode` has a donor trace.")

    ctx = None
    meta: dict[str, Any] = {}
    try:
        ctx, meta = resolve_policy_and_env(args)
        correct_delta = int(meta["future_condition_delta"])
        alt_deltas = [delta for delta in alt_deltas if delta > 0 and delta != correct_delta]
        if len(alt_deltas) == 0:
            raise ValueError("`--alt_deltas` must provide at least one positive delta different from correct delta.")

        logging.info("Loaded policy model=%s future_delta=%d n_action_steps=%d num_inference_steps=%s",
                     meta["policy_model"], correct_delta, meta["n_action_steps"], meta["num_inference_steps"])

        reference_traces: list[EpisodeTrace] = []
        all_reference_traces: list[EpisodeTrace] = []
        reference_attempts: list[dict[str, Any]] = []
        has_reference_filters = bool(args.reference_require_success) or (
            args.reference_min_reward_sum is not None
        ) or (args.reference_min_grasp_count is not None)
        default_reference_max_attempts = max(
            int(args.reference_episodes),
            int(args.reference_episodes) * (8 if has_reference_filters else 1),
        )
        reference_max_attempts = (
            int(args.reference_max_attempts)
            if args.reference_max_attempts is not None
            else default_reference_max_attempts
        )
        for attempt_index in range(reference_max_attempts):
            if len(reference_traces) >= int(args.reference_episodes):
                break

            episode_seed = None if args.episode_start_seed is None else int(args.episode_start_seed + attempt_index)
            start_context = collect_episode_start_context(ctx, episode_seed)
            trace = run_reference_episode(
                ctx,
                episode_index=attempt_index,
                seed=episode_seed,
                start_context=start_context,
                capture_probe_steps=set(),
            )
            all_reference_traces.append(trace)
            metrics = trace_metrics(trace)
            qualifies = reference_trace_qualifies(trace, args)
            logging.info(
                "Collected reference attempt=%d accepted=%s next_episode_index=%d seed=%s len=%d success=%s sum_reward=%.3f grasp_count=%d",
                attempt_index,
                qualifies,
                len(reference_traces),
                episode_seed,
                metrics["len"],
                metrics["success"],
                metrics["sum_reward"],
                metrics["ball_grasp_count"],
            )
            reference_attempts.append(
                {
                    "attempt_index": attempt_index,
                    "seed": episode_seed,
                    "accepted": qualifies,
                    "accepted_episode_index": len(reference_traces) if qualifies else None,
                    **metrics,
                }
            )
            if qualifies:
                reference_traces.append(trace)

        if len(reference_traces) < int(args.reference_episodes):
            if len(all_reference_traces) < int(args.reference_episodes):
                raise RuntimeError(
                    "Not enough reference episodes were collected. "
                    f"Needed {int(args.reference_episodes)}, got {len(all_reference_traces)} after {reference_max_attempts} attempts."
                )
            logging.warning(
                "Not enough reference episodes matched filters; falling back to best collected attempts. "
                "matched=%d requested=%d attempts=%d filters=(require_success=%s min_reward_sum=%s min_grasp_count=%s)",
                len(reference_traces),
                int(args.reference_episodes),
                reference_max_attempts,
                bool(args.reference_require_success),
                args.reference_min_reward_sum,
                args.reference_min_grasp_count,
            )
            fallback_traces = sorted(all_reference_traces, key=reference_trace_sort_key, reverse=True)[
                : int(args.reference_episodes)
            ]
            fallback_traces = sorted(fallback_traces, key=lambda item: int(item.episode_index))
            reference_traces = fallback_traces

        selected_probe_steps_by_episode: dict[int, list[int]] = {}
        selected_probe_details_by_episode: dict[int, list[dict[str, Any]]] = {}
        required_same_episode_delta = max([correct_delta, *alt_deltas])
        required_other_episode_delta = correct_delta
        for episode_index, trace in enumerate(reference_traces):
            donor_trace = reference_traces[(episode_index + 1) % len(reference_traces)]
            probe_steps, probe_step_details = choose_probe_steps(
                requested_probe_steps=requested_probe_steps,
                auto_probes_per_episode=int(args.auto_probes_per_episode),
                probe_selection_mode=str(args.probe_selection_mode),
                probe_min_step=int(args.probe_min_step),
                probe_max_step=args.probe_max_step,
                probe_min_gap=int(args.probe_min_gap),
                probe_min_motion_score=float(args.probe_min_motion_score),
                trace=trace,
                donor_trace=donor_trace,
                correct_delta=correct_delta,
                alt_deltas=alt_deltas,
                required_same_episode_delta=required_same_episode_delta,
                required_other_episode_delta=required_other_episode_delta,
            )
            selected_probe_steps_by_episode[episode_index] = probe_steps
            selected_probe_details_by_episode[episode_index] = probe_step_details
            logging.info(
                "Episode=%d donor=%d selected probe steps=%s details=%s",
                episode_index,
                donor_trace.episode_index,
                probe_steps,
                json_ready(probe_step_details),
            )

        for episode_index, probe_steps in selected_probe_steps_by_episode.items():
            if len(probe_steps) == 0:
                continue
            trace = reference_traces[episode_index]
            replay = run_reference_episode(
                ctx,
                episode_index=trace.episode_index,
                seed=trace.seed,
                start_context=trace.start_context,
                capture_probe_steps=set(probe_steps),
            )
            reference_traces[episode_index].probe_snapshots = replay.probe_snapshots

        normalizer = BallPosNormalizer(ctx.preprocessor, ctx.future_ball_pos_key)
        condition_specs = make_condition_specs(correct_delta, alt_deltas, args.include_history_only)
        condition_names = [spec["name"] for spec in condition_specs]
        probe_results: list[dict[str, Any]] = []
        full_episode_results: list[dict[str, Any]] = []
        total_probes = sum(len(steps) for steps in selected_probe_steps_by_episode.values())
        completed_probes = 0
        all_probe_start_time = time.perf_counter()

        for episode_index, trace in enumerate(reference_traces):
            donor_trace = reference_traces[(episode_index + 1) % len(reference_traces)]
            for probe_step in selected_probe_steps_by_episode.get(episode_index, []):
                probe_index = completed_probes + 1
                probe_snapshot = trace.probe_snapshots.get(probe_step)
                if probe_snapshot is None:
                    logging.warning("Missing replay snapshot for episode=%d probe_step=%d, skipping.", episode_index, probe_step)
                    continue
                probe_selection_record = None
                for detail in selected_probe_details_by_episode.get(episode_index, []):
                    if int(detail["probe_step"]) == int(probe_step):
                        probe_selection_record = detail
                        break

                probe_start_time = time.perf_counter()
                logging.info(
                    "Running probe %d/%d | episode=%d donor=%d probe_step=%d abs_step=%d",
                    probe_index,
                    total_probes,
                    trace.episode_index,
                    donor_trace.episode_index,
                    probe_step,
                    probe_snapshot.abs_step,
                )

                per_condition: dict[str, Any] = {}
                correct_hint_info = resolve_condition_hint(
                    condition_specs[0],
                    episode_trace=trace,
                    donor_trace=donor_trace,
                    policy_step=probe_step,
                )
                correct_override = normalizer(correct_hint_info["raw_ball_pos"])
                correct_chunk, correct_gate = generate_action_chunk(
                    ctx,
                    probe_snapshot.obs_history,
                    noise_seed=branch_noise_seed(args.seed, trace.episode_index, probe_step, probe_step),
                    ball_pos_override=correct_override,
                )

                for condition in condition_specs:
                    condition_start_time = time.perf_counter()
                    hint_info = resolve_condition_hint(
                        condition,
                        episode_trace=trace,
                        donor_trace=donor_trace,
                        policy_step=probe_step,
                    )
                    override = None
                    if hint_info["raw_ball_pos"] is not None:
                        override = normalizer(hint_info["raw_ball_pos"])

                    if condition["name"] == "correct":
                        chunk = correct_chunk
                        gate_debug = correct_gate
                    else:
                        chunk, gate_debug = generate_action_chunk(
                            ctx,
                            probe_snapshot.obs_history,
                            noise_seed=branch_noise_seed(args.seed, trace.episode_index, probe_step, probe_step),
                            ball_pos_override=override,
                        )

                    action_metrics = compute_action_diff_metrics(
                        correct_chunk=correct_chunk,
                        candidate_chunk=chunk,
                        compare_first_steps=int(args.compare_first_steps),
                    )
                    hint_metrics = compute_hint_metrics(
                        correct_hint_raw=correct_hint_info["raw_ball_pos"],
                        candidate_hint_raw=hint_info["raw_ball_pos"],
                    )
                    rollout_record = None
                    if not args.skip_rollouts:
                        rollout_record = branch_rollout(
                            ctx,
                            episode_trace=trace,
                            donor_trace=donor_trace,
                            probe_snapshot=probe_snapshot,
                            condition=condition,
                            normalizer=normalizer,
                            initial_chunk=chunk,
                            initial_gate_debug=gate_debug,
                            initial_hint_info=hint_info,
                        )

                    per_condition[condition["name"]] = {
                        "hint": {key: json_ready(value) for key, value in hint_info.items() if key != "raw_ball_pos"},
                        "raw_hint_ball_pos": json_ready(hint_info["raw_ball_pos"]),
                        "hint_metrics": json_ready(hint_metrics),
                        "future_gate_debug": json_ready(gate_debug),
                        "action_chunk": json_ready(chunk),
                        "action_metrics": json_ready(action_metrics),
                        "rollout": json_ready(rollout_record),
                    }
                    logging.info(
                        "Probe %d/%d condition=%s finished | chunk_l2=%.6f first_k_mean_l2=%.6f hint_l2=%s rollout_reward=%s success=%s best_ball_to_bowl_xy_impr=%s elapsed=%.2fs",
                        probe_index,
                        total_probes,
                        condition["name"],
                        float(action_metrics["chunk_l2_vs_correct"]),
                        float(action_metrics["first_k_mean_l2_vs_correct"]),
                        hint_metrics["hint_l2_vs_correct"],
                        None if rollout_record is None else rollout_record["sum_reward"],
                        None if rollout_record is None else rollout_record["success"],
                        None
                        if rollout_record is None
                        else (rollout_record.get("progress") or {}).get("best_ball_to_bowl_xy_dist_improvement"),
                        time.perf_counter() - condition_start_time,
                    )

                probe_results.append(
                    {
                        "episode_index": trace.episode_index,
                        "donor_episode_index": donor_trace.episode_index,
                        "probe_step": probe_step,
                        "abs_probe_step": probe_snapshot.abs_step,
                        "probe_selection": probe_selection_record,
                        "conditions": per_condition,
                    }
                )
                completed_probes += 1
                elapsed = time.perf_counter() - probe_start_time
                overall_elapsed = time.perf_counter() - all_probe_start_time
                avg_probe_time = overall_elapsed / max(1, completed_probes)
                remaining_probes = max(0, total_probes - completed_probes)
                eta_seconds = remaining_probes * avg_probe_time
                logging.info(
                    "Finished probe %d/%d | episode=%d probe_step=%d elapsed=%.2fs overall_elapsed=%.2fs eta=%.2fs",
                    completed_probes,
                    total_probes,
                    trace.episode_index,
                    probe_step,
                    elapsed,
                    overall_elapsed,
                    eta_seconds,
                )

        if args.full_episode_rollouts:
            total_full_episodes = len(reference_traces)
            for episode_index, trace in enumerate(reference_traces):
                donor_trace = reference_traces[(episode_index + 1) % len(reference_traces)]
                logging.info(
                    "Running full-episode comparison %d/%d | episode=%d donor=%d",
                    episode_index + 1,
                    total_full_episodes,
                    trace.episode_index,
                    donor_trace.episode_index,
                )
                per_condition: dict[str, Any] = {}
                correct_condition = condition_specs[0]
                correct_rollout_start_time = time.perf_counter()
                correct_rollout_record = full_episode_rollout(
                    ctx,
                    episode_trace=trace,
                    donor_trace=donor_trace,
                    condition=correct_condition,
                    normalizer=normalizer,
                )
                correct_initial_chunk = correct_rollout_record.get("initial_action_chunk")
                correct_initial_hint_raw = correct_rollout_record.get("initial_raw_hint_ball_pos")
                correct_initial_action_metrics = None
                if isinstance(correct_initial_chunk, torch.Tensor):
                    correct_initial_action_metrics = compute_action_diff_metrics(
                        correct_chunk=correct_initial_chunk,
                        candidate_chunk=correct_initial_chunk,
                        compare_first_steps=int(args.compare_first_steps),
                    )
                per_condition[correct_condition["name"]] = {
                    "initial_hint": None
                    if correct_rollout_record["initial_hint"] is None
                    else {
                        key: json_ready(value) for key, value in correct_rollout_record["initial_hint"].items()
                    },
                    "initial_raw_hint_ball_pos": json_ready(correct_initial_hint_raw),
                    "initial_hint_metrics": json_ready(
                        compute_hint_metrics(
                            correct_hint_raw=correct_initial_hint_raw,
                            candidate_hint_raw=correct_initial_hint_raw,
                        )
                    ),
                    "initial_future_gate_debug": json_ready(correct_rollout_record["initial_future_gate_debug"]),
                    "initial_action_chunk": json_ready(correct_initial_chunk),
                    "initial_action_metrics": json_ready(correct_initial_action_metrics),
                    "rollout": json_ready(
                        {
                            key: value
                            for key, value in correct_rollout_record.items()
                            if key
                            not in {
                                "initial_hint",
                                "initial_raw_hint_ball_pos",
                                "initial_future_gate_debug",
                                "initial_action_chunk",
                            }
                        }
                    ),
                }
                logging.info(
                    "Full-episode %d/%d condition=%s finished | init_chunk_l2=%s reward=%s success=%s best_ball_to_bowl_xy_impr=%s elapsed=%.2fs",
                    episode_index + 1,
                    total_full_episodes,
                    correct_condition["name"],
                    None
                    if correct_initial_action_metrics is None
                    else correct_initial_action_metrics["chunk_l2_vs_correct"],
                    correct_rollout_record["sum_reward"],
                    correct_rollout_record["success"],
                    (correct_rollout_record.get("progress") or {}).get("best_ball_to_bowl_xy_dist_improvement"),
                    time.perf_counter() - correct_rollout_start_time,
                )

                for condition in condition_specs[1:]:
                    rollout_start_time = time.perf_counter()
                    rollout_record = full_episode_rollout(
                        ctx,
                        episode_trace=trace,
                        donor_trace=donor_trace,
                        condition=condition,
                        normalizer=normalizer,
                    )
                    initial_hint_metrics = compute_hint_metrics(
                        correct_hint_raw=correct_initial_hint_raw,
                        candidate_hint_raw=rollout_record["initial_raw_hint_ball_pos"],
                    )
                    initial_action_chunk = rollout_record.get("initial_action_chunk")
                    initial_action_metrics = None
                    if isinstance(initial_action_chunk, torch.Tensor) and isinstance(correct_initial_chunk, torch.Tensor):
                        initial_action_metrics = compute_action_diff_metrics(
                            correct_chunk=correct_initial_chunk,
                            candidate_chunk=initial_action_chunk,
                            compare_first_steps=int(args.compare_first_steps),
                        )

                    per_condition[condition["name"]] = {
                        "initial_hint": None
                        if rollout_record["initial_hint"] is None
                        else {
                            key: json_ready(value) for key, value in rollout_record["initial_hint"].items()
                        },
                        "initial_raw_hint_ball_pos": json_ready(rollout_record["initial_raw_hint_ball_pos"]),
                        "initial_hint_metrics": json_ready(initial_hint_metrics),
                        "initial_future_gate_debug": json_ready(rollout_record["initial_future_gate_debug"]),
                        "initial_action_chunk": json_ready(initial_action_chunk),
                        "initial_action_metrics": json_ready(initial_action_metrics),
                        "rollout": json_ready(
                            {
                                key: value
                                for key, value in rollout_record.items()
                                if key
                                not in {
                                    "initial_hint",
                                    "initial_raw_hint_ball_pos",
                                    "initial_future_gate_debug",
                                    "initial_action_chunk",
                                }
                            }
                        ),
                    }
                    logging.info(
                        "Full-episode %d/%d condition=%s finished | init_chunk_l2=%s reward=%s success=%s best_ball_to_bowl_xy_impr=%s elapsed=%.2fs",
                        episode_index + 1,
                        total_full_episodes,
                        condition["name"],
                        None if initial_action_metrics is None else initial_action_metrics["chunk_l2_vs_correct"],
                        rollout_record["sum_reward"],
                        rollout_record["success"],
                        (rollout_record.get("progress") or {}).get("best_ball_to_bowl_xy_dist_improvement"),
                        time.perf_counter() - rollout_start_time,
                    )

                full_episode_results.append(
                    {
                        "episode_index": trace.episode_index,
                        "donor_episode_index": donor_trace.episode_index,
                        "reference_trace_metrics": trace_metrics(trace),
                        "conditions": per_condition,
                    }
                )

        summary = summarize_results(probe_results, condition_names)
        full_episode_summary = (
            summarize_full_episode_results(full_episode_results, condition_names)
            if full_episode_results
            else None
        )
        result = {
            "config": {
                "args": vars(args),
                "meta": meta,
                "condition_names": condition_names,
                "alt_deltas": alt_deltas,
            },
            "reference_collection": {
                "requested_reference_episodes": int(args.reference_episodes),
                "reference_max_attempts": int(reference_max_attempts),
                "matched_reference_episodes": int(
                    sum(1 for attempt in reference_attempts if bool(attempt.get("accepted", False)))
                ),
                "fallback_used": bool(
                    has_reference_filters
                    and sum(1 for attempt in reference_attempts if bool(attempt.get("accepted", False)))
                    < int(args.reference_episodes)
                ),
                "filters": {
                    "require_success": bool(args.reference_require_success),
                    "min_reward_sum": args.reference_min_reward_sum,
                    "min_grasp_count": args.reference_min_grasp_count,
                },
                "selected_attempt_indices": [int(trace.episode_index) for trace in reference_traces],
                "attempts": reference_attempts,
            },
            "reference_episodes": [
                {
                    "episode_index": trace.episode_index,
                    "seed": trace.seed,
                    **trace_metrics(trace),
                    "selected_probe_steps": selected_probe_steps_by_episode.get(list_index, []),
                    "selected_probe_step_details": selected_probe_details_by_episode.get(list_index, []),
                }
                for list_index, trace in enumerate(reference_traces)
            ],
            "summary": summary,
            "full_episode_summary": full_episode_summary,
            "probes": probe_results,
            "full_episode_results": full_episode_results,
        }

        result_path = output_dir / "future_hint_probe_results.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(json_ready(result), f, ensure_ascii=False, indent=2)

        summary_path = output_dir / "future_hint_probe_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)

        full_episode_summary_path = None
        if full_episode_summary is not None:
            full_episode_summary_path = output_dir / "future_hint_probe_full_episode_summary.json"
            with full_episode_summary_path.open("w", encoding="utf-8") as f:
                json.dump(json_ready(full_episode_summary), f, ensure_ascii=False, indent=2)

        print("\nSummary")
        print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
        if full_episode_summary is not None:
            print("\nFull Episode Summary")
            print(json.dumps(json_ready(full_episode_summary), ensure_ascii=False, indent=2))
        print(f"\nSaved detailed results to {result_path}")
        if full_episode_summary_path is not None:
            print(f"Saved full-episode summary to {full_episode_summary_path}")
    finally:
        if ctx is not None:
            close_envs({"env": ctx.vec_env})


if __name__ == "__main__":
    main()

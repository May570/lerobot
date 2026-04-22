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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

"""
cd /home/admin123/桌面/wjl/lerobot

mkdir -p /tmp/libero_cfg_wjl /tmp/mpl
cat > /tmp/libero_cfg_wjl/config.yaml <<'YAML'
benchmark_root: /home/admin123/桌面/wjl/LIBERO/libero/libero
bddl_files: /home/admin123/桌面/wjl/LIBERO/libero/libero/bddl_files
init_states: /home/admin123/桌面/wjl/LIBERO/libero/libero/init_files
datasets: /home/admin123/桌面/wjl/LIBERO/libero/datasets
assets: /home/admin123/桌面/wjl/LIBERO/libero/libero/assets
YAML

LIBERO_CONFIG_PATH=/tmp/libero_cfg_wjl \
NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=/tmp/mpl \
MUJOCO_GL=glx \
DISPLAY=:0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n lerobot_py312 \
python "src/lerobot/scripts/lerobot_eval_realtime.py" \
  --policy.path=/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/lerobot_dp/original/checkpoints/030000/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=10 \
  --eval.batch_size=2 \
  --policy.use_amp=true \
  --policy.device=cuda \
  --libero_legacy_obs_compat=true \
  --policy.n_action_steps=15 \
  --policy.num_inference_steps=20 \
  --output_dir="outputs/eval3/original_ckpt030000_bs4_$(date +%Y%m%d_%H%M%S)"

"""

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, OBS_STR, REWARD
try:
    from lerobot.utils.constants import OBS_STATE_RAW
except ImportError:
    OBS_STATE_RAW = f"{OBS_STATE}_raw"
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.libero_compat import (
    apply_rename_map_to_preprocessor,
    resolve_libero_rename_map,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)


_DYN_MINI_PLAN_CANDIDATES = [
    "rolling_ball_to_bowl.collection_plan_balanced500_v1.jsonl",
    "rolling_ball_to_bowl.collection_plan_diverse_repro_v2.jsonl",
    "rolling_ball_to_bowl.collection_plan_repro_var_v2.jsonl",
    "rolling_ball_to_bowl.eval_plan_balanced_holdout_v1.jsonl",
]


def _resolve_dyn_mini_default_plan_path() -> Path | None:
    try:
        from libero.libero import get_libero_path
    except Exception:  # noqa: BLE001
        return None

    try:
        init_root = Path(get_libero_path("init_states")) / "libero_dyn_mini"
    except Exception:  # noqa: BLE001
        return None
    if not init_root.exists():
        return None

    for filename in _DYN_MINI_PLAN_CANDIDATES:
        candidate = init_root / filename
        if candidate.exists():
            return candidate
    return None


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    rollout_fps: float | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    policy_config = getattr(policy, "config", None)
    future_mode = str(getattr(policy_config, "model", "orig"))
    future_ball_pos_key = str(getattr(policy_config, "future_ball_pos_key", "observation.ball_pos"))
    input_features = getattr(policy_config, "input_features", None)
    need_scene_ball_pos = future_mode in {"scene_only", "robot_scene"}
    if not need_scene_ball_pos and isinstance(input_features, dict):
        need_scene_ball_pos = future_ball_pos_key in input_features
    need_env_state = bool(getattr(policy_config, "use_labels_environment_state", False))
    if not need_env_state and isinstance(input_features, dict):
        need_env_state = OBS_ENV_STATE in input_features
    ball_pos_dim = 3
    if isinstance(input_features, dict) and future_ball_pos_key in input_features:
        feature = input_features[future_ball_pos_key]
        shape = getattr(feature, "shape", None)
        if shape:
            ball_pos_dim = int(np.prod(shape))
    env_state_dim = 1
    if isinstance(input_features, dict) and OBS_ENV_STATE in input_features:
        feature = input_features[OBS_ENV_STATE]
        shape = getattr(feature, "shape", None)
        if shape:
            env_state_dim = int(np.prod(shape))
    force_zero_env_state = os.environ.get("LEROBOT_FORCE_ZERO_ENVIRONMENT_STATE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    logging.info(
        "[realtime3][obs-cond] mode=%s need_scene_ball_pos=%d future_ball_pos_key=%s "
        "ball_pos_dim=%d need_env_state=%d env_state_dim=%d force_zero_env_state=%d",
        future_mode,
        int(need_scene_ball_pos),
        future_ball_pos_key,
        ball_pos_dim,
        int(need_env_state),
        env_state_dim,
        int(force_zero_env_state),
    )

    def _maybe_inject_ball_pos(raw_observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not need_scene_ball_pos:
            return raw_observation

        if "ball_pos" in raw_observation or "observation.ball_pos" in raw_observation:
            return raw_observation

        if future_ball_pos_key in raw_observation:
            candidate = np.asarray(raw_observation[future_ball_pos_key], dtype=np.float32)
            if candidate.ndim == 1:
                candidate = candidate.reshape(1, -1)
            elif candidate.ndim >= 2:
                candidate = candidate.reshape(candidate.shape[0], -1)
            if candidate.shape[-1] < ball_pos_dim:
                raise RuntimeError(
                    "[realtime3][ball-pos] missing/invalid ball position: "
                    f"`{future_ball_pos_key}` last dim={candidate.shape[-1]} < expected {ball_pos_dim}."
                )
            candidate = candidate[:, :ball_pos_dim]
            if not np.isfinite(candidate).all():
                raise RuntimeError(
                    f"[realtime3][ball-pos] `{future_ball_pos_key}` contains NaN/Inf; aborting evaluation."
                )
            augmented_observation = dict(raw_observation)
            augmented_observation["ball_pos"] = candidate
            return augmented_observation

        try:
            raw_ball_pos = env.call("get_ball_pos")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "[realtime3][ball-pos] env.call('get_ball_pos') failed in scene mode; aborting evaluation."
            ) from exc

        if not isinstance(raw_ball_pos, (list, tuple)) or len(raw_ball_pos) == 0:
            raise RuntimeError(
                "[realtime3][ball-pos] no ball positions returned from env.call('get_ball_pos'); "
                "aborting evaluation."
            )

        ball_rows: list[np.ndarray] = []
        for env_idx, item in enumerate(raw_ball_pos):
            if item is None:
                raise RuntimeError(
                    f"[realtime3][ball-pos] missing ball position for env_idx={env_idx}; aborting evaluation."
                )
            row = np.asarray(item, dtype=np.float32).reshape(-1)
            if row.size < ball_pos_dim:
                raise RuntimeError(
                    "[realtime3][ball-pos] invalid ball position shape for "
                    f"env_idx={env_idx}: dim={row.size} < expected {ball_pos_dim}."
                )
            if row.size > ball_pos_dim:
                row = row[:ball_pos_dim]
            if not np.isfinite(row).all():
                raise RuntimeError(
                    f"[realtime3][ball-pos] ball position has NaN/Inf for env_idx={env_idx}; aborting evaluation."
                )
            ball_rows.append(row.astype(np.float32, copy=False))

        expected_num_envs = getattr(env, "num_envs", None)
        if expected_num_envs is not None and len(ball_rows) != int(expected_num_envs):
            raise RuntimeError(
                "[realtime3][ball-pos] env.call('get_ball_pos') size mismatch: "
                f"got {len(ball_rows)}, expected {int(expected_num_envs)}."
            )

        augmented_observation = dict(raw_observation)
        augmented_observation["ball_pos"] = np.stack(ball_rows, axis=0)
        return augmented_observation

    def _maybe_inject_environment_state(raw_observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not need_env_state:
            return raw_observation

        if force_zero_env_state:
            expected_num_envs = getattr(env, "num_envs", None)
            if expected_num_envs is None:
                sample_key = next(iter(raw_observation), None)
                if sample_key is None:
                    raise RuntimeError(
                        "[realtime3][env-state] cannot infer batch size while forcing zero environment_state."
                    )
                expected_num_envs = int(np.asarray(raw_observation[sample_key]).shape[0])
            augmented_observation = dict(raw_observation)
            augmented_observation["environment_state"] = np.zeros(
                (int(expected_num_envs), env_state_dim),
                dtype=np.float32,
            )
            return augmented_observation

        env_state_keys = ["environment_state", OBS_ENV_STATE]
        existing_key = next((key for key in env_state_keys if key in raw_observation), None)
        if existing_key is not None:
            candidate = np.asarray(raw_observation[existing_key], dtype=np.float32)
            if candidate.ndim == 1:
                candidate = candidate.reshape(-1, 1)
            elif candidate.ndim >= 2:
                candidate = candidate.reshape(candidate.shape[0], -1)
            if candidate.shape[-1] != env_state_dim:
                raise RuntimeError(
                    "[realtime3][env-state] invalid environment_state shape: "
                    f"last dim={candidate.shape[-1]} expected {env_state_dim}."
                )
            if not np.isfinite(candidate).all():
                raise RuntimeError("[realtime3][env-state] environment_state contains NaN/Inf; aborting evaluation.")
            if existing_key == "environment_state":
                return raw_observation
            augmented_observation = dict(raw_observation)
            augmented_observation["environment_state"] = candidate.astype(np.float32, copy=False)
            return augmented_observation

        if env_state_dim != 1:
            raise RuntimeError(
                "[realtime3][env-state] can only synthesize 1-D environment_state online from "
                f"`is_ball_grasped`, but policy expects dim={env_state_dim}."
            )

        try:
            raw_grasp = env.call("is_ball_grasped")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "[realtime3][env-state] env.call('is_ball_grasped') failed while injecting "
                "`environment_state`; aborting evaluation."
            ) from exc

        if not isinstance(raw_grasp, (list, tuple)) or len(raw_grasp) == 0:
            raise RuntimeError(
                "[realtime3][env-state] no grasp states returned from env.call('is_ball_grasped'); "
                "aborting evaluation."
            )

        env_state = np.asarray(raw_grasp, dtype=np.float32).reshape(-1, 1)
        if not np.isfinite(env_state).all():
            raise RuntimeError("[realtime3][env-state] synthesized environment_state has NaN/Inf; aborting evaluation.")

        expected_num_envs = getattr(env, "num_envs", None)
        if expected_num_envs is not None and env_state.shape[0] != int(expected_num_envs):
            raise RuntimeError(
                "[realtime3][env-state] env.call('is_ball_grasped') size mismatch: "
                f"got {env_state.shape[0]}, expected {int(expected_num_envs)}."
            )

        augmented_observation = dict(raw_observation)
        augmented_observation["environment_state"] = env_state
        return augmented_observation

    def _prepare_policy_observation(raw_observation: dict[str, np.ndarray], step_idx: int) -> tuple[dict, dict]:
        raw_observation = _maybe_inject_ball_pos(raw_observation)
        raw_observation = _maybe_inject_environment_state(raw_observation)
        # "base_obs" preserves the original eval payload behavior used for optional dataset/video bookkeeping.
        base_obs = preprocess_observation(raw_observation)
        if need_scene_ball_pos and "observation.ball_pos" not in base_obs:
            raise RuntimeError(
                "[realtime3][ball-pos] scene mode requires `observation.ball_pos` after preprocessing; "
                "aborting evaluation."
            )
        if need_env_state and OBS_ENV_STATE not in base_obs:
            raise RuntimeError(
                "[realtime3][env-state] policy requires `observation.environment_state` after preprocessing; "
                "aborting evaluation."
            )
        policy_obs = add_envs_task(env, deepcopy(base_obs))
        policy_obs = env_preprocessor(policy_obs)
        raw_state_for_kalman = None
        if OBS_STATE in policy_obs:
            # Preserve unnormalized state for Kalman-related rollout features.
            raw_state_for_kalman = policy_obs[OBS_STATE].detach().clone()
        policy_obs = preprocessor(policy_obs)
        if need_scene_ball_pos and future_ball_pos_key not in policy_obs:
            raise RuntimeError(
                "[realtime3][ball-pos] scene mode requires "
                f"`{future_ball_pos_key}` before policy inference; aborting evaluation."
            )
        if need_env_state and OBS_ENV_STATE not in policy_obs:
            raise RuntimeError(
                "[realtime3][env-state] policy requires "
                f"`{OBS_ENV_STATE}` before policy inference; aborting evaluation."
            )
        if raw_state_for_kalman is not None and OBS_STATE in policy_obs:
            raw_state_for_kalman = raw_state_for_kalman.to(
                device=policy_obs[OBS_STATE].device, dtype=policy_obs[OBS_STATE].dtype
            )
            policy_obs[OBS_STATE_RAW] = raw_state_for_kalman

        # Inject deterministic rollout timestamps based on env fps for stable dt.
        if rollout_fps is not None and rollout_fps > 0 and OBS_STATE in policy_obs:
            batch_size = int(policy_obs[OBS_STATE].shape[0])
            obs_state = policy_obs[OBS_STATE]
            ts = torch.full(
                (batch_size,),
                float(step_idx) / float(rollout_fps),
                device=obs_state.device,
                dtype=obs_state.dtype,
            )
            policy_obs["timestamp"] = ts
        return base_obs, policy_obs

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    all_ball_grasp_events = []
    gate_trace_per_env: list[list[dict[str, Any]]] = [[] for _ in range(env.num_envs)]

    max_steps = env.call("_max_episode_steps")[0]
    check_env_attributes_and_types(env)

    # Warm up simulation for a few no-op steps to stabilize state and populate observation history.
    # Cap to keep at least one control step in the rollout.
    warmup_steps = min(5, max(0, int(max_steps) - 1))
    step = 0
    base_observation, policy_observation = _prepare_policy_observation(observation, step)
    warmup_obs_history: deque[tuple[int, dict[str, Tensor]]] = deque(maxlen=2)
    warmup_obs_history.append((step, policy_observation))
    warmup_executed = 0
    if warmup_steps > 0:
        try:
            zero_action = np.zeros_like(env.action_space.sample())
        except Exception:  # noqa: BLE001
            action_shape = getattr(env.action_space, "shape", None)
            if action_shape is None:
                action_shape = (env.num_envs, 7)
            zero_action = np.zeros(action_shape, dtype=np.float32)

        for _ in range(warmup_steps):
            observation, _, _, _, _ = env.step(zero_action)
            warmup_executed += 1
            if render_callback is not None:
                render_callback(env)
            _, warmup_policy_observation = _prepare_policy_observation(observation, warmup_executed)
            warmup_obs_history.append((warmup_executed, warmup_policy_observation))

        step = warmup_executed
        base_observation, policy_observation = _prepare_policy_observation(observation, step)
        logging.info(
            "[realtime][warmup] requested=%d executed=%d",
            warmup_steps,
            warmup_executed,
        )

    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    if step > 0:
        progbar.update(step)

    # True pipelined control:
    # - Execution thread (main loop) always executes env steps.
    # - Inference worker thread continuously infers action chunks from the latest 2 observations.
    # - At inference end, drop actions that were consumed during inference and replace action_queue.
    supports_background_chunking = hasattr(policy, "predict_action_chunk") and hasattr(policy, "_queues")
    obs_queue: deque[tuple[int, dict[str, Tensor]]] = deque(maxlen=2)
    action_queue: deque[Tensor] = deque()
    infer_round = 0
    consumed_total = 0
    obs_version = 0
    stop_infer = False
    infer_error: Exception | None = None
    infer_thread: threading.Thread | None = None
    state_cv = threading.Condition()

    def _to_policy_input(batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_for_policy = {k: v for k, v in batch.items() if k != ACTION}
        image_features = getattr(policy.config, "image_features", None)
        if image_features:
            # Mirror DiffusionPolicy.select_action preprocessing so predict_action_chunk receives OBS_IMAGES.
            batch_for_policy = dict(batch_for_policy)
            batch_for_policy[OBS_IMAGES] = torch.stack([batch_for_policy[key] for key in image_features], dim=-4)
        return batch_for_policy

    def _read_future_gate_debug_from_policy() -> dict[str, list[float]] | None:
        diffusion = getattr(policy, "diffusion", None)
        if diffusion is None and hasattr(policy, "base_model"):
            diffusion = getattr(policy.base_model, "diffusion", None)
        if diffusion is None:
            return None
        getter = getattr(diffusion, "get_last_future_gate_debug", None)
        if not callable(getter):
            return None
        try:
            debug_info = getter(clear=True)
        except TypeError:
            debug_info = getter()
        if not isinstance(debug_info, dict):
            return None
        return debug_info

    def _infer_action_chunk_from_obs(
        old_obs: dict[str, Tensor], new_obs: dict[str, Tensor]
    ) -> tuple[list[Tensor], float, dict[str, list[float]] | None]:
        device_str = str(getattr(policy.config, "device", "cpu"))
        device_type = "cuda" if device_str.startswith("cuda") else "cpu"
        use_amp = bool(getattr(policy.config, "use_amp", False))

        old_policy_input = _to_policy_input(old_obs)
        new_policy_input = _to_policy_input(new_obs)
        # Maintain strict temporal order in policy queues: older frame first, newer frame second.
        infer_t0 = time.perf_counter()
        policy._queues = populate_queues(policy._queues, old_policy_input)
        policy._queues = populate_queues(policy._queues, new_policy_input)
        with torch.inference_mode(), (
            torch.autocast(device_type=device_type) if use_amp else nullcontext()
        ):
            action_chunk = policy.predict_action_chunk(new_policy_input)
        future_gate_debug = _read_future_gate_debug_from_policy()
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(1)
        assert action_chunk.ndim == 3, "Action chunk dimensions should be (batch, chunk, action_dim)"

        chunk_actions_cpu: list[Tensor] = []
        for chunk_step in range(action_chunk.shape[1]):
            action_t = action_chunk[:, chunk_step]
            action_t = postprocessor(action_t)
            action_transition = {ACTION: action_t}
            action_transition = env_postprocessor(action_transition)
            chunk_actions_cpu.append(action_transition[ACTION].detach().to("cpu"))

        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        return chunk_actions_cpu, infer_ms, future_gate_debug

    def _inference_worker() -> None:
        nonlocal infer_round, stop_infer, infer_error
        last_inferred_obs_version = -1
        try:
            while True:
                with state_cv:
                    while True:
                        if stop_infer:
                            return
                        have_two_obs = len(obs_queue) >= 2
                        action_queue_empty = len(action_queue) == 0
                        obs_updated = obs_version != last_inferred_obs_version
                        if have_two_obs and (obs_updated or action_queue_empty):
                            old_obs_step, old_obs = obs_queue[0]
                            new_obs_step, new_obs = obs_queue[-1]
                            infer_round += 1
                            round_id = infer_round
                            obs_version_snapshot = obs_version
                            action_queue_before = len(action_queue)
                            consumed_start = consumed_total
                            break
                        state_cv.wait(timeout=0.05)

                logging.info(
                    "[realtime][infer-start] round=%d obs_steps=(%d,%d) action_queue_before=%d consumed_total=%d",
                    round_id,
                    old_obs_step,
                    new_obs_step,
                    action_queue_before,
                    consumed_start,
                )
                chunk_actions_cpu, infer_ms, future_gate_debug = _infer_action_chunk_from_obs(old_obs, new_obs)

                with state_cv:
                    consumed_during = max(0, consumed_total - consumed_start)
                    drop_n = min(consumed_during, len(chunk_actions_cpu))
                    kept_actions = chunk_actions_cpu[drop_n:]
                    action_queue.clear()
                    action_queue.extend(kept_actions)
                    action_queue_after = len(action_queue)
                    last_inferred_obs_version = obs_version_snapshot
                    state_cv.notify_all()

                if isinstance(future_gate_debug, dict) and len(future_gate_debug) > 0:
                    for env_idx in range(env.num_envs):
                        gate_record: dict[str, Any] = {
                            "round": int(round_id),
                            "obs_steps": [int(old_obs_step), int(new_obs_step)],
                        }
                        for key, values in future_gate_debug.items():
                            value: float | None = None
                            if isinstance(values, (list, tuple)) and env_idx < len(values):
                                raw = values[env_idx]
                                try:
                                    cast = float(raw)
                                    value = cast if np.isfinite(cast) else None
                                except Exception:  # noqa: BLE001
                                    value = None
                            gate_record[key] = value
                        gate_trace_per_env[env_idx].append(gate_record)

                logging.info(
                    "[realtime][infer-end] round=%d infer_ms=%.1f produced=%d consumed_during=%d dropped=%d kept=%d action_queue_after=%d",
                    round_id,
                    infer_ms,
                    len(chunk_actions_cpu),
                    consumed_during,
                    drop_n,
                    len(kept_actions),
                    action_queue_after,
                )
        except Exception as exc:  # noqa: BLE001
            with state_cv:
                infer_error = exc
                stop_infer = True
                state_cv.notify_all()

    if supports_background_chunking:
        # Bootstrap observation queue with warmup history (old -> new).
        if len(warmup_obs_history) == 0:
            warmup_obs_history.append((step, policy_observation))
        if len(warmup_obs_history) == 1:
            warmup_obs_history.append(warmup_obs_history[0])
        with state_cv:
            for frame in list(warmup_obs_history)[-2:]:
                obs_queue.append(frame)
                obs_version += 1
            state_cv.notify_all()

        infer_thread = threading.Thread(target=_inference_worker, name="realtime-infer", daemon=True)
        infer_thread.start()

    track_ball_grasp = False
    grasp_prev = np.zeros(env.num_envs, dtype=bool)
    if hasattr(env, "call"):
        try:
            raw_grasp = env.call("is_ball_grasped")
            if isinstance(raw_grasp, (list, tuple)) and len(raw_grasp) == env.num_envs:
                grasp_prev = np.asarray(raw_grasp, dtype=bool)
                track_ball_grasp = True
                logging.info("[realtime][grasp] enabled grasp event tracking.")
        except Exception:  # noqa: BLE001
            track_ball_grasp = False

    execution_error: Exception | None = None
    try:
        while not np.all(done) and step < max_steps:
            if return_observations:
                all_observations.append(deepcopy(base_observation))

            if supports_background_chunking:
                waited_for_actions = False
                with state_cv:
                    while len(action_queue) == 0 and not stop_infer:
                        if not waited_for_actions:
                            logging.info(
                                "[realtime][exec-wait] step=%d action_queue_empty=1 waiting_for_inference=1",
                                step,
                            )
                            waited_for_actions = True
                        state_cv.wait(timeout=0.05)

                    if infer_error is not None:
                        raise RuntimeError("Inference worker failed.") from infer_error
                    if len(action_queue) == 0:
                        raise RuntimeError("Inference worker stopped with an empty action queue.")

                    action = action_queue.popleft()
                    consumed_total += 1
                    action_queue_len_after_pop = len(action_queue)

                if waited_for_actions:
                    logging.info(
                        "[realtime][exec-wait-end] step=%d action_queue_len=%d resume_execution=1",
                        step,
                        action_queue_len_after_pop,
                    )
            else:
                with torch.inference_mode():
                    action = policy.select_action(policy_observation)
                action = postprocessor(action)
                action_transition = {ACTION: action}
                action_transition = env_postprocessor(action_transition)
                action = action_transition[ACTION]
                action = action.detach().to("cpu")
                action_queue_len_after_pop = -1

            # Convert to CPU / numpy.
            action_numpy: np.ndarray = action.numpy()
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            if render_callback is not None:
                render_callback(env)

            # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
            # available if none of the envs finished.
            if "final_info" in info:
                final_info = info["final_info"]
                if not isinstance(final_info, dict):
                    raise RuntimeError(
                        "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                        "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                    )
                successes = final_info["is_success"].tolist()
            else:
                successes = [False] * env.num_envs

            grasp_event_step = np.zeros(env.num_envs, dtype=np.int32)
            if track_ball_grasp:
                raw_grasp = env.call("is_ball_grasped")
                if not isinstance(raw_grasp, (list, tuple)) or len(raw_grasp) != env.num_envs:
                    raise RuntimeError(
                        "[realtime][grasp] env.call('is_ball_grasped') returned invalid shape "
                        f"{type(raw_grasp)} len={len(raw_grasp) if isinstance(raw_grasp, (list, tuple)) else 'NA'}."
                    )
                grasp_now = np.asarray(raw_grasp, dtype=bool)
                grasp_event_step = np.logical_and(np.logical_not(grasp_prev), grasp_now).astype(np.int32)
                grasp_prev = grasp_now
                if int(np.sum(grasp_event_step)) > 0:
                    logging.info(
                        "[realtime][grasp] step=%d new_grasps=%d",
                        step,
                        int(np.sum(grasp_event_step)),
                    )

            # Keep track of which environments are done so far.
            # Mark the episode as done if we reach the maximum step limit.
            # This ensures that the rollout always terminates cleanly at `max_steps`,
            # and allows logging/saving (e.g., videos) to be triggered consistently.
            done = terminated | truncated | done
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)

            all_actions.append(torch.from_numpy(action_numpy))
            all_rewards.append(torch.from_numpy(reward))
            all_dones.append(torch.from_numpy(done))
            all_successes.append(torch.tensor(successes))
            all_ball_grasp_events.append(torch.from_numpy(grasp_event_step))

            logging.info(
                "[realtime][exec] step=%d action_queue_after_pop=%d reward_mean=%.4f done_envs=%d/%d success_envs=%d/%d",
                step,
                action_queue_len_after_pop,
                float(np.mean(reward)),
                int(done.sum()),
                int(env.num_envs),
                int(np.sum(successes)),
                int(env.num_envs),
            )

            step += 1
            running_success_rate = (
                einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
            )
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            progbar.update()

            if step < max_steps and not np.all(done):
                base_observation, policy_observation = _prepare_policy_observation(observation, step)
                if supports_background_chunking:
                    with state_cv:
                        obs_queue.append((step, policy_observation))
                        obs_version += 1
                        state_cv.notify_all()
    except Exception as exc:  # noqa: BLE001
        execution_error = exc
    finally:
        if supports_background_chunking:
            with state_cv:
                stop_infer = True
                state_cv.notify_all()
            if infer_thread is not None:
                infer_thread.join(timeout=10.0)
                if infer_thread.is_alive():
                    logging.warning("[realtime] inference worker did not exit cleanly within timeout.")

    if execution_error is not None:
        raise execution_error
    if infer_error is not None:
        raise RuntimeError("Inference worker failed during rollout.") from infer_error

    # Track the final observation.
    if return_observations:
        final_observation = preprocess_observation(observation)
        all_observations.append(deepcopy(final_observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        "ball_grasp_event": torch.stack(all_ball_grasp_events, dim=1),
        "future_gate_trace": gate_trace_per_env,
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    rollout_fps: float | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    ball_grasp_counts = []
    ball_grasp_successes = []
    all_gate_traces: list[list[dict[str, Any]]] = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            rollout_fps=rollout_fps,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if "ball_grasp_event" in rollout_data:
            batch_grasp_counts = einops.reduce((rollout_data["ball_grasp_event"] * mask), "b n -> b", "sum")
            batch_grasp_counts = batch_grasp_counts.to(torch.int64)
            batch_grasp_success = batch_grasp_counts > 0
            ball_grasp_counts.extend(batch_grasp_counts.tolist())
            ball_grasp_successes.extend(batch_grasp_success.tolist())
        else:
            ball_grasp_counts.extend([0] * env.num_envs)
            ball_grasp_successes.extend([False] * env.num_envs)
        batch_gate_traces = rollout_data.get("future_gate_trace", [])
        if isinstance(batch_gate_traces, list) and len(batch_gate_traces) == env.num_envs:
            all_gate_traces.extend(batch_gate_traces)
        else:
            all_gate_traces.extend([[] for _ in range(env.num_envs)])
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    per_episode_records = []
    for i in range(n_episodes):
        per_episode_records.append(
            {
                "episode_ix": i,
                "sum_reward": sum_rewards[i],
                "max_reward": max_rewards[i],
                "success": all_successes[i],
                "seed": all_seeds[i] if i < len(all_seeds) else None,
                "ball_grasp_count": int(ball_grasp_counts[i]) if i < len(ball_grasp_counts) else 0,
                "ball_grasp_success": bool(ball_grasp_successes[i]) if i < len(ball_grasp_successes) else False,
                "future_gate_trace": all_gate_traces[i] if i < len(all_gate_traces) else [],
            }
        )

    info = {
        "per_episode": per_episode_records,
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "avg_ball_grasp_count": float(np.nanmean(ball_grasp_counts[:n_episodes])),
            "pc_ball_grasp_success": float(np.nanmean(ball_grasp_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    if getattr(cfg.env, "type", None) == "libero" and getattr(cfg.env, "task", None) == "libero_dyn_mini":
        min_dyn_mini_episodes_raw = os.getenv("LEROBOT_DYN_MINI_MIN_EPISODES", "100")
        try:
            min_dyn_mini_episodes = int(min_dyn_mini_episodes_raw)
        except ValueError:
            logging.warning(
                "[realtime3][dyn-mini] invalid LEROBOT_DYN_MINI_MIN_EPISODES=%s, falling back to 100.",
                min_dyn_mini_episodes_raw,
            )
            min_dyn_mini_episodes = 100

        if min_dyn_mini_episodes > 0 and getattr(cfg.eval, "n_episodes", 0) < min_dyn_mini_episodes:
            logging.info(
                "[realtime3][dyn-mini] overriding eval.n_episodes from %d to %d for richer coverage.",
                int(cfg.eval.n_episodes),
                int(min_dyn_mini_episodes),
            )
            cfg.eval.n_episodes = min_dyn_mini_episodes
        elif min_dyn_mini_episodes <= 0:
            logging.info(
                "[realtime3][dyn-mini] minimum episode override disabled via LEROBOT_DYN_MINI_MIN_EPISODES=%d.",
                min_dyn_mini_episodes,
            )
        if hasattr(cfg.env, "task_ids") and cfg.env.task_ids is None:
            cfg.env.task_ids = [0]
        if hasattr(cfg.env, "init_states"):
            cfg.env.init_states = True
        if hasattr(cfg.env, "init_plan_path") and not getattr(cfg.env, "init_plan_path", None):
            default_plan = _resolve_dyn_mini_default_plan_path()
            if default_plan is not None:
                cfg.env.init_plan_path = str(default_plan)
                logging.info("[realtime3][dyn-mini] using init plan: %s", default_plan)
            else:
                logging.warning(
                    "[realtime3][dyn-mini] no default init plan found. Falling back to init_states cycling."
                )

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy_feature_keys = []
    if cfg.policy is not None and cfg.policy.input_features:
        policy_feature_keys = list(cfg.policy.input_features.keys())
    effective_rename_map = resolve_libero_rename_map(
        enable_legacy_compat=cfg.libero_legacy_obs_compat,
        env_cfg=cfg.env,
        feature_keys=policy_feature_keys,
        user_rename_map=cfg.rename_map,
    )
    if effective_rename_map != cfg.rename_map:
        logging.warning(
            "Enabled LIBERO legacy observation compatibility mapping: %s",
            effective_rename_map,
        )

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=effective_rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": effective_rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    apply_rename_map_to_preprocessor(preprocessor, effective_rename_map)

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.n_episodes,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            rollout_fps=getattr(cfg.env, "fps", None),
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.get("per_group", {}).items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)
    gate_trace_path = Path(cfg.output_dir) / "gate_trace_per_episode.jsonl"
    with open(gate_trace_path, "w", encoding="utf-8") as f:
        for record in info.get("gate_trace_per_episode", []):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info("Saved gate trace records to %s", gate_trace_path)

    logging.info("End of eval")


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    rollout_fps: float | None = None,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}} sequentially.

    NOTE:
    `max_parallel_tasks` is accepted for CLI compatibility but is intentionally ignored
    here to keep realtime evaluation behavior deterministic and easier to debug.
    """
    start_t = time.time()
    per_group_raw: dict[str, dict[str, list]] = {}
    overall_sum_rewards: list[float] = []
    overall_max_rewards: list[float] = []
    overall_successes: list[bool] = []
    overall_ball_grasp_counts: list[int] = []
    overall_ball_grasp_successes: list[bool] = []
    overall_video_paths: list[str] = []
    gate_trace_per_episode: list[dict[str, Any]] = []
    per_task_infos: list[dict] = []
    global_episode_ix = 0

    for task_group, group_envs in envs.items():
        group_sum_rewards: list[float] = []
        group_max_rewards: list[float] = []
        group_successes: list[bool] = []
        group_ball_grasp_counts: list[int] = []
        group_ball_grasp_successes: list[bool] = []
        group_video_paths: list[str] = []

        for task_id, env in group_envs.items():
            task_videos_dir = None
            if videos_dir is not None:
                task_videos_dir = videos_dir / f"{task_group}_{task_id}"
                task_videos_dir.mkdir(parents=True, exist_ok=True)

            task_result = eval_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=n_episodes,
                max_episodes_rendered=max_episodes_rendered,
                videos_dir=task_videos_dir,
                return_episode_data=return_episode_data,
                start_seed=start_seed,
                rollout_fps=rollout_fps,
            )

            per_episode = task_result["per_episode"]
            task_sum_rewards = [ep["sum_reward"] for ep in per_episode]
            task_max_rewards = [ep["max_reward"] for ep in per_episode]
            task_successes = [ep["success"] for ep in per_episode]
            task_ball_grasp_counts = [int(ep.get("ball_grasp_count", 0)) for ep in per_episode]
            task_ball_grasp_successes = [bool(ep.get("ball_grasp_success", False)) for ep in per_episode]
            task_video_paths = task_result.get("video_paths", [])

            group_sum_rewards.extend(task_sum_rewards)
            group_max_rewards.extend(task_max_rewards)
            group_successes.extend(task_successes)
            group_ball_grasp_counts.extend(task_ball_grasp_counts)
            group_ball_grasp_successes.extend(task_ball_grasp_successes)
            group_video_paths.extend(task_video_paths)

            overall_sum_rewards.extend(task_sum_rewards)
            overall_max_rewards.extend(task_max_rewards)
            overall_successes.extend(task_successes)
            overall_ball_grasp_counts.extend(task_ball_grasp_counts)
            overall_ball_grasp_successes.extend(task_ball_grasp_successes)
            overall_video_paths.extend(task_video_paths)

            for ep in per_episode:
                gate_trace_per_episode.append(
                    {
                        "global_episode_ix": global_episode_ix,
                        "task_group": task_group,
                        "task_id": int(task_id),
                        "episode_ix_in_task": int(ep.get("episode_ix", global_episode_ix)),
                        "success": bool(ep.get("success", False)),
                        "ball_grasp_count": int(ep.get("ball_grasp_count", 0)),
                        "future_gate_trace": ep.get("future_gate_trace", []),
                    }
                )
                global_episode_ix += 1

            per_task_infos.append(
                {
                    "task_group": task_group,
                    "task_id": task_id,
                    "metrics": {
                        "sum_rewards": task_sum_rewards,
                        "max_rewards": task_max_rewards,
                        "successes": task_successes,
                        "ball_grasp_counts": task_ball_grasp_counts,
                        "ball_grasp_successes": task_ball_grasp_successes,
                        "video_paths": task_video_paths,
                    },
                }
            )

        per_group_raw[task_group] = {
            "sum_rewards": group_sum_rewards,
            "max_rewards": group_max_rewards,
            "successes": group_successes,
            "ball_grasp_counts": group_ball_grasp_counts,
            "ball_grasp_successes": group_ball_grasp_successes,
            "video_paths": group_video_paths,
        }

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in per_group_raw.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "avg_ball_grasp_count": _agg_from_list(acc["ball_grasp_counts"]),
            "pc_ball_grasp_success": (
                _agg_from_list(acc["ball_grasp_successes"]) * 100 if acc["ball_grasp_successes"] else float("nan")
            ),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall_sum_rewards),
        "avg_max_reward": _agg_from_list(overall_max_rewards),
        "pc_success": _agg_from_list(overall_successes) * 100 if overall_successes else float("nan"),
        "avg_ball_grasp_count": _agg_from_list(overall_ball_grasp_counts),
        "pc_ball_grasp_success": (
            _agg_from_list(overall_ball_grasp_successes) * 100 if overall_ball_grasp_successes else float("nan")
        ),
        "n_episodes": len(overall_sum_rewards),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall_sum_rewards)),
        "video_paths": list(overall_video_paths),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
        "gate_trace_per_episode": gate_trace_per_episode,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import os
import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.processor import RobotObservation


logger = logging.getLogger(__name__)
_EPISODE_START_CACHE_VERSION = 1


def load_episode_start_states(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).expanduser()
    with np.load(cache_path, allow_pickle=False) as data:
        if "sim_states" not in data:
            raise ValueError(f"Episode-start cache at {cache_path} is missing `sim_states`.")
        sim_states = np.asarray(data["sim_states"], dtype=np.float64)
        table_body_quats = None
        if "table_body_quats" in data:
            table_body_quats = np.asarray(data["table_body_quats"], dtype=np.float64)
        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"].item()))
    if sim_states.ndim != 2:
        raise ValueError(f"`sim_states` in {cache_path} must be rank-2, got shape {sim_states.shape}.")
    if table_body_quats is not None and table_body_quats.shape != (sim_states.shape[0], 4):
        raise ValueError(
            f"`table_body_quats` in {cache_path} must have shape ({sim_states.shape[0]}, 4), "
            f"got {table_body_quats.shape}."
        )
    metadata.setdefault("version", _EPISODE_START_CACHE_VERSION)
    return {
        "path": cache_path,
        "sim_states": sim_states,
        "table_body_quats": table_body_quats,
        "metadata": metadata,
    }


def save_episode_start_states(
    path: str | Path,
    *,
    sim_states: np.ndarray,
    table_body_quats: np.ndarray | None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "sim_states": np.asarray(sim_states, dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(dict(metadata or {}), ensure_ascii=True), dtype=np.str_),
    }
    if table_body_quats is not None:
        payload["table_body_quats"] = np.asarray(table_body_quats, dtype=np.float64)
    np.savez_compressed(out_path, **payload)
    return out_path


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    return init_states


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
        init_plan_path: str | None = None,
        episode_start_states_path: str | None = None,
        init_plan_loop: bool = True,
        init_plan_default_direction_deg: float = 270.0,
        init_plan_default_speed: float = 0.30,
        init_plan_default_ball_start_x: float = 0.04,
        init_plan_default_ball_start_y: float = 0.26,
        init_plan_ball_start_xy_safety_scale: float = 0.92,
        init_plan_ball_start_z_clearance: float = 0.0015,
        init_plan_launch_settle_steps: int = 6,
        init_plan_launch_ramp_steps: int = 8,
        init_plan_warmup_steps: int = 0,
        ball_grasp_eval_mode: str = "legacy",
        ball_grasp_strict_lift_multiplier: float = 1.0,
        ball_grasp_strict_grip_center_max_dist: float = 0.055,
        ball_grasp_strict_require_pad_contact: bool = True,
    ):
        super().__init__()
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = _parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self.episode_length = episode_length
        # Load once and keep
        self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._reset_stride = n_envs  # when performing a reset, append `_reset_stride` to `init_state_id`.

        self.init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        self._env = self._make_envs_task(task_suite, self.task_id)
        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        self._init_plan_path = Path(init_plan_path).expanduser() if init_plan_path else None
        self._episode_start_states_path = (
            Path(episode_start_states_path).expanduser() if episode_start_states_path else None
        )
        self._init_plan_loop = bool(init_plan_loop)
        self._init_plan_default_direction_deg = float(init_plan_default_direction_deg)
        self._init_plan_default_speed = float(init_plan_default_speed)
        self._init_plan_default_ball_start_x = float(init_plan_default_ball_start_x)
        self._init_plan_default_ball_start_y = float(init_plan_default_ball_start_y)
        self._init_plan_ball_start_xy_safety_scale = float(init_plan_ball_start_xy_safety_scale)
        self._init_plan_ball_start_z_clearance = float(init_plan_ball_start_z_clearance)
        self._init_plan_launch_settle_steps = int(init_plan_launch_settle_steps)
        self._init_plan_launch_ramp_steps = int(init_plan_launch_ramp_steps)
        self._init_plan_warmup_steps = int(init_plan_warmup_steps)
        self._ball_grasp_eval_mode = str(ball_grasp_eval_mode)
        if self._ball_grasp_eval_mode not in {"legacy", "strict"}:
            raise ValueError(
                f"`ball_grasp_eval_mode` must be one of {{'legacy', 'strict'}}. Got {self._ball_grasp_eval_mode!r}."
            )
        self._ball_grasp_strict_lift_multiplier = float(ball_grasp_strict_lift_multiplier)
        self._ball_grasp_strict_grip_center_max_dist = float(ball_grasp_strict_grip_center_max_dist)
        self._ball_grasp_strict_require_pad_contact = bool(ball_grasp_strict_require_pad_contact)
        self._init_plan_rows = self._load_init_plan_rows(self._init_plan_path)
        self._init_plan_index = self.episode_index
        self._init_plan_stride = n_envs
        self._episode_start_state_index = self.episode_index
        self._episode_start_state_stride = n_envs
        self._episode_start_states: np.ndarray | None = None
        self._episode_start_table_body_quats: np.ndarray | None = None
        self._episode_start_metadata: dict[str, Any] = {}
        self._active_episode_start_cache_idx: int | None = None
        self._dyn_ball_body_id: int | None = None
        self._dyn_ball_joint_name: str | None = None
        self._dyn_table_collision_geom_id: int | None = None
        self._dyn_table_body_id: int | None = None
        self._dyn_base_table_quat: np.ndarray | None = None
        self._dyn_ball_geom_ids: tuple[int, ...] = tuple()
        self._dyn_left_finger_geom_ids: tuple[int, ...] = tuple()
        self._dyn_right_finger_geom_ids: tuple[int, ...] = tuple()
        self._dyn_left_fingerpad_geom_ids: tuple[int, ...] = tuple()
        self._dyn_right_fingerpad_geom_ids: tuple[int, ...] = tuple()
        self._dyn_grip_site_id: int | None = None
        self._dyn_ball_radius: float | None = None
        self._dyn_handles_ready = False
        if self._episode_start_states_path is not None:
            loaded_cache = load_episode_start_states(self._episode_start_states_path)
            self._episode_start_states = loaded_cache["sim_states"]
            self._episode_start_table_body_quats = loaded_cache["table_body_quats"]
            self._episode_start_metadata = loaded_cache["metadata"]
            logger.info(
                "Loaded %d fixed episode-start states from %s",
                int(self._episode_start_states.shape[0]),
                self._episode_start_states_path,
            )
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self):
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        image = image[::-1, ::-1]  # flip both H and W for visualization
        return image

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    @staticmethod
    def _load_init_plan_rows(plan_path: Path | None) -> list[dict[str, Any]]:
        if plan_path is None:
            return []
        if not plan_path.exists():
            raise FileNotFoundError(f"LIBERO init plan file does not exist: {plan_path}")
        rows: list[dict[str, Any]] = []
        with plan_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        if len(rows) == 0:
            raise ValueError(f"LIBERO init plan file is empty: {plan_path}")
        logger.info("Loaded %d init-plan rows from %s", len(rows), plan_path)
        return rows

    @staticmethod
    def _safe_float(raw: Any, default: float) -> float:
        try:
            return float(raw)
        except Exception:  # noqa: BLE001
            return float(default)

    @staticmethod
    def _safe_int(raw: Any, default: int) -> int:
        try:
            return int(raw)
        except Exception:  # noqa: BLE001
            return int(default)

    @staticmethod
    def _find_body_id_contains(model: Any, name_fragment: str) -> int:
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name and name_fragment in name:
                return int(i)
        raise KeyError(f"No body name containing '{name_fragment}'")

    @staticmethod
    def _find_table_collision_geom_id(model: Any) -> int:
        try:
            return int(model.geom_name2id("table_collision"))
        except Exception:  # noqa: BLE001
            for i in range(model.ngeom):
                name = model.geom_id2name(i)
                if name and "table_collision" in name:
                    return int(i)
        raise KeyError("Could not find table collision geom.")

    @staticmethod
    def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = axis / norm
        half = 0.5 * float(angle_rad)
        s = math.sin(half)
        return np.asarray([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)

    @classmethod
    def _tilt_quat(cls, direction_deg: float, tilt_deg: float) -> np.ndarray:
        theta = math.radians(float(direction_deg))
        axis = np.asarray([-math.sin(theta), math.cos(theta), 0.0], dtype=np.float64)
        return cls._axis_angle_to_quat(axis=axis, angle_rad=math.radians(float(tilt_deg)))

    def _resolve_dyn_handles(self) -> None:
        if self._dyn_handles_ready:
            return

        model = self._env.sim.model
        try:
            self._dyn_ball_joint_name = "ball_1_joint0"
            model.joint_name2id(self._dyn_ball_joint_name)
        except Exception:  # noqa: BLE001
            self._dyn_ball_joint_name = None

        try:
            self._dyn_ball_body_id = self._find_body_id_contains(model, "ball_1")
        except Exception:  # noqa: BLE001
            self._dyn_ball_body_id = None

        try:
            self._dyn_table_collision_geom_id = self._find_table_collision_geom_id(model)
        except Exception:  # noqa: BLE001
            self._dyn_table_collision_geom_id = None

        if self._dyn_table_collision_geom_id is not None:
            try:
                self._dyn_table_body_id = int(model.geom_bodyid[self._dyn_table_collision_geom_id])
                self._dyn_base_table_quat = np.asarray(
                    model.body_quat[self._dyn_table_body_id], dtype=np.float64
                ).copy()
            except Exception:  # noqa: BLE001
                self._dyn_table_body_id = None
                self._dyn_base_table_quat = None

        if self._dyn_ball_body_id is not None:
            ball_geom_ids: list[int] = []
            ball_radius_candidates: list[float] = []
            for gid in range(model.ngeom):
                if int(model.geom_bodyid[gid]) != int(self._dyn_ball_body_id):
                    continue
                name = model.geom_id2name(gid)
                if name and "vis" in name.lower():
                    continue
                ball_geom_ids.append(int(gid))
                size = np.asarray(model.geom_size[gid], dtype=np.float64).reshape(-1)
                if size.size > 0:
                    ball_radius_candidates.append(float(size[0]))
            self._dyn_ball_geom_ids = tuple(ball_geom_ids)
            self._dyn_ball_radius = max(ball_radius_candidates) if ball_radius_candidates else 0.02

        left_names: set[str] = set()
        right_names: set[str] = set()
        left_pad_names: set[str] = set()
        right_pad_names: set[str] = set()
        try:
            gripper = self._env.robots[0].gripper
            important_geoms = getattr(gripper, "important_geoms", {}) or {}
            for key, values in important_geoms.items():
                key_l = str(key).lower()
                geom_names = values if isinstance(values, (list, tuple)) else [values]
                if "left" in key_l and "finger" in key_l:
                    left_names.update(str(n) for n in geom_names if n)
                if "right" in key_l and "finger" in key_l:
                    right_names.update(str(n) for n in geom_names if n)
                if "left" in key_l and "fingerpad" in key_l:
                    left_pad_names.update(str(n) for n in geom_names if n)
                if "right" in key_l and "fingerpad" in key_l:
                    right_pad_names.update(str(n) for n in geom_names if n)
            important_sites = getattr(gripper, "important_sites", {}) or {}
            grip_site_name = important_sites.get("grip_site")
            if isinstance(grip_site_name, str) and grip_site_name:
                try:
                    self._dyn_grip_site_id = int(model.site_name2id(grip_site_name))
                except Exception:  # noqa: BLE001
                    self._dyn_grip_site_id = None
        except Exception:  # noqa: BLE001
            pass

        if not left_names or not right_names:
            for gid in range(model.ngeom):
                name = model.geom_id2name(gid)
                if not name:
                    continue
                lowered = name.lower()
                if "gripper0_finger1" in lowered:
                    left_names.add(name)
                elif "gripper0_finger2" in lowered:
                    right_names.add(name)
                if "gripper0_finger1_pad" in lowered:
                    left_pad_names.add(name)
                elif "gripper0_finger2_pad" in lowered:
                    right_pad_names.add(name)

        left_ids: list[int] = []
        for name in sorted(left_names):
            try:
                left_ids.append(int(model.geom_name2id(name)))
            except Exception:  # noqa: BLE001
                continue
        right_ids: list[int] = []
        for name in sorted(right_names):
            try:
                right_ids.append(int(model.geom_name2id(name)))
            except Exception:  # noqa: BLE001
                continue
        self._dyn_left_finger_geom_ids = tuple(left_ids)
        self._dyn_right_finger_geom_ids = tuple(right_ids)
        left_pad_ids: list[int] = []
        for name in sorted(left_pad_names):
            try:
                left_pad_ids.append(int(model.geom_name2id(name)))
            except Exception:  # noqa: BLE001
                continue
        right_pad_ids: list[int] = []
        for name in sorted(right_pad_names):
            try:
                right_pad_ids.append(int(model.geom_name2id(name)))
            except Exception:  # noqa: BLE001
                continue
        self._dyn_left_fingerpad_geom_ids = tuple(left_pad_ids)
        self._dyn_right_fingerpad_geom_ids = tuple(right_pad_ids)

        self._dyn_handles_ready = True

    def get_ball_pos(self) -> np.ndarray | None:
        self._resolve_dyn_handles()
        if self._dyn_ball_body_id is None:
            return None
        try:
            pos = np.asarray(self._env.sim.data.body_xpos[self._dyn_ball_body_id], dtype=np.float32).copy()
        except Exception:  # noqa: BLE001
            return None
        return pos

    def is_ball_grasped(self) -> bool:
        """Best-effort grasp detector for dyn-mini ball with selectable strictness."""
        self._resolve_dyn_handles()
        if self._dyn_ball_body_id is None or len(self._dyn_ball_geom_ids) == 0:
            return False

        if self._ball_grasp_eval_mode == "strict":
            return self._is_ball_grasped_strict()
        return self._is_ball_grasped_legacy()

    def _has_dual_finger_ball_contact(self, left_geom_ids: set[int], right_geom_ids: set[int]) -> bool:
        if not left_geom_ids or not right_geom_ids:
            return False
        data = self._env.sim.data
        ball_geom_ids = set(self._dyn_ball_geom_ids)
        left_contact = False
        right_contact = False
        for cid in range(int(data.ncon)):
            contact = data.contact[cid]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if (g1 in ball_geom_ids and g2 in left_geom_ids) or (g2 in ball_geom_ids and g1 in left_geom_ids):
                left_contact = True
            if (g1 in ball_geom_ids and g2 in right_geom_ids) or (g2 in ball_geom_ids and g1 in right_geom_ids):
                right_contact = True
            if left_contact and right_contact:
                break
        return left_contact and right_contact

    def _ball_lifted_from_table(self, *, lift_multiplier: float) -> bool:
        if self._dyn_table_collision_geom_id is None:
            return True

        data = self._env.sim.data
        try:
            ball_z = float(data.body_xpos[self._dyn_ball_body_id][2])
            table_center_z = float(data.geom_xpos[self._dyn_table_collision_geom_id][2])
            table_half_z = float(self._env.sim.model.geom_size[self._dyn_table_collision_geom_id][2])
        except Exception:  # noqa: BLE001
            return True

        table_top_z = table_center_z + table_half_z
        ball_radius = float(self._dyn_ball_radius if self._dyn_ball_radius is not None else 0.02)
        return ball_z > table_top_z + max(0.001, float(lift_multiplier) * ball_radius)

    def _ball_close_to_gripper_center(self, *, max_dist: float) -> bool:
        if self._dyn_grip_site_id is None:
            return True
        data = self._env.sim.data
        try:
            ball_pos = np.asarray(data.body_xpos[self._dyn_ball_body_id], dtype=np.float64).reshape(3)
            grip_pos = np.asarray(data.site_xpos[self._dyn_grip_site_id], dtype=np.float64).reshape(3)
        except Exception:  # noqa: BLE001
            return True
        return float(np.linalg.norm(ball_pos - grip_pos)) <= float(max_dist)

    def _is_ball_grasped_legacy(self) -> bool:
        if len(self._dyn_left_finger_geom_ids) == 0 or len(self._dyn_right_finger_geom_ids) == 0:
            return False
        dual_contact = self._has_dual_finger_ball_contact(
            left_geom_ids=set(self._dyn_left_finger_geom_ids),
            right_geom_ids=set(self._dyn_right_finger_geom_ids),
        )
        if not dual_contact:
            return False
        # Keep legacy behavior unchanged.
        return self._ball_lifted_from_table(lift_multiplier=0.5)

    def _is_ball_grasped_strict(self) -> bool:
        if self._ball_grasp_strict_require_pad_contact:
            if len(self._dyn_left_fingerpad_geom_ids) == 0 or len(self._dyn_right_fingerpad_geom_ids) == 0:
                return False
            left_ids = set(self._dyn_left_fingerpad_geom_ids)
            right_ids = set(self._dyn_right_fingerpad_geom_ids)
        else:
            if len(self._dyn_left_finger_geom_ids) == 0 or len(self._dyn_right_finger_geom_ids) == 0:
                return False
            left_ids = set(self._dyn_left_finger_geom_ids)
            right_ids = set(self._dyn_right_finger_geom_ids)

        if not self._has_dual_finger_ball_contact(left_geom_ids=left_ids, right_geom_ids=right_ids):
            return False
        if not self._ball_lifted_from_table(lift_multiplier=self._ball_grasp_strict_lift_multiplier):
            return False
        if not self._ball_close_to_gripper_center(max_dist=self._ball_grasp_strict_grip_center_max_dist):
            return False
        return True

    def _apply_table_tilt(self, direction_deg: float | None, tilt_deg: float | None) -> None:
        self._resolve_dyn_handles()
        if self._dyn_table_body_id is None:
            return
        model = self._env.sim.model
        if tilt_deg is None or direction_deg is None:
            if self._dyn_base_table_quat is not None:
                model.body_quat[self._dyn_table_body_id] = self._dyn_base_table_quat.astype(
                    model.body_quat.dtype, copy=False
                )
            self._env.sim.forward()
            return

        quat = self._tilt_quat(direction_deg=float(direction_deg), tilt_deg=float(tilt_deg))
        model.body_quat[self._dyn_table_body_id] = quat.astype(model.body_quat.dtype, copy=False)
        self._env.sim.forward()

    def get_sim_state(self) -> np.ndarray:
        return np.asarray(self._env.get_sim_state(), dtype=np.float64).copy()

    def get_table_body_quat(self) -> np.ndarray | None:
        self._resolve_dyn_handles()
        if self._dyn_table_body_id is None:
            return None
        return np.asarray(self._env.sim.model.body_quat[self._dyn_table_body_id], dtype=np.float64).copy()

    def _restore_table_body_quat(self, quat: np.ndarray | None) -> None:
        self._resolve_dyn_handles()
        if self._dyn_table_body_id is None:
            return
        target = self._dyn_base_table_quat if quat is None else np.asarray(quat, dtype=np.float64)
        if target is None:
            return
        if target.shape != (4,):
            raise ValueError(f"Expected table-body quaternion with shape (4,), got {target.shape}.")
        model = self._env.sim.model
        model.body_quat[self._dyn_table_body_id] = target.astype(model.body_quat.dtype, copy=False)
        self._env.sim.forward()

    def _estimate_ball_radius(self, fallback: float = 0.02) -> float:
        self._resolve_dyn_handles()
        if self._dyn_ball_body_id is None:
            return float(fallback)
        model = self._env.sim.model
        candidates: list[float] = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != int(self._dyn_ball_body_id):
                continue
            r = float(model.geom_size[gid][0])
            if r > 1e-6:
                candidates.append(r)
        if not candidates:
            return float(fallback)
        return float(np.median(np.asarray(candidates, dtype=np.float64)))

    def _set_ball_linear_velocity(self, speed: float, direction_deg: float) -> RobotObservation:
        self._resolve_dyn_handles()
        if self._dyn_ball_joint_name is None:
            return self._env.env._get_observations()
        model = self._env.sim.model
        state = self._env.sim.get_state().flatten().copy()
        nq = int(model.nq)
        nv = int(model.nv)
        qvel = state[1 + nq : 1 + nq + nv]

        ball_joint_id = int(model.joint_name2id(self._dyn_ball_joint_name))
        dof_adr = int(model.jnt_dofadr[ball_joint_id])
        qvel[dof_adr : dof_adr + 6] = 0.0

        theta = math.radians(float(direction_deg))
        qvel[dof_adr + 0] = float(speed) * math.cos(theta)
        qvel[dof_adr + 1] = float(speed) * math.sin(theta)

        state[1 + nq : 1 + nq + nv] = qvel
        return self._env.set_init_state(state)

    def _place_ball_on_table(self, start_x: float, start_y: float) -> None:
        self._resolve_dyn_handles()
        if self._dyn_ball_joint_name is None or self._dyn_table_collision_geom_id is None:
            return
        model = self._env.sim.model
        data = self._env.sim.data
        table_gid = int(self._dyn_table_collision_geom_id)

        table_size = np.asarray(model.geom_size[table_gid], dtype=np.float64)
        half_x, half_y, half_z = float(table_size[0]), float(table_size[1]), float(table_size[2])
        x_lim = max(1e-4, half_x * float(self._init_plan_ball_start_xy_safety_scale))
        y_lim = max(1e-4, half_y * float(self._init_plan_ball_start_xy_safety_scale))
        x_local = float(np.clip(float(start_x), -x_lim, x_lim))
        y_local = float(np.clip(float(start_y), -y_lim, y_lim))

        table_pos = np.asarray(data.geom_xpos[table_gid], dtype=np.float64).copy()
        table_rot = np.asarray(data.geom_xmat[table_gid], dtype=np.float64).reshape(3, 3).copy()
        ball_radius = self._estimate_ball_radius(fallback=0.02)
        local_xyz = np.asarray(
            [x_local, y_local, half_z + ball_radius + float(self._init_plan_ball_start_z_clearance)],
            dtype=np.float64,
        )
        world_xyz = table_pos + table_rot @ local_xyz

        qpos = np.asarray(data.get_joint_qpos(self._dyn_ball_joint_name), dtype=np.float64).copy()
        qpos[:3] = world_xyz
        data.set_joint_qpos(self._dyn_ball_joint_name, qpos)
        data.set_joint_qvel(self._dyn_ball_joint_name, np.zeros(6, dtype=np.float64))
        self._env.sim.forward()

    def _set_ball_velocity_with_ramp(
        self,
        speed: float,
        direction_deg: float,
        settle_steps: int,
        ramp_steps: int,
    ) -> RobotObservation:
        no_op = np.asarray(get_libero_dummy_action(), dtype=np.float32)
        obs = self._set_ball_linear_velocity(speed=0.0, direction_deg=direction_deg)
        for _ in range(max(0, int(settle_steps))):
            obs, _, _, _ = self._env.step(no_op)

        n_ramp = max(1, int(ramp_steps))
        for i in range(1, n_ramp + 1):
            frac = float(i) / float(n_ramp)
            obs = self._set_ball_linear_velocity(speed=float(speed) * frac, direction_deg=direction_deg)
            obs, _, _, _ = self._env.step(no_op)
        return obs

    def _apply_init_plan_row(self, raw_obs: RobotObservation) -> tuple[RobotObservation, bool]:
        if len(self._init_plan_rows) == 0:
            return raw_obs, False
        if not self._init_plan_loop and self._init_plan_index >= len(self._init_plan_rows):
            return raw_obs, False

        row = self._init_plan_rows[self._init_plan_index % len(self._init_plan_rows)]
        self._init_plan_index += self._init_plan_stride

        direction_deg = self._safe_float(
            row.get("direction_deg", self._init_plan_default_direction_deg),
            self._init_plan_default_direction_deg,
        )
        speed = self._safe_float(row.get("speed", self._init_plan_default_speed), self._init_plan_default_speed)
        ball_start_x = self._safe_float(
            row.get("ball_start_x", self._init_plan_default_ball_start_x),
            self._init_plan_default_ball_start_x,
        )
        ball_start_y = self._safe_float(
            row.get("ball_start_y", self._init_plan_default_ball_start_y),
            self._init_plan_default_ball_start_y,
        )
        tilt_raw = row.get("tilt_deg", None)
        tilt_deg = self._safe_float(tilt_raw, 0.0) if tilt_raw is not None else None
        settle_steps = self._safe_int(
            row.get("launch_settle_steps", self._init_plan_launch_settle_steps),
            self._init_plan_launch_settle_steps,
        )
        ramp_steps = self._safe_int(
            row.get("launch_ramp_steps", self._init_plan_launch_ramp_steps),
            self._init_plan_launch_ramp_steps,
        )
        warmup_steps = self._safe_int(
            row.get("warmup_steps", self._init_plan_warmup_steps),
            self._init_plan_warmup_steps,
        )

        try:
            self._apply_table_tilt(direction_deg=direction_deg, tilt_deg=tilt_deg)
            self._place_ball_on_table(start_x=ball_start_x, start_y=ball_start_y)
            raw_obs = self._set_ball_velocity_with_ramp(
                speed=speed,
                direction_deg=direction_deg,
                settle_steps=settle_steps,
                ramp_steps=ramp_steps,
            )
            for _ in range(max(0, warmup_steps)):
                raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
            return raw_obs, True
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to apply LIBERO init-plan row on task=%s task_id=%d. Falling back to default reset.",
                self.task,
                self.task_id,
            )
            return raw_obs, False

    def _restore_episode_start_from_cache(self, *, consume: bool) -> RobotObservation:
        if self._episode_start_states is None:
            raise RuntimeError("Episode-start cache is not loaded.")
        cache_len = int(self._episode_start_states.shape[0])
        if consume:
            cache_idx = int(self._episode_start_state_index)
        else:
            if self._active_episode_start_cache_idx is None:
                raise RuntimeError("Cannot replay cached episode start before any cached episode has been consumed.")
            cache_idx = int(self._active_episode_start_cache_idx)
        if cache_idx < 0 or cache_idx >= cache_len:
            raise RuntimeError(
                f"Episode-start cache at {self._episode_start_states_path} contains {cache_len} states, "
                f"but env tried to restore episode index {cache_idx}."
            )
        table_quat = None
        if self._episode_start_table_body_quats is not None:
            table_quat = self._episode_start_table_body_quats[cache_idx]
        self._restore_table_body_quat(table_quat)
        raw_obs = self._env.set_init_state(self._episode_start_states[cache_idx].copy())
        self._active_episode_start_cache_idx = cache_idx
        if consume:
            self._episode_start_state_index += self._episode_start_state_stride
        return raw_obs

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def reset(self, seed=None, **kwargs):
        options = kwargs.pop("options", None)
        explicit_consume = kwargs.pop("consume_episode_start", None)
        if explicit_consume is None and isinstance(options, Mapping):
            explicit_consume = options.get("consume_episode_start")
        if explicit_consume is None:
            # Formal batch resets should advance the cache cursor. Auto-resets triggered by the vector env on
            # the step after termination arrive without an explicit seed/options payload and must replay the
            # currently active cached episode start instead of consuming the next one.
            consume_episode_start = self._episode_start_states is None or seed is not None
            if self._episode_start_states is not None and self._active_episode_start_cache_idx is None:
                consume_episode_start = True
        else:
            consume_episode_start = bool(explicit_consume)
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        used_init_plan = False
        if self._episode_start_states is not None:
            raw_obs = self._restore_episode_start_from_cache(consume=consume_episode_start)
            used_init_plan = True
        else:
            if self.init_states and self._init_states is not None:
                raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
                self.init_state_id += self._reset_stride  # Change init_state_id when reset
            raw_obs, used_init_plan = self._apply_init_plan_row(raw_obs)

        if not used_init_plan:
            # After reset, objects may be unstable (slightly floating, intersecting, etc.).
            # Step the simulator with a no-op action for a few frames so everything settles.
            # Increasing this value can improve determinism and reproducibility across resets.
            for _ in range(self.num_steps_wait):
                raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            # Keep the terminal observation intact. The enclosing VectorEnv performs the batched autoreset on the
            # next step, which is where we decide whether the cached episode cursor should advance.
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
            )
            try:
                out[suite_name][tid] = env_cls(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            except TypeError:
                out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}

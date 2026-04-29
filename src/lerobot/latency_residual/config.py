"""Configuration objects shared by the latency residual training stack."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.utils.constants import ACTION, OBS_STATE


def json_ready(value: Any) -> Any:
    """Recursively convert dataclass/path/numpy-friendly values into JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


@dataclass
class LatencyResidualDataConfig:
    """Inputs required to build the residual training dataset from sidecar + raw data."""

    repo_id: str = "libero_dyn_mini"
    root: str | None = "/share/project/wujiling/datasets/libero_dyn_mini"
    revision: str | None = None
    episodes: list[int] | None = None
    sidecar_path: str = "/share/project/wujiling/datasets/libero_dyn_mini_sidecar"
    sidecar_parquet_name: str = "pred_action_chunks.parquet"
    sidecar_summary_name: str = "summary.json"
    state_key: str = OBS_STATE
    ball_key: str | None = "observation.ball_pos"
    action_key: str = ACTION
    history_len: int = 2
    delay_values: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    step_offsets: list[int] = field(default_factory=lambda: list(range(8)))
    # Set to 0 to disable local-plan context entirely. This makes the residual
    # model depend only on stale/current observations, base action, delay, and
    # step offset.
    local_plan_horizon: int = 0
    step_weight_gamma: float = 1.0
    max_sidecar_rows: int | None = None

    def validate(self) -> None:
        if self.history_len <= 0:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}.")
        if self.local_plan_horizon < 0:
            raise ValueError(
                f"local_plan_horizon must be >= 0, got {self.local_plan_horizon}."
            )
        if not self.delay_values:
            raise ValueError("delay_values must not be empty.")
        if not self.step_offsets:
            raise ValueError("step_offsets must not be empty.")
        if min(self.delay_values) < 0:
            raise ValueError(f"delay_values must be non-negative, got {self.delay_values}.")
        if min(self.step_offsets) < 0:
            raise ValueError(f"step_offsets must be non-negative, got {self.step_offsets}.")
        if self.step_weight_gamma <= 0:
            raise ValueError(
                f"step_weight_gamma must be > 0, got {self.step_weight_gamma}."
            )


@dataclass
class LatencyResidualFeatureSpec:
    """Shape summary inferred from the offline dataset + sidecar."""

    state_dim: int
    ball_dim: int
    action_dim: int
    chunk_len: int
    history_len: int
    local_plan_horizon: int

    @property
    def uses_ball(self) -> bool:
        return self.ball_dim > 0


@dataclass
class LatencyResidualModelConfig:
    """Model-side hyperparameters for the residual MLP."""

    state_dim: int = 8
    ball_dim: int = 3
    action_dim: int = 7
    history_len: int = 2
    # Keep this at 0 for the first ablation without short-horizon action-plan
    # context. Set it to >0 later if we want to feed a short suffix of the base
    # chunk back into the residual model.
    local_plan_horizon: int = 0
    max_delay_value: int = 6
    max_step_offset: int = 7
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    dropout: float = 0.0
    delay_embedding_dim: int = 16
    step_offset_embedding_dim: int = 16
    residual_scale: float = 0.25
    residual_l2_weight: float = 1e-4

    def sync_from_data(
        self,
        feature_spec: LatencyResidualFeatureSpec,
        data_cfg: LatencyResidualDataConfig,
    ) -> None:
        """Freeze model shapes against the actual sidecar-backed training data."""
        self.state_dim = int(feature_spec.state_dim)
        self.ball_dim = int(feature_spec.ball_dim)
        self.action_dim = int(feature_spec.action_dim)
        self.history_len = int(feature_spec.history_len)
        self.local_plan_horizon = int(feature_spec.local_plan_horizon)
        self.max_delay_value = int(max(data_cfg.delay_values))
        self.max_step_offset = int(max(data_cfg.step_offsets))

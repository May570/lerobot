"""Latency-compensation residual model utilities.

This package contains the first offline training path for the realtime residual
adapter discussed for Diffusion Policy chunking latency compensation.
"""

from .checkpoint import (
    LATENCY_RESIDUAL_CONFIG_NAME,
    LATENCY_RESIDUAL_METADATA_NAME,
    LATENCY_RESIDUAL_MODEL_NAME,
    load_latency_residual_checkpoint,
    save_latency_residual_checkpoint,
)
from .config import (
    LatencyResidualDataConfig,
    LatencyResidualFeatureSpec,
    LatencyResidualModelConfig,
    json_ready,
)
from .dataset import LatencyResidualDataset, make_latency_residual_episode_split
from .model import LatencyResidualMLP

__all__ = [
    "LATENCY_RESIDUAL_CONFIG_NAME",
    "LATENCY_RESIDUAL_METADATA_NAME",
    "LATENCY_RESIDUAL_MODEL_NAME",
    "LatencyResidualDataConfig",
    "LatencyResidualDataset",
    "LatencyResidualFeatureSpec",
    "LatencyResidualMLP",
    "LatencyResidualModelConfig",
    "json_ready",
    "load_latency_residual_checkpoint",
    "make_latency_residual_episode_split",
    "save_latency_residual_checkpoint",
]

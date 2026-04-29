"""Checkpoint helpers for the latency residual model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import LatencyResidualModelConfig, json_ready
from .model import LatencyResidualMLP

LATENCY_RESIDUAL_CONFIG_NAME = "config.json"
LATENCY_RESIDUAL_MODEL_NAME = "model.safetensors"
LATENCY_RESIDUAL_METADATA_NAME = "metadata.json"


def save_latency_residual_checkpoint(
    model: LatencyResidualMLP,
    config: LatencyResidualModelConfig,
    save_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a self-contained residual checkpoint directory."""
    from safetensors.torch import save_model as save_model_as_safetensor

    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    save_model_as_safetensor(model_to_save, str(path / LATENCY_RESIDUAL_MODEL_NAME))
    (path / LATENCY_RESIDUAL_CONFIG_NAME).write_text(
        json.dumps(json_ready(config), indent=2, ensure_ascii=False)
    )
    if metadata is not None:
        (path / LATENCY_RESIDUAL_METADATA_NAME).write_text(
            json.dumps(json_ready(metadata), indent=2, ensure_ascii=False)
        )
    return path


def load_latency_residual_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | None = None,
) -> tuple[LatencyResidualMLP, LatencyResidualModelConfig]:
    """Load a residual checkpoint saved with save_latency_residual_checkpoint."""
    from safetensors.torch import load_model as load_model_as_safetensor

    path = Path(checkpoint_dir)
    config_path = path / LATENCY_RESIDUAL_CONFIG_NAME
    model_path = path / LATENCY_RESIDUAL_MODEL_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"Missing residual config file: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing residual model file: {model_path}")

    config = LatencyResidualModelConfig(**json.loads(config_path.read_text()))
    model = LatencyResidualMLP(config)
    load_model_as_safetensor(model, str(model_path), strict=True)
    if device is not None:
        model.to(device)
    model.eval()
    return model, config

"""Small residual MLP used for latency compensation at action execution time."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import LatencyResidualModelConfig


class LatencyResidualMLP(nn.Module):
    """Predict a small action residual from stale/current low-dimensional context.

    The model is intentionally lightweight so it can later be used inline inside a
    realtime evaluation loop after the slow diffusion chunk has already been
    generated.
    """

    def __init__(self, config: LatencyResidualModelConfig):
        super().__init__()
        self.config = config

        if config.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be >= 1, got {config.num_hidden_layers}."
            )
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {config.hidden_dim}.")
        if config.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {config.action_dim}.")

        self.delay_embedding = nn.Embedding(
            int(config.max_delay_value) + 1,
            int(config.delay_embedding_dim),
        )
        self.step_offset_embedding = nn.Embedding(
            int(config.max_step_offset) + 1,
            int(config.step_offset_embedding_dim),
        )

        input_dim = (
            2 * int(config.history_len) * int(config.state_dim)
            + 2 * int(config.history_len) * int(config.ball_dim)
            + int(config.action_dim)
            + int(config.local_plan_horizon) * int(config.action_dim)
            + int(config.delay_embedding_dim)
            + int(config.step_offset_embedding_dim)
        )

        layers: list[nn.Module] = []
        hidden_dim = int(config.hidden_dim)
        current_dim = input_dim
        for _ in range(int(config.num_hidden_layers)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(float(config.dropout)))
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, int(config.action_dim))

    def _build_features(self, batch: dict[str, Tensor]) -> Tensor:
        dtype = self.head.weight.dtype
        device = self.head.weight.device

        pieces = [
            batch["anchor_state_history"].to(device=device, dtype=dtype).flatten(start_dim=1),
            batch["current_state_history"].to(device=device, dtype=dtype).flatten(start_dim=1),
            batch["base_action"].to(device=device, dtype=dtype),
        ]
        if int(self.config.local_plan_horizon) > 0:
            if "local_plan" not in batch:
                raise KeyError(
                    "local_plan is required by the model config but is missing from the batch."
                )
            pieces.append(batch["local_plan"].to(device=device, dtype=dtype).flatten(start_dim=1))
        if "anchor_ball_history" in batch and "current_ball_history" in batch:
            pieces.extend(
                [
                    batch["anchor_ball_history"].to(device=device, dtype=dtype).flatten(start_dim=1),
                    batch["current_ball_history"].to(device=device, dtype=dtype).flatten(start_dim=1),
                ]
            )

        delay_steps = batch["delay_steps"].to(device=device, dtype=torch.long)
        step_offset = batch["step_offset"].to(device=device, dtype=torch.long)
        pieces.append(self.delay_embedding(delay_steps))
        pieces.append(self.step_offset_embedding(step_offset))
        return torch.cat(pieces, dim=-1)

    def predict_residual(self, batch: dict[str, Tensor]) -> Tensor:
        hidden = self.backbone(self._build_features(batch))
        residual = self.head(hidden)
        return float(self.config.residual_scale) * torch.tanh(residual)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        residual = self.predict_residual(batch)
        base_action = batch["base_action"].to(device=residual.device, dtype=residual.dtype)
        corrected_action = base_action + residual
        return {
            "residual": residual,
            "base_action": base_action,
            "corrected_action": corrected_action,
        }

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
        outputs = self(batch)
        target_action = batch["target_action"].to(
            device=outputs["corrected_action"].device,
            dtype=outputs["corrected_action"].dtype,
        )
        sample_weight = batch["sample_weight"].to(
            device=outputs["corrected_action"].device,
            dtype=outputs["corrected_action"].dtype,
        ).unsqueeze(-1)

        action_loss = F.smooth_l1_loss(
            outputs["corrected_action"],
            target_action,
            reduction="none",
        )
        weight_denom = sample_weight.sum().clamp_min(1e-6)
        weighted_action_loss = (action_loss * sample_weight).sum() / (
            weight_denom * float(self.config.action_dim)
        )

        residual_l2 = (outputs["residual"].square().sum(dim=-1) * sample_weight.squeeze(-1)).sum() / weight_denom
        total_loss = weighted_action_loss + float(self.config.residual_l2_weight) * residual_l2

        with torch.no_grad():
            base_l1 = (
                (outputs["base_action"] - target_action).abs() * sample_weight
            ).sum() / (weight_denom * float(self.config.action_dim))
            corrected_l1 = (
                (outputs["corrected_action"] - target_action).abs() * sample_weight
            ).sum() / (weight_denom * float(self.config.action_dim))
            target_residual_l1 = (
                (batch["target_residual"].to(outputs["residual"].device, outputs["residual"].dtype) - outputs["residual"]).abs()
                * sample_weight
            ).sum() / (weight_denom * float(self.config.action_dim))
            metrics = {
                "loss": float(total_loss.detach().item()),
                "action_loss": float(weighted_action_loss.detach().item()),
                "residual_l2": float(residual_l2.detach().item()),
                "base_l1": float(base_l1.detach().item()),
                "corrected_l1": float(corrected_l1.detach().item()),
                "target_residual_l1": float(target_residual_l1.detach().item()),
                "improvement_l1": float((base_l1 - corrected_l1).detach().item()),
                "mean_residual_norm": float(outputs["residual"].norm(dim=-1).mean().detach().item()),
            }
        return total_loss, metrics, outputs

    @torch.inference_mode()
    def predict_corrected_action(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convenience wrapper for runtime use."""
        outputs = self(batch)
        return {
            "corrected_action": outputs["corrected_action"],
            "residual": outputs["residual"],
        }

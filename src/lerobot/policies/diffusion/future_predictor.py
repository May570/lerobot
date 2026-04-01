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
from __future__ import annotations

from torch import Tensor, nn


class FuturePredictor(nn.Module):
    """Lightweight predictor for one-step future latent features.

    Inputs are latent observation features with shape (B, S_in, D), where `S_in` is the
    number of history steps fed to the predictor and `D` is the latent observation feature
    dimension from the existing Diffusion Policy observation encoder.
    """

    def __init__(
        self,
        predictor_type: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_input_steps: int = 1,
    ) -> None:
        super().__init__()
        self.predictor_type = predictor_type
        self.num_input_steps = num_input_steps

        if predictor_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim * num_input_steps, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        elif predictor_type == "gru":
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.readout = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported future predictor type: {predictor_type}")

    def forward(self, input_seq: Tensor) -> Tensor:
        """
        Args:
            input_seq: Tensor of shape (B, S_in, D).

        Returns:
            Prediction tensor of shape (B, D_out).
        """
        if input_seq.ndim != 3:
            raise ValueError(
                f"`input_seq` should have shape (B, S_in, D). Got {tuple(input_seq.shape)}."
            )
        if input_seq.shape[1] != self.num_input_steps:
            raise ValueError(
                f"`input_seq` has S_in={input_seq.shape[1]} but expected {self.num_input_steps}."
            )

        if self.predictor_type == "mlp":
            return self.net(input_seq.flatten(start_dim=1))

        _, h_n = self.gru(input_seq)
        return self.readout(h_n[-1])

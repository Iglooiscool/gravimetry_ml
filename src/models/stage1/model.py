"""Stage 1 model definition."""

from __future__ import annotations

import torch
from torch import nn

from ..common import build_mlp_with_dropout


class Stage1Regressor(nn.Module):
    """Stage 1 network that predicts coefficients from gradient data."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 256, 128),
        dropout_rates: tuple[float, ...] = (0.2, 0.2, 0.2),
    ):
        super().__init__()
        self.network = build_mlp_with_dropout(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


__all__ = ["Stage1Regressor"]

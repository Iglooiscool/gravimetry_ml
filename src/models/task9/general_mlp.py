"""General Task 9 mask MLP."""

from __future__ import annotations

import torch
from torch import nn

from ..common import build_mlp_with_dropout


class Task9GeneralMaskMLP(nn.Module):
    """General MLP that maps coefficient vectors to flattened masks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 1024, 2048),
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


__all__ = ["Task9GeneralMaskMLP"]

"""Architecture used by the official one-model workflow."""

from __future__ import annotations

import torch
from torch import nn

from ..common import build_mlp_with_dropout
from ..stage2.model import Stage2CoordConvDecoder


class GradientToMaskModel(Stage2CoordConvDecoder):
    """Predict mask logits directly from noisy gradient features.

    The inherited coordinate-aware decoder is intentionally reused here. Its
    input is gradient data, not predicted coefficients, which makes this a
    one-model system rather than a two-model stack.
    """


class GradientToMaskMLP(nn.Module):
    """PDF-style direct gradient-to-mask MLP for the one-model baseline."""

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


__all__ = ["GradientToMaskMLP", "GradientToMaskModel"]

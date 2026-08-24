"""Architectures used by the one-model workflow."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

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


class MultiTaskGradientModel(nn.Module):
    """Shared gradient encoder with coefficient and coordinate-mask heads."""

    def __init__(
        self,
        input_dim: int,
        coefficient_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 1024),
        dropout_rates: tuple[float, ...] = (0.1, 0.1),
        latent_grid_size: int = 16,
        latent_channels: int = 160,
        decoder_channels: tuple[int, ...] = (160, 128, 96, 64, 32),
    ):
        super().__init__()
        grid_size = int(round(math.sqrt(output_dim)))
        if grid_size * grid_size != output_dim:
            raise ValueError("MultiTaskGradientModel requires a square output grid")
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        shared_hidden_dims = hidden_dims[:-1]
        shared_dim = hidden_dims[-1]
        shared_dropout = dropout_rates[: len(shared_hidden_dims)]
        self.shared_encoder = build_mlp_with_dropout(
            input_dim=input_dim,
            output_dim=shared_dim,
            hidden_dims=shared_hidden_dims,
            dropout_rates=shared_dropout,
        )
        self.coefficient_head = nn.Linear(shared_dim, coefficient_dim)
        self.latent_grid_size = latent_grid_size
        self.latent_channels = latent_channels
        self.output_grid_size = grid_size
        self.mask_projection = nn.Linear(shared_dim, latent_channels * latent_grid_size * latent_grid_size)

        decoder_layers: list[nn.Module] = []
        in_channels = latent_channels
        for out_channels in decoder_channels:
            decoder_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)
        self.mask_head = nn.Conv2d(in_channels + 2, 1, kernel_size=1)

        coordinate_values = torch.linspace(-1.0, 1.0, grid_size)
        y_coordinates, x_coordinates = torch.meshgrid(coordinate_values, coordinate_values, indexing="ij")
        self.register_buffer(
            "coordinate_grid",
            torch.stack((x_coordinates, y_coordinates), dim=0).unsqueeze(0),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_encoder(inputs)
        coefficient_predictions = self.coefficient_head(shared_features)
        projected = self.mask_projection(shared_features)
        feature_map = projected.reshape(
            inputs.shape[0],
            self.latent_channels,
            self.latent_grid_size,
            self.latent_grid_size,
        )
        decoded = self.decoder(feature_map)
        if decoded.shape[-1] != self.output_grid_size or decoded.shape[-2] != self.output_grid_size:
            decoded = F.interpolate(
                decoded,
                size=(self.output_grid_size, self.output_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        coordinate_grid = self.coordinate_grid.expand(inputs.shape[0], -1, -1, -1)
        mask_logits = self.mask_head(torch.cat((decoded, coordinate_grid), dim=1))
        return mask_logits.reshape(inputs.shape[0], -1), coefficient_predictions


__all__ = ["GradientToMaskMLP", "GradientToMaskModel", "MultiTaskGradientModel"]

"""Stage 2 model definition."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..common import build_mlp_with_dropout


class Stage2MaskPredictor(nn.Module):
    """Stage 2 network that predicts flattened mask logits from coefficients."""

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


class Stage2ConvDecoder(nn.Module):
    """Stage 2 decoder that projects coefficients into a small spatial latent map."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 1024),
        dropout_rates: tuple[float, ...] = (0.2, 0.2),
        latent_grid_size: int = 8,
        latent_channels: int = 64,
        decoder_channels: tuple[int, ...] = (64, 32, 16),
    ):
        super().__init__()
        grid_size = int(round(math.sqrt(output_dim)))
        if grid_size * grid_size != output_dim:
            raise ValueError("Stage2ConvDecoder requires a square output grid")
        if latent_grid_size <= 0:
            raise ValueError("latent_grid_size must be positive")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")

        self.output_grid_size = grid_size
        self.latent_grid_size = latent_grid_size
        self.latent_channels = latent_channels
        latent_dim = latent_channels * latent_grid_size * latent_grid_size
        self.projection = build_mlp_with_dropout(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
        )

        decoder_layers: list[nn.Module] = []
        in_channels = latent_channels
        for out_channels in decoder_channels:
            decoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            decoder_layers.append(nn.ReLU())
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.projection(inputs)
        feature_map = projected.reshape(
            inputs.shape[0],
            self.latent_channels,
            self.latent_grid_size,
            self.latent_grid_size,
        )
        decoded = self.decoder(feature_map)
        if decoded.shape[-1] != self.output_grid_size or decoded.shape[-2] != self.output_grid_size:
            decoded = F.interpolate(decoded, size=(self.output_grid_size, self.output_grid_size), mode="bilinear", align_corners=False)
        logits = self.output_head(decoded)
        return logits.reshape(inputs.shape[0], -1)


class Stage2CoordConvDecoder(nn.Module):
    """Stage 2 decoder that adds normalized x/y coordinate channels before the output head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 1024),
        dropout_rates: tuple[float, ...] = (0.2, 0.2),
        latent_grid_size: int = 8,
        latent_channels: int = 64,
        decoder_channels: tuple[int, ...] = (64, 32, 16),
    ):
        super().__init__()
        grid_size = int(round(math.sqrt(output_dim)))
        if grid_size * grid_size != output_dim:
            raise ValueError("Stage2CoordConvDecoder requires a square output grid")
        if latent_grid_size <= 0:
            raise ValueError("latent_grid_size must be positive")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")

        self.output_grid_size = grid_size
        self.latent_grid_size = latent_grid_size
        self.latent_channels = latent_channels
        latent_dim = latent_channels * latent_grid_size * latent_grid_size
        self.projection = build_mlp_with_dropout(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
        )

        decoder_layers: list[nn.Module] = []
        in_channels = latent_channels
        for out_channels in decoder_channels:
            decoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            decoder_layers.append(nn.ReLU())
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_head = nn.Conv2d(in_channels + 2, 1, kernel_size=1)

        coordinate_values = torch.linspace(-1.0, 1.0, grid_size)
        y_coordinates, x_coordinates = torch.meshgrid(coordinate_values, coordinate_values, indexing="ij")
        coordinate_grid = torch.stack((x_coordinates, y_coordinates), dim=0)
        self.register_buffer("coordinate_grid", coordinate_grid.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.projection(inputs)
        feature_map = projected.reshape(
            inputs.shape[0],
            self.latent_channels,
            self.latent_grid_size,
            self.latent_grid_size,
        )
        decoded = self.decoder(feature_map)
        if decoded.shape[-1] != self.output_grid_size or decoded.shape[-2] != self.output_grid_size:
            decoded = F.interpolate(decoded, size=(self.output_grid_size, self.output_grid_size), mode="bilinear", align_corners=False)
        coordinate_grid = self.coordinate_grid.expand(inputs.shape[0], -1, -1, -1)
        logits = self.output_head(torch.cat((decoded, coordinate_grid), dim=1))
        return logits.reshape(inputs.shape[0], -1)


__all__ = ["Stage2MaskPredictor", "Stage2ConvDecoder", "Stage2CoordConvDecoder"]

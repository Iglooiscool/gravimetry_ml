"""Shared model-building helpers used by both stages."""

from __future__ import annotations

from torch import nn


def build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    """Build a feedforward stack with linear and ReLU layers."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def build_mlp_with_dropout(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    dropout_rates: tuple[float, ...] = (),
) -> nn.Sequential:
    """Build a feedforward stack with optional dropout after each hidden layer."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for layer_index, hidden_dim in enumerate(hidden_dims):
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
        if layer_index < len(dropout_rates) and dropout_rates[layer_index] > 0:
            layers.append(nn.Dropout(dropout_rates[layer_index]))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


__all__ = ["build_mlp", "build_mlp_with_dropout"]

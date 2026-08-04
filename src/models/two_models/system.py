"""Container describing the connected two-model system."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass
class TwoModelSystem:
    """The gradient-to-coefficient model followed by the mask model."""

    coefficient_model: nn.Module
    mask_model: nn.Module


__all__ = ["TwoModelSystem"]

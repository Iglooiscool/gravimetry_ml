"""Container describing the experimental three-model system."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass
class ThreeModelSystem:
    """Coefficient model, general mask model, and specialist mask model."""

    coefficient_model: nn.Module
    general_mask_model: nn.Module
    specialist_mask_model: nn.Module | None = None


__all__ = ["ThreeModelSystem"]

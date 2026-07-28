"""Grid construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Settings for the square grid used across the project."""

    grid_size: int = 200
    x_min: float = -1.0
    x_max: float = 1.0

    def validate(self) -> None:
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")


@dataclass
class GridData:
    """The grid arrays and spacing values after the grid has been built."""

    x: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dA: float


def create_grid(spec: GridSpec) -> GridData:
    """Build the square mesh used to draw shapes and approximate integrals."""

    spec.validate()
    x = np.linspace(spec.x_min, spec.x_max, spec.grid_size)
    X, Y = np.meshgrid(x, x)
    dx = float(x[1] - x[0])
    dA = dx * dx
    return GridData(x=x, X=X, Y=Y, dx=dx, dA=dA)


__all__ = ["GridSpec", "GridData", "create_grid"]

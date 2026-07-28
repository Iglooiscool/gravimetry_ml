"""Helpers shared by shape specifications and sampling."""

from __future__ import annotations

import numpy as np


def rotated_coords(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    center: tuple[float, float],
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate grid coordinates into a shape's local frame."""

    center_x, center_y = center
    shifted_x = grid_x - center_x
    shifted_y = grid_y - center_y
    rotated_x = shifted_x * np.cos(theta) + shifted_y * np.sin(theta)
    rotated_y = -shifted_x * np.sin(theta) + shifted_y * np.cos(theta)
    return rotated_x, rotated_y


__all__ = ["rotated_coords"]

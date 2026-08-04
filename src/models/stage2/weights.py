"""Per-pixel weighting helpers for Stage 2 losses."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_shape_edge_pixel_weights(
    masks: np.ndarray,
    shape_types: tuple[str, ...],
    grid_size: int,
    edge_weight: float,
    edge_width: int,
    edge_weight_mode: str = "rectangle",
    annulus_edge_weight: float = 1.0,
    annulus_edge_width: int | None = None,
) -> np.ndarray:
    """Build per-pixel weights that emphasize selected shape boundaries."""

    pixel_weights = np.ones_like(masks, dtype=np.float32)
    if edge_weight <= 1.0 and annulus_edge_weight <= 1.0:
        return pixel_weights
    if edge_width < 0:
        raise ValueError("edge_width must be non-negative")
    annulus_width = edge_width if annulus_edge_width is None else int(annulus_edge_width)
    if annulus_width < 0:
        raise ValueError("annulus_edge_width must be non-negative")

    if edge_weight_mode == "rectangle":
        weighted_shape_types = {"rectangle"}
    elif edge_weight_mode == "all":
        weighted_shape_types = set(shape_types)
    else:
        raise ValueError("edge_weight_mode must be 'rectangle' or 'all'")

    structure = np.ones((3, 3), dtype=bool)
    for sample_index, shape_type in enumerate(shape_types):
        if shape_type not in weighted_shape_types:
            continue

        mask = masks[sample_index].reshape(grid_size, grid_size).astype(bool)
        eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
        boundary = np.logical_xor(mask, eroded)
        if edge_width > 0:
            boundary = ndimage.binary_dilation(boundary, structure=structure, iterations=edge_width)

        weight_map = np.ones((grid_size, grid_size), dtype=np.float32)
        weight_map[boundary] = float(edge_weight)

        if shape_type == "annulus" and annulus_edge_weight > 1.0:
            annulus_eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
            annulus_boundary = np.logical_xor(mask, annulus_eroded)
            if annulus_width > 0:
                annulus_boundary = ndimage.binary_dilation(annulus_boundary, structure=structure, iterations=annulus_width)
            weight_map[annulus_boundary] = np.maximum(weight_map[annulus_boundary], float(annulus_edge_weight))

        pixel_weights[sample_index] = weight_map.reshape(-1)

    return pixel_weights


def compute_rectangle_edge_pixel_weights(
    masks: np.ndarray,
    shape_types: tuple[str, ...],
    grid_size: int,
    edge_weight: float,
    edge_width: int,
) -> np.ndarray:
    """Backward-compatible rectangle-only edge-weight helper."""

    return compute_shape_edge_pixel_weights(
        masks=masks,
        shape_types=shape_types,
        grid_size=grid_size,
        edge_weight=edge_weight,
        edge_width=edge_width,
        edge_weight_mode="rectangle",
    )


__all__ = ["compute_rectangle_edge_pixel_weights", "compute_shape_edge_pixel_weights"]

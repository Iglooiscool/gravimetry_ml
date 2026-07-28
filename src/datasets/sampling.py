"""Dataset-facing shape sampling helpers."""

from __future__ import annotations

import numpy as np

from shapes import ShapeSamplingConfig, sample_weighted_shape


def sample_two_stage_shape(
    random_generator: np.random.Generator,
    coefficient_order: int,
    center_scale: float,
    sampling_config: ShapeSamplingConfig | None = None,
    shape_weights_override: tuple[tuple[str, float], ...] | None = None,
):
    """Sample one shape for the current two-stage training distribution."""

    return sample_weighted_shape(
        rng=random_generator,
        n_value=coefficient_order,
        rho=center_scale,
        config=sampling_config,
        shape_weights_override=shape_weights_override,
    )


__all__ = ["sample_two_stage_shape"]

"""Purpose: expose the public shape API from focused shape modules."""

from .base import ShapeSpec
from .fixed import create_fixed_benchmark_shapes
from .sampling import ShapeSamplingConfig, random_center, sample_random_shape, sample_weighted_shape
from .specs import (
    AnnulusSpec,
    CircleSpec,
    EllipseSpec,
    RectangleSpec,
    TwoCirclesSpec,
)
from .utils import rotated_coords

__all__ = [
    "ShapeSpec",
    "EllipseSpec",
    "CircleSpec",
    "RectangleSpec",
    "AnnulusSpec",
    "TwoCirclesSpec",
    "ShapeSamplingConfig",
    "random_center",
    "sample_random_shape",
    "sample_weighted_shape",
    "create_fixed_benchmark_shapes",
    "rotated_coords",
]

"""Random shape generation rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ShapeSpec
from .specs import AnnulusSpec, CircleSpec, EllipseSpec, RectangleSpec, TwoCirclesSpec


@dataclass(frozen=True)
class ShapeSamplingConfig:
    """Ranges used when randomly sampling shape parameters."""

    ellipse_a: tuple[float, float] = (0.2, 0.8)
    ellipse_b: tuple[float, float] = (0.1, 0.6)
    circle_radius: tuple[float, float] = (0.15, 0.55)
    rectangle_width: tuple[float, float] = (0.2, 0.9)
    rectangle_height: tuple[float, float] = (0.2, 0.7)
    annulus_outer: tuple[float, float] = (0.3, 0.8)
    annulus_inner: tuple[float, float] = (0.05, 0.4)
    two_circles_radius: tuple[float, float] = (0.1, 0.35)
    center_min: float = -0.6
    center_max: float = 0.6


def random_center(rng: np.random.Generator, low_value: float, high_value: float) -> tuple[float, float]:
    """Sample a center point from a square box."""

    return float(rng.uniform(low_value, high_value)), float(rng.uniform(low_value, high_value))


def sample_random_shape(
    rng: np.random.Generator,
    shape_type: str,
    config: ShapeSamplingConfig | None = None,
) -> ShapeSpec:
    """Sample one random shape from a chosen shape family."""

    current_config = config or ShapeSamplingConfig()
    rotation_angle = float(rng.uniform(0.0, np.pi))

    if shape_type == "ellipse":
        return EllipseSpec(
            a=float(rng.uniform(*current_config.ellipse_a)),
            b=float(rng.uniform(*current_config.ellipse_b)),
            theta=rotation_angle,
            center=random_center(rng, current_config.center_min, current_config.center_max),
        )
    if shape_type == "rectangle":
        return RectangleSpec(
            width=float(rng.uniform(*current_config.rectangle_width)),
            height=float(rng.uniform(*current_config.rectangle_height)),
            theta=rotation_angle,
            center=random_center(rng, current_config.center_min, current_config.center_max),
        )
    if shape_type == "circle":
        return CircleSpec(
            radius=float(rng.uniform(*current_config.circle_radius)),
            center=random_center(rng, current_config.center_min, current_config.center_max),
        )
    if shape_type == "annulus":
        outer_radius = float(rng.uniform(*current_config.annulus_outer))
        inner_radius = float(rng.uniform(*current_config.annulus_inner))
        if inner_radius >= outer_radius:
            inner_radius = outer_radius * 0.5
        return AnnulusSpec(
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            center=random_center(rng, current_config.center_min, current_config.center_max),
        )
    if shape_type == "two_circles":
        return TwoCirclesSpec(
            radius1=float(rng.uniform(*current_config.two_circles_radius)),
            center1=random_center(rng, current_config.center_min, current_config.center_max),
            radius2=float(rng.uniform(*current_config.two_circles_radius)),
            center2=random_center(rng, current_config.center_min, current_config.center_max),
        )
    raise ValueError(f"Unsupported shape_type: {shape_type}")


def sample_weighted_shape(
    rng: np.random.Generator,
    n_value: int,
    rho: float,
    config: ShapeSamplingConfig | None = None,
    shape_weights_override: tuple[tuple[str, float], ...] | None = None,
) -> ShapeSpec:
    """Sample one random shape using the current weighted shape rules."""

    current_config = config or ShapeSamplingConfig()
    if shape_weights_override is not None:
        shape_weights = shape_weights_override
    elif n_value >= 6:
        shape_weights = (
            ("two_circles", 0.30),
            ("annulus", 0.25),
            ("rectangle", 0.25),
            ("ellipse", 0.15),
            ("circle", 0.05),
        )
    else:
        shape_weights = (
            ("two_circles", 0.45),
            ("annulus", 0.10),
            ("rectangle", 0.20),
            ("ellipse", 0.15),
            ("circle", 0.10),
        )

    random_draw = float(rng.random())
    cumulative_probability = 0.0
    selected_shape_type = "ellipse"
    for shape_type, shape_weight in shape_weights:
        cumulative_probability += shape_weight
        if random_draw < cumulative_probability:
            selected_shape_type = shape_type
            break

    if selected_shape_type == "circle":
        sampled_center = random_center(rng, current_config.center_min * rho, current_config.center_max * rho)
        return CircleSpec(radius=0.5, center=sampled_center)

    return sample_random_shape(rng, selected_shape_type, current_config)


__all__ = [
    "ShapeSamplingConfig",
    "random_center",
    "sample_random_shape",
    "sample_weighted_shape",
]

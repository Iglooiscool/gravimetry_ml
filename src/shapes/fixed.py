"""Fixed benchmark shape sets."""

from __future__ import annotations

import numpy as np

from .base import ShapeSpec
from .specs import AnnulusSpec, CircleSpec, EllipseSpec, RectangleSpec, TwoCirclesSpec


def create_fixed_benchmark_shapes() -> list[tuple[str, ShapeSpec]]:
    """Return the fixed benchmark shapes used for project comparisons."""

    return [
        ("Ellipse (a=0.6, b=0.3, theta=45)", EllipseSpec(a=0.6, b=0.3, theta=np.pi / 4.0, center=(0.1, 0.1))),
        ("Circle (radius=0.5)", CircleSpec(radius=0.5, center=(-0.1, -0.1))),
        ("Annulus (R=0.7, r=0.3)", AnnulusSpec(outer_radius=0.7, inner_radius=0.3, center=(0.0, 0.0))),
        ("Rectangle (0.8 x 0.4, theta=30)", RectangleSpec(width=0.8, height=0.4, theta=np.pi / 6.0, center=(0.2, -0.1))),
        ("Two Disconnected Circles", TwoCirclesSpec(radius1=0.3, center1=(-0.4, 0.3), radius2=0.25, center2=(0.4, -0.2))),
    ]


__all__ = ["create_fixed_benchmark_shapes"]

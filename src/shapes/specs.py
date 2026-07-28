"""Concrete shape specifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ShapeSpec
from .utils import rotated_coords


@dataclass(frozen=True)
class CircleSpec(ShapeSpec):
    """A filled circle defined by a radius and a center."""

    radius: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def type(self) -> str:
        return "circle"

    def validate(self) -> None:
        if self.radius <= 0:
            raise ValueError("circle radius must be positive")

    def compute_mask(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        self.validate()
        center_x, center_y = self.center
        distance_from_center = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
        return distance_from_center <= self.radius


@dataclass(frozen=True)
class EllipseSpec(ShapeSpec):
    """A rotated ellipse defined by two axis lengths, an angle, and a center."""

    a: float
    b: float
    theta: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def type(self) -> str:
        return "ellipse"

    def validate(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("ellipse axes a and b must be positive")

    def compute_mask(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        self.validate()
        rotated_x, rotated_y = rotated_coords(grid_x, grid_y, self.center, self.theta)
        return (rotated_x / self.a) ** 2 + (rotated_y / self.b) ** 2 <= 1.0


@dataclass(frozen=True)
class RectangleSpec(ShapeSpec):
    """A rotated rectangle defined by width, height, angle, and center."""

    width: float
    height: float
    theta: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def type(self) -> str:
        return "rectangle"

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("rectangle width and height must be positive")

    def compute_mask(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        self.validate()
        rotated_x, rotated_y = rotated_coords(grid_x, grid_y, self.center, self.theta)
        return (np.abs(rotated_x) <= self.width / 2.0) & (np.abs(rotated_y) <= self.height / 2.0)


@dataclass(frozen=True)
class AnnulusSpec(ShapeSpec):
    """A ring shape defined by an inner radius, outer radius, and center."""

    outer_radius: float
    inner_radius: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def type(self) -> str:
        return "annulus"

    def validate(self) -> None:
        if self.outer_radius <= 0 or self.inner_radius < 0:
            raise ValueError("annulus radii must satisfy outer>0 and inner>=0")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("annulus must satisfy outer_radius > inner_radius")

    def compute_mask(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        self.validate()
        center_x, center_y = self.center
        distance_from_center = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
        return (distance_from_center <= self.outer_radius) & (distance_from_center >= self.inner_radius)


@dataclass(frozen=True)
class TwoCirclesSpec(ShapeSpec):
    """Two disconnected circles treated as one shape."""

    radius1: float
    center1: tuple[float, float]
    radius2: float
    center2: tuple[float, float]

    @property
    def type(self) -> str:
        return "two_circles"

    def validate(self) -> None:
        if self.radius1 <= 0 or self.radius2 <= 0:
            raise ValueError("circle radii must be positive")

    def compute_mask(self, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        self.validate()
        first_center_x, first_center_y = self.center1
        second_center_x, second_center_y = self.center2
        distance_to_first_circle = np.sqrt((grid_x - first_center_x) ** 2 + (grid_y - first_center_y) ** 2)
        distance_to_second_circle = np.sqrt((grid_x - second_center_x) ** 2 + (grid_y - second_center_y) ** 2)
        return (distance_to_first_circle <= self.radius1) | (distance_to_second_circle <= self.radius2)


__all__ = [
    "AnnulusSpec",
    "CircleSpec",
    "EllipseSpec",
    "RectangleSpec",
    "TwoCirclesSpec",
]

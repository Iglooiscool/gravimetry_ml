"""Boundary measurement helpers."""

from __future__ import annotations

import numpy as np


def measurement_points_on_unit_circle(number_of_points: int) -> np.ndarray:
    """Make evenly spaced measurement points on the unit circle."""

    if number_of_points < 1:
        raise ValueError("number_of_points must be positive")
    angle_values = np.linspace(0.0, 2.0 * np.pi, number_of_points + 1)[:-1]
    return np.exp(1j * angle_values)


def compute_gradient_data(coefficients: np.ndarray, measurement_points: np.ndarray, n_max: int) -> np.ndarray:
    """Compute the complex gradient data used as the Stage 1 input."""

    gradient_values = np.zeros(measurement_points.size, dtype=np.complex128)
    for point_index, measurement_point in enumerate(measurement_points):
        gradient_at_point = 0.0j
        for coefficient_order in range(n_max + 1):
            coefficient_value = coefficients[coefficient_order]
            gradient_at_point += 2.0 * np.conj(coefficient_value) * (measurement_point ** (coefficient_order + 1))
        gradient_values[point_index] = gradient_at_point
    return gradient_values


def build_measurement_matrix(measurement_points: np.ndarray, n_max: int) -> np.ndarray:
    """Build the matrix that turns coefficients into boundary measurements."""

    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    measurement_matrix = np.zeros((measurement_points.size, n_max + 1), dtype=np.complex128)
    for row_index, measurement_point in enumerate(measurement_points):
        for coefficient_order in range(n_max + 1):
            measurement_matrix[row_index, coefficient_order] = 2.0 * np.conj(measurement_point) ** (-coefficient_order - 1)
    return measurement_matrix


def condition_number(matrix: np.ndarray) -> float:
    """Measure how sensitive the measurement matrix is."""

    return float(np.linalg.cond(matrix))


def coefficients_to_measurements(coefficients: np.ndarray, measurement_matrix: np.ndarray) -> np.ndarray:
    """Turn coefficients into the complex boundary measurements."""

    return measurement_matrix @ coefficients


__all__ = [
    "measurement_points_on_unit_circle",
    "compute_gradient_data",
    "build_measurement_matrix",
    "condition_number",
    "coefficients_to_measurements",
]

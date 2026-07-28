"""Coefficient computation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoefficientResult:
    """Complex coefficients and a few simple ways to read them."""

    coefficients: np.ndarray

    @property
    def magnitudes(self) -> np.ndarray:
        return np.abs(self.coefficients)

    @property
    def real(self) -> np.ndarray:
        return self.coefficients.real

    @property
    def imag(self) -> np.ndarray:
        return self.coefficients.imag


def compute_coefficients(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    dA: float,
    n_max: int = 5,
    scale: float | None = None,
) -> CoefficientResult:
    """Turn a binary shape mask into the project's complex coefficient vector."""

    if n_max < 0:
        raise ValueError("n_max must be non-negative")

    complex_grid = X + 1j * Y
    coefficient_values = np.zeros(n_max + 1, dtype=np.complex128)
    scale_factor = (1.0 / (4.0 * np.pi)) if scale is None else float(scale)

    for coefficient_order in range(n_max + 1):
        integrand = (complex_grid**coefficient_order) * mask
        coefficient_values[coefficient_order] = scale_factor * np.sum(integrand) * dA

    return CoefficientResult(coefficients=coefficient_values)


__all__ = ["CoefficientResult", "compute_coefficients"]

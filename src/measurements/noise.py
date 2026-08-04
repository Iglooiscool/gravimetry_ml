"""Noise helpers for synthetic measurements."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(values: np.ndarray, sigma: float, random_generator: np.random.Generator) -> np.ndarray:
    """Add independent absolute Gaussian noise to real or complex arrays."""

    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    noise_scale = float(sigma)
    if np.iscomplexobj(values):
        noise = random_generator.normal(0.0, noise_scale, size=values.shape) + 1j * random_generator.normal(0.0, noise_scale, size=values.shape)
        return values + noise
    return values + random_generator.normal(0.0, noise_scale, size=values.shape)


__all__ = ["add_gaussian_noise"]

"""Noise helpers for synthetic measurements."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(values: np.ndarray, noise_level: float, random_generator: np.random.Generator) -> np.ndarray:
    """Add relative Gaussian noise to real or complex arrays."""

    noise_scale = max(float(np.max(np.abs(values))), 1e-8) * float(noise_level)
    if np.iscomplexobj(values):
        noise = random_generator.normal(0.0, noise_scale, size=values.shape) + 1j * random_generator.normal(0.0, noise_scale, size=values.shape)
        return values + noise
    return values + random_generator.normal(0.0, noise_scale, size=values.shape)


__all__ = ["add_gaussian_noise"]

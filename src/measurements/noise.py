"""Noise helpers for synthetic measurements."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    values: np.ndarray,
    sigma: float,
    random_generator: np.random.Generator,
    mode: str = "absolute",
) -> np.ndarray:
    """Add independent Gaussian noise using absolute or legacy relative scale."""

    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    if mode not in {"absolute", "relative_max"}:
        raise ValueError("mode must be 'absolute' or 'relative_max'")
    noise_scale = float(sigma)
    if mode == "relative_max":
        noise_scale *= max(float(np.max(np.abs(values))), 1e-8)
    if np.iscomplexobj(values):
        noise = random_generator.normal(0.0, noise_scale, size=values.shape) + 1j * random_generator.normal(0.0, noise_scale, size=values.shape)
        return values + noise
    return values + random_generator.normal(0.0, noise_scale, size=values.shape)


__all__ = ["add_gaussian_noise"]

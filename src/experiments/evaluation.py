"""Reusable helpers for the official training-noise data protocol."""

from __future__ import annotations

import numpy as np

from measurements import add_gaussian_noise, coefficient_features_to_complex


def sigma_label(sigma: float) -> str:
    """Format a non-negative sigma for stable directory names."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0.0:
        return "000"
    text = f"{sigma:g}"
    if "." in text:
        text = text.replace("0.", "").replace(".", "")
    return text or "0"


def feature_matrix_with_noise(
    feature_matrix: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Return a real/imaginary feature matrix with independent absolute noise.

    The feature layout is the repository-wide ``[real, imag]`` concatenation.
    Noise is generated independently for every real and imaginary component.
    """

    if feature_matrix.ndim != 2 or feature_matrix.shape[1] % 2:
        raise ValueError("feature_matrix must be 2-D with an even feature dimension")
    complex_values = coefficient_features_to_complex(feature_matrix)
    noisy_values = add_gaussian_noise(
        complex_values,
        sigma=sigma,
        random_generator=np.random.default_rng(seed),
        mode="absolute",
    )
    result = np.concatenate([noisy_values.real, noisy_values.imag], axis=1).astype(np.float32)
    if result.shape != feature_matrix.shape:
        raise RuntimeError(f"noise conversion changed shape from {feature_matrix.shape} to {result.shape}")
    return result.astype(np.float32, copy=False)


__all__ = ["feature_matrix_with_noise", "sigma_label"]

"""Feature-layout conversion helpers."""

from __future__ import annotations

import numpy as np


def measurements_to_feature_vector(values: np.ndarray) -> np.ndarray:
    """Flatten complex measurements into a real-valued feature vector."""

    return np.concatenate([values.real, values.imag]).astype(np.float32)


def coefficients_to_feature_vector(coefficients: np.ndarray) -> np.ndarray:
    """Flatten complex coefficients into a real-valued feature vector."""

    return np.concatenate([coefficients.real, coefficients.imag]).astype(np.float32)


def coefficient_features_to_complex(features: np.ndarray) -> np.ndarray:
    """Turn flattened real/imag features back into complex coefficients."""

    half_index = features.shape[-1] // 2
    return features[..., :half_index] + 1j * features[..., half_index:]


__all__ = [
    "measurements_to_feature_vector",
    "coefficients_to_feature_vector",
    "coefficient_features_to_complex",
]

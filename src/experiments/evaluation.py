"""Reusable evaluation helpers for clean and noisy test measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from measurements import add_gaussian_noise, coefficient_features_to_complex
from models import evaluate_stage2_predictions, evaluate_stage2_predictions_by_shape


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


def evaluate_noise_sweep(
    predict_logits: Callable[[np.ndarray], np.ndarray],
    clean_test_features: np.ndarray,
    test_masks: np.ndarray,
    test_shape_types: tuple[str, ...],
    threshold: float,
    noise_levels: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005, 0.01),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one trained predictor on clean and noisy test measurements.

    This is supplementary robustness evaluation. It never changes the model,
    validation threshold, or training data.
    """

    overall_rows: list[dict[str, float]] = []
    shape_rows: list[dict[str, object]] = []
    for level in noise_levels:
        features = clean_test_features if level == 0.0 else feature_matrix_with_noise(
            clean_test_features, level, seed + int(round(level * 1_000_000))
        )
        logits = predict_logits(features)
        metrics = evaluate_stage2_predictions(test_masks, logits, threshold)
        overall_rows.append({"noise_sigma": level, **metrics})
        by_shape = evaluate_stage2_predictions_by_shape(
            test_masks, logits, threshold, test_shape_types
        )
        for shape, shape_metrics in by_shape.items():
            shape_rows.append({"noise_sigma": level, "shape": shape, **shape_metrics})
    return pd.DataFrame(overall_rows), pd.DataFrame(shape_rows)


def save_noise_results(
    overall: pd.DataFrame,
    by_shape: pd.DataFrame,
    output_dir: Path,
    stem: str = "noise_robustness",
) -> None:
    """Persist aggregate and per-test-noise sweep artifacts.

    ``output_dir`` is the training-condition directory. Each test noise level
    receives its own ``test_sigma...`` child directory in addition to the
    aggregate files at the training-condition root.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_dir / f"{stem}.csv", index=False)
    by_shape.to_csv(output_dir / f"{stem}_by_shape.csv", index=False)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(
            {"overall": overall.to_dict(orient="records"), "by_shape": by_shape.to_dict(orient="records")},
            indent=2,
        ),
        encoding="utf-8",
    )
    for level in overall["noise_sigma"].tolist():
        test_dir = output_dir / f"test_sigma{sigma_label(float(level))}"
        test_dir.mkdir(parents=True, exist_ok=True)
        overall_row = overall[overall["noise_sigma"] == level]
        shape_rows = by_shape[by_shape["noise_sigma"] == level]
        overall_row.to_csv(test_dir / "metrics.csv", index=False)
        shape_rows.to_csv(test_dir / "metrics_by_shape.csv", index=False)
        (test_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "overall": overall_row.to_dict(orient="records"),
                    "by_shape": shape_rows.to_dict(orient="records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


__all__ = ["evaluate_noise_sweep", "feature_matrix_with_noise", "save_noise_results", "sigma_label"]

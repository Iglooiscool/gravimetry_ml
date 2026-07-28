"""Purpose: keep the scalar metric helpers separate from model definitions.

This file contains both regression metrics for Stage 1 and binary-mask metrics
for Stage 2 so training code can stay focused on optimization.
"""

from __future__ import annotations

import numpy as np


# Purpose:
# Compute the mean squared error between true and predicted values.
#
# Inputs:
# - y_true: ground-truth array
# - y_pred: predicted array with the same shape
#
# Returns:
# - A float containing the mean squared error
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# Purpose:
# Compute the mean absolute error between true and predicted values.
#
# Inputs:
# - y_true: ground-truth array
# - y_pred: predicted array with the same shape
#
# Returns:
# - A float containing the mean absolute error
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# Purpose:
# Compute the binary intersection-over-union used to score reconstruction masks.
#
# Inputs:
# - y_true: ground-truth binary mask values
# - y_pred: predicted binary mask values
#
# Returns:
# - A float containing the IoU score
def binary_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_bin = y_true.astype(bool)
    y_pred_bin = y_pred.astype(bool)
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


# Purpose:
# Package the regression metrics reported for Stage 1 evaluation.
#
# Inputs:
# - y_true: ground-truth regression targets
# - y_pred: predicted regression targets
#
# Returns:
# - A dictionary containing MSE and MAE values
def evaluate_regression_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# Purpose:
# Turn raw logits into thresholded masks before computing reconstruction scores.
#
# Inputs:
# - y_true: ground-truth binary masks
# - logits: raw Stage 2 outputs before sigmoid
# - threshold: probability threshold for converting to binary predictions
#
# Returns:
# - A dictionary containing mean IoU and pixel accuracy
def evaluate_stage2_predictions(y_true: np.ndarray, logits: np.ndarray, threshold: float) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    binary = probabilities >= threshold
    ious = [binary_iou(true_row, pred_row) for true_row, pred_row in zip(y_true, binary, strict=False)]
    accuracy = float(np.mean((binary.astype(np.float32) == y_true).astype(np.float32)))
    return {
        "mean_iou": float(np.mean(ious)),
        "pixel_accuracy": accuracy,
    }


def evaluate_stage2_predictions_by_shape(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold: float,
    shape_types: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    """Compute Stage 2 metrics grouped by shape type."""

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    binary = probabilities >= threshold
    metrics_by_shape: dict[str, dict[str, float | int]] = {}

    for shape_type in sorted(set(shape_types)):
        indices = [index for index, current_shape_type in enumerate(shape_types) if current_shape_type == shape_type]
        if not indices:
            continue

        shape_true = y_true[indices]
        shape_binary = binary[indices]
        ious = [binary_iou(true_row, pred_row) for true_row, pred_row in zip(shape_true, shape_binary, strict=False)]
        accuracy = float(np.mean((shape_binary.astype(np.float32) == shape_true).astype(np.float32)))
        metrics_by_shape[shape_type] = {
            "sample_count": len(indices),
            "mean_iou": float(np.mean(ious)),
            "pixel_accuracy": accuracy,
        }

    return metrics_by_shape


def select_best_stage2_threshold(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold_candidates: tuple[float, ...],
) -> dict[str, object]:
    """Choose the threshold with the best validation IoU."""

    best_threshold = threshold_candidates[0]
    best_metrics = evaluate_stage2_predictions(y_true, logits, best_threshold)
    threshold_metrics: list[dict[str, float]] = []
    for threshold in threshold_candidates:
        current_metrics = evaluate_stage2_predictions(y_true, logits, threshold)
        threshold_metrics.append(
            {
                "threshold": float(threshold),
                "mean_iou": current_metrics["mean_iou"],
                "pixel_accuracy": current_metrics["pixel_accuracy"],
            }
        )
        if current_metrics["mean_iou"] > best_metrics["mean_iou"]:
            best_threshold = threshold
            best_metrics = current_metrics
    return {
        "selected_threshold": float(best_threshold),
        "validation_metrics": best_metrics,
        "candidates": threshold_metrics,
    }

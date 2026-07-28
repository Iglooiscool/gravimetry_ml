"""Pipeline evaluation helpers."""

from __future__ import annotations

from models import (
    evaluate_regression_predictions,
    evaluate_stage2_predictions,
    evaluate_stage2_predictions_by_shape,
    select_best_stage2_threshold,
)


def select_stage2_threshold(run_config, validation_masks, predicted_validation_masks) -> tuple[float, dict[str, object]]:
    """Pick the threshold and return a compact summary."""

    threshold_used = run_config.threshold
    fixed_metrics = evaluate_stage2_predictions(validation_masks, predicted_validation_masks, threshold_used)
    threshold_summary = {
        "selected_threshold": float(threshold_used),
        "selection_mode": "fixed",
        "validation_metrics": fixed_metrics,
        "candidates": [
            {
                "threshold": float(threshold_used),
                "mean_iou": fixed_metrics["mean_iou"],
                "pixel_accuracy": fixed_metrics["pixel_accuracy"],
            }
        ],
    }
    if run_config.use_validation_threshold_sweep:
        threshold_summary = select_best_stage2_threshold(
            validation_masks,
            predicted_validation_masks,
            run_config.threshold_candidates,
        )
        threshold_summary["selection_mode"] = "validation_sweep"
        threshold_used = float(threshold_summary["selected_threshold"])
    return float(threshold_used), threshold_summary


def build_run_metrics(dataset_bundle, predicted_test_coefficients, predicted_fixed_coefficients, predicted_test_masks, predicted_fixed_masks, threshold_used: float) -> tuple[dict[str, object], dict[str, object]]:
    """Compute headline metrics and by-shape Stage 2 breakdowns."""

    metrics = {
        "stage1_test": evaluate_regression_predictions(dataset_bundle.test.coefficients, predicted_test_coefficients),
        "stage1_fixed": evaluate_regression_predictions(dataset_bundle.fixed.coefficients, predicted_fixed_coefficients),
        "stage2_test": evaluate_stage2_predictions(dataset_bundle.test.masks, predicted_test_masks, threshold_used),
        "stage2_fixed": evaluate_stage2_predictions(dataset_bundle.fixed.masks, predicted_fixed_masks, threshold_used),
    }
    metrics_by_shape = {
        "stage2_test": evaluate_stage2_predictions_by_shape(
            dataset_bundle.test.masks,
            predicted_test_masks,
            threshold_used,
            dataset_bundle.test.shape_types,
        ),
        "stage2_fixed": evaluate_stage2_predictions_by_shape(
            dataset_bundle.fixed.masks,
            predicted_fixed_masks,
            threshold_used,
            dataset_bundle.fixed.shape_types,
        ),
    }
    return metrics, metrics_by_shape


__all__ = ["build_run_metrics", "select_stage2_threshold"]

"""Evaluation helpers for the Task 9 workflow."""

from __future__ import annotations

from models import evaluate_regression_predictions, evaluate_stage2_predictions, evaluate_stage2_predictions_by_shape, select_best_stage2_threshold
from pipeline.diagnostics import summarize_training_history


def build_task9_metrics(
    dataset_bundle,
    predicted_test_coefficients,
    predicted_fixed_coefficients,
    general_test_logits,
    general_fixed_logits,
    combined_test_logits,
    combined_fixed_logits,
    threshold: float,
):
    """Compute headline Task 9 metrics."""

    metrics = {
        "stage1_test": evaluate_regression_predictions(dataset_bundle.test.coefficients, predicted_test_coefficients),
        "stage1_fixed": evaluate_regression_predictions(dataset_bundle.fixed.coefficients, predicted_fixed_coefficients),
        "task9_general_test": evaluate_stage2_predictions(dataset_bundle.test.masks, general_test_logits, threshold),
        "task9_general_fixed": evaluate_stage2_predictions(dataset_bundle.fixed.masks, general_fixed_logits, threshold),
        "task9_combined_test": evaluate_stage2_predictions(dataset_bundle.test.masks, combined_test_logits, threshold),
        "task9_combined_fixed": evaluate_stage2_predictions(dataset_bundle.fixed.masks, combined_fixed_logits, threshold),
    }
    metrics_by_shape = {
        "task9_general_test": evaluate_stage2_predictions_by_shape(
            dataset_bundle.test.masks,
            general_test_logits,
            threshold,
            dataset_bundle.test.shape_types,
        ),
        "task9_general_fixed": evaluate_stage2_predictions_by_shape(
            dataset_bundle.fixed.masks,
            general_fixed_logits,
            threshold,
            dataset_bundle.fixed.shape_types,
        ),
        "task9_combined_test": evaluate_stage2_predictions_by_shape(
            dataset_bundle.test.masks,
            combined_test_logits,
            threshold,
            dataset_bundle.test.shape_types,
        ),
        "task9_combined_fixed": evaluate_stage2_predictions_by_shape(
            dataset_bundle.fixed.masks,
            combined_fixed_logits,
            threshold,
            dataset_bundle.fixed.shape_types,
        ),
    }
    return metrics, metrics_by_shape


def select_task9_threshold(run_config, validation_masks, combined_validation_logits) -> tuple[float, dict[str, object]]:
    """Select the Task 9 mask threshold from validation predictions."""

    threshold_used = run_config.threshold
    fixed_metrics = evaluate_stage2_predictions(validation_masks, combined_validation_logits, threshold_used)
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
            combined_validation_logits,
            run_config.threshold_candidates,
        )
        threshold_summary["selection_mode"] = "validation_sweep"
        threshold_used = float(threshold_summary["selected_threshold"])
    return float(threshold_used), threshold_summary


def build_task9_diagnostics(
    dataset_bundle,
    general_test_true_coeff_logits,
    general_fixed_true_coeff_logits,
    combined_test_true_coeff_logits,
    combined_fixed_true_coeff_logits,
    threshold: float,
    routing_counts: dict[str, int],
):
    """Build Task 9-specific diagnostic summaries."""

    return {
        "task9_general_with_true_coefficients": {
            "test": evaluate_stage2_predictions(dataset_bundle.test.masks, general_test_true_coeff_logits, threshold),
            "fixed": evaluate_stage2_predictions(dataset_bundle.fixed.masks, general_fixed_true_coeff_logits, threshold),
        },
        "task9_combined_with_true_coefficients": {
            "test": evaluate_stage2_predictions(dataset_bundle.test.masks, combined_test_true_coeff_logits, threshold),
            "fixed": evaluate_stage2_predictions(dataset_bundle.fixed.masks, combined_fixed_true_coeff_logits, threshold),
        },
        "routing": routing_counts,
    }


def build_task9_training_summary(stage1_history, general_history, specialist_history, run_config) -> dict[str, object]:
    """Summarize the training histories for all active Task 9 stages."""

    summary = {
        "stage1": summarize_training_history(stage1_history.history, run_config.model.stage1.training.epochs),
        "task9_general": summarize_training_history(general_history.history, run_config.model.general.training.epochs),
    }
    if specialist_history is not None:
        summary["task9_specialist"] = summarize_training_history(specialist_history.history, run_config.model.specialist.training.epochs)
    return summary


__all__ = ["build_task9_metrics", "build_task9_diagnostics", "build_task9_training_summary", "select_task9_threshold", "summarize_training_history"]

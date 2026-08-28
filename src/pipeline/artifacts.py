"""Persistence and figure-saving helpers for pipeline runs."""

from __future__ import annotations

from dataclasses import asdict

import torch

from plotting import save_measurement_points_plot, save_reconstruction_examples, save_shape_gallery, save_two_stage_summary
from shapes import create_fixed_benchmark_shapes


def save_model_weights(stage1_model, stage2_model, run_output_dir) -> None:
    """Persist trained model weights for a run."""

    torch.save(stage1_model.state_dict(), run_output_dir / "stage1_model.pt")
    torch.save(stage2_model.state_dict(), run_output_dir / "stage2_model.pt")


def save_run_figures(
    run_config,
    dataset_bundle,
    predicted_fixed_masks,
    predicted_fixed_masks_with_true_coefficients,
    run_output_dir,
) -> dict[str, str]:
    """Persist standard plots for a run and return their paths."""

    return {
        "measurement_points": str(save_measurement_points_plot(dataset_bundle.measurement_points, run_output_dir / "measurement_points.png")),
        "fixed_shapes": str(
            save_shape_gallery(
                create_fixed_benchmark_shapes(),
                run_output_dir / "fixed_shapes.png",
                grid_size=run_config.grid_size,
                title=f"Fixed benchmark shapes | training noise sigma={run_config.noise_sigma:g} | fixed inputs clean",
            )
        ),
        "fixed_general_predictions": str(
            save_reconstruction_examples(
                true_masks=dataset_bundle.fixed.masks,
                predicted_logits=predicted_fixed_masks,
                names=dataset_bundle.fixed.names,
                output_path=run_output_dir / "fixed_general_predictions.png",
                grid_size=run_config.grid_size,
                threshold=run_config.threshold,
                title=f"Two-stage general predictions | training noise sigma={run_config.noise_sigma:g}",
            )
        ),
        "fixed_general_true_coeff_predictions": str(
            save_reconstruction_examples(
                true_masks=dataset_bundle.fixed.masks,
                predicted_logits=predicted_fixed_masks_with_true_coefficients,
                names=dataset_bundle.fixed.names,
                output_path=run_output_dir / "fixed_general_true_coeff_predictions.png",
                grid_size=run_config.grid_size,
                threshold=run_config.threshold,
                title=f"Two-stage true-coefficient diagnostic | training noise sigma={run_config.noise_sigma:g}",
            )
        ),
        "fixed_reconstructions": str(
            save_reconstruction_examples(
                true_masks=dataset_bundle.fixed.masks,
                predicted_logits=predicted_fixed_masks,
                names=dataset_bundle.fixed.names,
                output_path=run_output_dir / "fixed_reconstructions.png",
                grid_size=run_config.grid_size,
                threshold=run_config.threshold,
                title=f"Two-stage fixed reconstructions | training noise sigma={run_config.noise_sigma:g}",
            )
        ),
    }


def build_run_summary(run_config, dataset_bundle, dataset_paths, figure_paths, metrics, metrics_by_shape, diagnostics, threshold_summary, stage1_history, stage2_history, training_summary) -> dict[str, object]:
    """Assemble the persisted summary payload."""

    return {
        "config": {**asdict(run_config), "output_dir": str(run_config.output_dir)},
        "condition_number": dataset_bundle.matrix_condition_number,
        "metrics": metrics,
        "metrics_by_shape": metrics_by_shape,
        "diagnostics": diagnostics,
        "threshold_summary": threshold_summary,
        "dataset_paths": {key: str(value) for key, value in dataset_paths.items()},
        "figure_paths": figure_paths,
        "stage1_history": stage1_history.history,
        "stage2_history": stage2_history.history,
        "training_summary": training_summary,
    }


def write_run_summary(summary: dict[str, object], run_output_dir) -> None:
    """Write the run summary JSON to disk."""

    save_two_stage_summary(summary, run_output_dir / "summary.json")


__all__ = ["build_run_summary", "save_model_weights", "save_run_figures", "write_run_summary"]

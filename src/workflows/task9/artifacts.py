"""Persistence helpers for Task 9 runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch

from plotting import save_measurement_points_plot, save_reconstruction_examples, save_shape_gallery, save_two_stage_summary
from shapes import create_fixed_benchmark_shapes


def save_task9_model_weights(stage1_model, general_head, specialist_head, run_output_dir) -> dict[str, str]:
    """Persist trained Task 9 model weights and return their paths."""

    model_paths = {
        "stage1": str(run_output_dir / "stage1_model.pt"),
        "task9_general": str(run_output_dir / "task9_general_model.pt"),
    }
    torch.save(stage1_model.state_dict(), model_paths["stage1"])
    torch.save(general_head.model.state_dict(), model_paths["task9_general"])
    if specialist_head is not None:
        model_paths["task9_specialist"] = str(run_output_dir / "task9_specialist_model.pt")
        torch.save(specialist_head.model.state_dict(), model_paths["task9_specialist"])
    return model_paths


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _binary_masks(logits: np.ndarray, threshold: float) -> np.ndarray:
    return (_sigmoid(logits) >= threshold).astype(np.float32)


def _binary_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_bin = y_true.astype(bool)
    y_pred_bin = y_pred.astype(bool)
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _sample_ious(true_masks: np.ndarray, logits: np.ndarray, threshold: float) -> np.ndarray:
    binary_predictions = _binary_masks(logits, threshold)
    return np.array([_binary_iou(true_row, pred_row) for true_row, pred_row in zip(true_masks, binary_predictions, strict=False)], dtype=np.float32)


def _select_example_indices(logits: np.ndarray, true_masks: np.ndarray, *, threshold: float, seed: int) -> dict[str, np.ndarray]:
    ious = _sample_ious(true_masks, logits, threshold)
    example_count = min(3, len(ious))
    rng = np.random.default_rng(seed)
    return {
        "best": np.argsort(ious)[-example_count:][::-1],
        "worst": np.argsort(ious)[:example_count],
        "random": np.sort(rng.choice(len(ious), size=example_count, replace=False)),
    }


def _save_model_comparison_grid(
    *,
    true_masks: np.ndarray,
    general_logits: np.ndarray,
    combined_logits: np.ndarray,
    names: tuple[str, ...],
    indices: np.ndarray,
    grid_size: int,
    threshold: float,
    output_path: Path,
    title_prefix: str,
) -> Path:
    general_binary = _binary_masks(general_logits, threshold)
    combined_binary = _binary_masks(combined_logits, threshold)
    general_ious = _sample_ious(true_masks, general_logits, threshold)
    combined_ious = _sample_ious(true_masks, combined_logits, threshold)

    fig = Figure(figsize=(9, 3 * len(indices)))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(indices), 3)
    if len(indices) == 1:
        axes = np.array([axes])

    for row_index, sample_index in enumerate(indices):
        axes[row_index, 0].imshow(true_masks[sample_index].reshape(grid_size, grid_size), cmap="gray", origin="lower")
        axes[row_index, 0].set_title(f"True: {names[sample_index]}")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(general_binary[sample_index].reshape(grid_size, grid_size), cmap="gray", origin="lower")
        axes[row_index, 1].set_title(f"General\nIoU={general_ious[sample_index]:.3f}")
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(combined_binary[sample_index].reshape(grid_size, grid_size), cmap="gray", origin="lower")
        axes[row_index, 2].set_title(f"Combined\nIoU={combined_ious[sample_index]:.3f}")
        axes[row_index, 2].axis("off")

    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


def save_task9_figures(
    run_config,
    dataset_bundle,
    general_test_logits,
    combined_test_logits,
    general_fixed_logits,
    combined_fixed_logits,
    run_output_dir,
    threshold_used: float | None = None,
) -> dict[str, str]:
    """Persist standard Task 9 plots and return their paths."""

    threshold = run_config.threshold if threshold_used is None else threshold_used
    test_example_indices = _select_example_indices(
        combined_test_logits,
        dataset_bundle.test.masks,
        threshold=threshold,
        seed=run_config.seed,
    )

    return {
        "measurement_points": str(save_measurement_points_plot(dataset_bundle.measurement_points, run_output_dir / "measurement_points.png")),
        "fixed_shapes": str(save_shape_gallery(create_fixed_benchmark_shapes(), run_output_dir / "fixed_shapes.png", grid_size=run_config.grid_size)),
        "fixed_reconstructions": str(
            save_reconstruction_examples(
                true_masks=dataset_bundle.fixed.masks,
                predicted_logits=combined_fixed_logits,
                names=dataset_bundle.fixed.names,
                output_path=run_output_dir / "fixed_reconstructions.png",
                grid_size=run_config.grid_size,
                threshold=threshold,
            )
        ),
        "test_best_reconstructions": str(
            save_reconstruction_examples(
                true_masks=dataset_bundle.test.masks[test_example_indices["best"]],
                predicted_logits=combined_test_logits[test_example_indices["best"]],
                names=tuple(dataset_bundle.test.names[index] for index in test_example_indices["best"]),
                output_path=run_output_dir / "test_best_reconstructions.png",
                grid_size=run_config.grid_size,
                threshold=threshold,
            )
        ),
        "test_worst_reconstructions": str(
            save_reconstruction_examples(
                true_masks=dataset_bundle.test.masks[test_example_indices["worst"]],
                predicted_logits=combined_test_logits[test_example_indices["worst"]],
                names=tuple(dataset_bundle.test.names[index] for index in test_example_indices["worst"]),
                output_path=run_output_dir / "test_worst_reconstructions.png",
                grid_size=run_config.grid_size,
                threshold=threshold,
            )
        ),
        "test_random_reconstructions": str(
            save_reconstruction_examples(
                true_masks=dataset_bundle.test.masks[test_example_indices["random"]],
                predicted_logits=combined_test_logits[test_example_indices["random"]],
                names=tuple(dataset_bundle.test.names[index] for index in test_example_indices["random"]),
                output_path=run_output_dir / "test_random_reconstructions.png",
                grid_size=run_config.grid_size,
                threshold=threshold,
            )
        ),
        "test_general_vs_combined": str(
            _save_model_comparison_grid(
                true_masks=dataset_bundle.test.masks,
                general_logits=general_test_logits,
                combined_logits=combined_test_logits,
                names=dataset_bundle.test.names,
                indices=test_example_indices["random"],
                grid_size=run_config.grid_size,
                threshold=threshold,
                output_path=run_output_dir / "test_general_vs_combined.png",
                title_prefix="Held-out test: general vs combined",
            )
        ),
        "fixed_general_vs_combined": str(
            _save_model_comparison_grid(
                true_masks=dataset_bundle.fixed.masks,
                general_logits=general_fixed_logits,
                combined_logits=combined_fixed_logits,
                names=dataset_bundle.fixed.names,
                indices=np.arange(len(dataset_bundle.fixed.names)),
                grid_size=run_config.grid_size,
                threshold=threshold,
                output_path=run_output_dir / "fixed_general_vs_combined.png",
                title_prefix="Fixed benchmark: general vs combined",
            )
        ),
    }


def build_task9_summary(
    run_config,
    general_dataset_paths,
    specialist_dataset_paths,
    model_paths,
    figure_paths,
    metrics,
    metrics_by_shape,
    diagnostics,
    training_summary,
    threshold_summary=None,
) -> dict[str, object]:
    """Assemble the persisted Task 9 summary payload."""

    return {
        "config": {**asdict(run_config), "output_dir": str(run_config.output_dir)},
        "datasets": {
            "general": general_dataset_paths,
            "specialist": specialist_dataset_paths,
        },
        "model_paths": model_paths,
        "figure_paths": figure_paths,
        "metrics": metrics,
        "metrics_by_shape": metrics_by_shape,
        "diagnostics": diagnostics,
        "threshold_summary": threshold_summary,
        "training_summary": training_summary,
    }


def write_task9_summary(summary: dict[str, object], run_output_dir) -> None:
    """Write the Task 9 summary JSON to disk."""

    save_two_stage_summary(summary, run_output_dir / "summary.json")


__all__ = [
    "save_task9_model_weights",
    "save_task9_figures",
    "build_task9_summary",
    "write_task9_summary",
]

"""Purpose: keep the output plots and summary writing in one simple file.

This file handles the measurement plots, shape galleries, reconstruction plots,
and JSON summaries shared by notebooks and reusable runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from measurements import GridSpec, create_grid


# Purpose:
# Show where the synthetic measurements are taken on the unit circle.
#
# Inputs:
# - measurement_points: complex unit-circle sensor locations
# - output_path: file path where the plot should be saved
#
# Returns:
# - The path where the plot was saved
def save_measurement_points_plot(measurement_points: np.ndarray, output_path: Path) -> Path:
    fig = Figure(figsize=(4, 4))
    FigureCanvasAgg(fig)
    axis = fig.add_subplot(1, 1, 1)
    angle_values = np.linspace(0.0, 2.0 * np.pi, 400)
    axis.plot(np.cos(angle_values), np.sin(angle_values), linestyle="--", color="lightgray")
    axis.scatter(measurement_points.real, measurement_points.imag, color="tab:blue")
    axis.set_aspect("equal")
    axis.set_title("Measurement Points")
    axis.set_xlabel("Re")
    axis.set_ylabel("Im")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


# Purpose:
# Plot the condition number as N changes across a sweep.
#
# Inputs:
# - results: list of result rows with N and condition_number keys
# - output_path: file path where the plot should be saved
#
# Returns:
# - The path where the plot was saved
def save_condition_number_plot(results: list[dict[str, float]], output_path: Path) -> Path:
    fig = Figure(figsize=(5, 3))
    FigureCanvasAgg(fig)
    axis = fig.add_subplot(1, 1, 1)
    axis.plot([row["N"] for row in results], [row["condition_number"] for row in results], marker="o")
    axis.set_xlabel("N")
    axis.set_ylabel("cond(M)")
    axis.set_title("Measurement Matrix Conditioning")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


# Purpose:
# Save a gallery image for a list of shape specifications.
#
# Inputs:
# - shape_entries: list of (display_name, shape_spec) pairs
# - output_path: file path where the plot should be saved
# - grid_size: square grid size used to draw the shapes
#
# Returns:
# - The path where the plot was saved
def save_shape_gallery(shape_entries: list[tuple[str, object]], output_path: Path, grid_size: int = 32) -> Path:
    grid_data = create_grid(GridSpec(grid_size=grid_size))
    fig = Figure(figsize=(3 * len(shape_entries), 3))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(shape_entries))
    if len(shape_entries) == 1:
        axes = [axes]
    for axis, (name, shape) in zip(axes, shape_entries, strict=False):
        axis.imshow(shape.compute_mask(grid_data.X, grid_data.Y), cmap="gray", origin="lower")
        axis.set_title(name)
        axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


# Purpose:
# Save side-by-side true and predicted masks for a reconstruction example set.
#
# Inputs:
# - true_masks: ground-truth flattened masks
# - predicted_logits: raw Stage 2 outputs before sigmoid
# - names: labels for the examples
# - output_path: file path where the plot should be saved
# - grid_size: side length of the square mask grid
# - threshold: probability threshold for binary predictions
#
# Returns:
# - The path where the plot was saved
def save_reconstruction_examples(
    true_masks: np.ndarray,
    predicted_logits: np.ndarray,
    names: tuple[str, ...],
    output_path: Path,
    grid_size: int,
    threshold: float,
) -> Path:
    probabilities = 1.0 / (1.0 + np.exp(-predicted_logits))
    predicted_masks = (probabilities >= threshold).astype(np.float32)
    row_count = true_masks.shape[0]
    fig = Figure(figsize=(6, 3 * row_count))
    FigureCanvasAgg(fig)
    axes = fig.subplots(row_count, 2)
    if row_count == 1:
        axes = np.array([axes])
    for row_index in range(row_count):
        axes[row_index, 0].imshow(true_masks[row_index].reshape(grid_size, grid_size), cmap="gray", origin="lower")
        axes[row_index, 0].set_title(f"True: {names[row_index]}")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].imshow(predicted_masks[row_index].reshape(grid_size, grid_size), cmap="gray", origin="lower")
        axes[row_index, 1].set_title("Predicted")
        axes[row_index, 1].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


# Purpose:
# Save an experiment summary as JSON.
#
# Inputs:
# - summary: dictionary of values to write
# - output_path: file path where the JSON should be saved
#
# Returns:
# - The path where the summary was saved
def save_two_stage_summary(summary: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "save_shape_gallery",
    "save_measurement_points_plot",
    "save_condition_number_plot",
    "save_reconstruction_examples",
    "save_two_stage_summary",
]

"""Task 2 dataset export workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config.runs import Task2GenerateConfig
from measurements import compute_coefficients, create_grid
from shapes import sample_random_shape


def generate_task2_dataset(task2_config: Task2GenerateConfig) -> dict[str, Path]:
    """Keep the Task 2 coefficient-generation workflow available."""

    grid_data = create_grid(task2_config.grid)
    random_generator = np.random.default_rng(task2_config.seed)
    rows: list[dict[str, object]] = []

    for shape_type in task2_config.shape_types:
        for _ in range(task2_config.samples_per_shape):
            shape = sample_random_shape(random_generator, shape_type, task2_config.sampling)
            mask = shape.compute_mask(grid_data.X, grid_data.Y)
            coefficient_result = compute_coefficients(mask, grid_data.X, grid_data.Y, grid_data.dA, n_max=task2_config.n_max)
            rows.append(
                {
                    "shape": shape.to_record(),
                    "a0_real": float(coefficient_result.real[0]),
                    "coeff_real": coefficient_result.real.tolist(),
                    "coeff_imag": coefficient_result.imag.tolist(),
                    "coeff_mag": coefficient_result.magnitudes.tolist(),
                    "mask_pixels": int(mask.sum()),
                }
            )

    task2_config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = task2_config.output_dir / "task2_coefficients_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row) + "\n")

    summary_path = task2_config.output_dir / "task2_summary.json"
    summary = {
        "total_samples": len(rows),
        "samples_per_shape": task2_config.samples_per_shape,
        "shape_types": list(task2_config.shape_types),
        "n_max": task2_config.n_max,
        "seed": task2_config.seed,
        "grid_size": task2_config.grid.grid_size,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"metadata": metadata_path, "summary": summary_path}


__all__ = ["generate_task2_dataset"]

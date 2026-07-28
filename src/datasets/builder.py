"""Two-stage dataset construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from measurements import (
    GridSpec,
    add_gaussian_noise,
    build_measurement_matrix,
    coefficients_to_feature_vector,
    coefficients_to_measurements,
    compute_coefficients,
    compute_gradient_data,
    condition_number,
    create_grid,
    measurement_points_on_unit_circle,
    measurements_to_feature_vector,
)
from shapes import ShapeSamplingConfig, create_fixed_benchmark_shapes

from .sampling import sample_two_stage_shape
from .types import TwoStageDatasetBundle, TwoStageDatasetSplit

if TYPE_CHECKING:
    from config.runs import TwoStageRunConfig


def _build_split(
    sample_count: int,
    run_config: TwoStageRunConfig,
    random_generator: np.random.Generator,
    measurement_matrix: np.ndarray,
    sampling_config: ShapeSamplingConfig,
    fixed_shapes: list[tuple[str, object]] | None = None,
    allowed_shape_types: tuple[str, ...] | None = None,
    shape_weights_override: tuple[tuple[str, float], ...] | None = None,
) -> TwoStageDatasetSplit:
    grid_data = create_grid(GridSpec(grid_size=run_config.grid_size))
    gradient_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    shape_type_rows: list[str] = []
    name_rows: list[str] = []
    allowed_types = set(allowed_shape_types) if allowed_shape_types is not None else None

    gradient_measurement_points = measurement_points_on_unit_circle(run_config.num_measure_points)

    if fixed_shapes is None:
        shape_entries: list[tuple[str | None, object]] = []
        while len(shape_entries) < sample_count:
            sampled_shape = sample_two_stage_shape(
                random_generator=random_generator,
                coefficient_order=run_config.N,
                center_scale=run_config.rho,
                sampling_config=sampling_config,
                shape_weights_override=shape_weights_override,
            )
            if allowed_types is None or sampled_shape.type in allowed_types:
                shape_entries.append((None, sampled_shape))
    else:
        shape_entries = fixed_shapes
        if allowed_types is not None:
            shape_entries = [entry for entry in shape_entries if entry[1].type in allowed_types]

    if not shape_entries:
        raise ValueError("No shapes available for the requested allowed_shape_types selection")

    for display_name, shape in shape_entries:
        mask = shape.compute_mask(grid_data.X, grid_data.Y)
        coefficient_values = compute_coefficients(mask, grid_data.X, grid_data.Y, grid_data.dA, n_max=run_config.N).coefficients
        clean_measurement_values = coefficients_to_measurements(coefficient_values, measurement_matrix)
        add_gaussian_noise(clean_measurement_values, run_config.noise_level, random_generator)
        gradient_values = compute_gradient_data(coefficient_values, gradient_measurement_points, run_config.N)
        noisy_gradient_values = add_gaussian_noise(gradient_values, run_config.noise_level, random_generator)

        gradient_rows.append(measurements_to_feature_vector(noisy_gradient_values))
        coefficient_rows.append(coefficients_to_feature_vector(coefficient_values))
        mask_rows.append(mask.astype(np.float32).reshape(-1))
        shape_type_rows.append(shape.type)
        name_rows.append(display_name or shape.type)

    return TwoStageDatasetSplit(
        gradient_data=np.stack(gradient_rows).astype(np.float32),
        coefficients=np.stack(coefficient_rows).astype(np.float32),
        masks=np.stack(mask_rows).astype(np.float32),
        shape_types=tuple(shape_type_rows),
        names=tuple(name_rows),
    )


def build_two_stage_datasets(
    run_config: TwoStageRunConfig,
    sampling_config: ShapeSamplingConfig | None = None,
    allowed_shape_types: tuple[str, ...] | None = None,
) -> TwoStageDatasetBundle:
    """Generate two-stage train/validation/test/fixed datasets."""

    random_generator = np.random.default_rng(run_config.seed)
    sampling_settings = sampling_config or ShapeSamplingConfig()
    measurement_points = measurement_points_on_unit_circle(run_config.num_measure_points)
    measurement_matrix = build_measurement_matrix(measurement_points, run_config.N)
    fixed_shapes = create_fixed_benchmark_shapes()

    return TwoStageDatasetBundle(
        train=_build_split(
            run_config.training_samples,
            run_config,
            random_generator,
            measurement_matrix,
            sampling_settings,
            allowed_shape_types=allowed_shape_types,
            shape_weights_override=run_config.training_shape_weights,
        ),
        validation=_build_split(
            run_config.validation_samples,
            run_config,
            random_generator,
            measurement_matrix,
            sampling_settings,
            allowed_shape_types=allowed_shape_types,
        ),
        test=_build_split(
            run_config.test_samples,
            run_config,
            random_generator,
            measurement_matrix,
            sampling_settings,
            allowed_shape_types=allowed_shape_types,
        ),
        fixed=_build_split(
            len(fixed_shapes),
            run_config,
            random_generator,
            measurement_matrix,
            sampling_settings,
            fixed_shapes=fixed_shapes,
            allowed_shape_types=allowed_shape_types,
        ),
        measurement_points=measurement_points,
        measurement_matrix=measurement_matrix,
        matrix_condition_number=condition_number(measurement_matrix),
    )


__all__ = ["build_two_stage_datasets"]

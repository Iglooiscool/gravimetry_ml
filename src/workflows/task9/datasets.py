"""Task 9 dataset preparation."""

from __future__ import annotations

import numpy as np

from config.model_stack import TwoStageStackConfig
from config.runs import TwoStageRunConfig
from datasets import build_two_stage_datasets
from datasets.types import TwoStageDatasetBundle, TwoStageDatasetSplit
from measurements import GridSpec, build_measurement_matrix, coefficients_to_feature_vector, compute_coefficients, compute_gradient_data, condition_number, create_grid, measurement_points_on_unit_circle, measurements_to_feature_vector
from shapes import TwoCirclesSpec


def _base_two_stage_run_config(
    run_config,
    *,
    training_samples: int,
    validation_samples: int,
    test_samples: int,
    training_shape_weights=None,
) -> TwoStageRunConfig:
    return TwoStageRunConfig(
        N=run_config.N,
        training_samples=training_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        rho=run_config.rho,
        grid_size=run_config.grid_size,
        threshold=run_config.threshold,
        noise_sigma=run_config.noise_sigma,
        seed=run_config.seed,
        training_shape_weights=training_shape_weights,
        model=TwoStageStackConfig(),
        output_dir=run_config.output_dir,
    )


def build_task9_general_dataset(run_config):
    """Build the standard mixed-shape dataset used by Task 9."""

    return build_two_stage_datasets(
        _base_two_stage_run_config(
            run_config,
            training_samples=run_config.training_samples,
            validation_samples=run_config.validation_samples,
            test_samples=run_config.test_samples,
        )
    )


def augment_task9_general_training_split(dataset_split: TwoStageDatasetSplit, general_config) -> TwoStageDatasetSplit:
    """Duplicate rectangle samples to match the Task 9 edge-focused augmentation idea."""

    if not general_config.enable_rectangle_edge_augmentation or general_config.rectangle_augmentation_copies <= 0:
        return dataset_split

    rectangle_indices = [index for index, shape_type in enumerate(dataset_split.shape_types) if shape_type == "rectangle"]
    if not rectangle_indices:
        return dataset_split

    duplicated_gradient_rows = [dataset_split.gradient_data]
    duplicated_coefficient_rows = [dataset_split.coefficients]
    duplicated_mask_rows = [dataset_split.masks]
    duplicated_shape_types = list(dataset_split.shape_types)
    duplicated_names = list(dataset_split.names)

    for _ in range(general_config.rectangle_augmentation_copies):
        duplicated_gradient_rows.append(dataset_split.gradient_data[rectangle_indices])
        duplicated_coefficient_rows.append(dataset_split.coefficients[rectangle_indices])
        duplicated_mask_rows.append(dataset_split.masks[rectangle_indices])
        duplicated_shape_types.extend(dataset_split.shape_types[index] for index in rectangle_indices)
        duplicated_names.extend(dataset_split.names[index] for index in rectangle_indices)

    return TwoStageDatasetSplit(
        gradient_data=np.concatenate(duplicated_gradient_rows, axis=0),
        coefficients=np.concatenate(duplicated_coefficient_rows, axis=0),
        masks=np.concatenate(duplicated_mask_rows, axis=0),
        shape_types=tuple(duplicated_shape_types),
        names=tuple(duplicated_names),
    )


def augment_task9_feature_rows(features: np.ndarray, shape_types: tuple[str, ...], general_config) -> np.ndarray:
    """Duplicate feature rows to match the Task 9 rectangle augmentation policy."""

    if not general_config.enable_rectangle_edge_augmentation or general_config.rectangle_augmentation_copies <= 0:
        return features

    rectangle_indices = [index for index, shape_type in enumerate(shape_types) if shape_type == "rectangle"]
    if not rectangle_indices:
        return features

    duplicated_feature_rows = [features]
    for _ in range(general_config.rectangle_augmentation_copies):
        duplicated_feature_rows.append(features[rectangle_indices])
    return np.concatenate(duplicated_feature_rows, axis=0)


def _clamp_center(center: tuple[float, float]) -> tuple[float, float]:
    return (float(max(-0.9, min(0.9, center[0]))), float(max(-0.9, min(0.9, center[1]))))


def _sample_two_circle_specialist_shape(random_generator: np.random.Generator) -> tuple[str, TwoCirclesSpec]:
    regime_draw = float(random_generator.random())

    if regime_draw < 0.25:
        regime = "separated"
        radius1 = float(0.15 + 0.3 * random_generator.random())
        radius2 = float(0.15 + 0.3 * random_generator.random())
        center1 = (-0.5 + 0.3 * float(random_generator.random()), -0.3 + 0.6 * float(random_generator.random()))
        min_distance = radius1 + radius2 + 0.2
        distance = min_distance + 0.3 * float(random_generator.random())
        angle = 2.0 * np.pi * float(random_generator.random())
        center2 = (center1[0] + distance * float(np.cos(angle)), center1[1] + distance * float(np.sin(angle)))
    elif regime_draw < 0.50:
        regime = "touching"
        radius1 = float(0.2 + 0.3 * random_generator.random())
        radius2 = float(0.2 + 0.3 * random_generator.random())
        center1 = (-0.4 + 0.3 * float(random_generator.random()), -0.3 + 0.6 * float(random_generator.random()))
        distance = radius1 + radius2 + 0.02 * float(random_generator.random())
        angle = 2.0 * np.pi * float(random_generator.random())
        center2 = (center1[0] + distance * float(np.cos(angle)), center1[1] + distance * float(np.sin(angle)))
    elif regime_draw < 0.75:
        regime = "overlapping"
        radius1 = float(0.25 + 0.3 * random_generator.random())
        radius2 = float(0.25 + 0.3 * random_generator.random())
        center1 = (-0.3 + 0.3 * float(random_generator.random()), -0.3 + 0.6 * float(random_generator.random()))
        overlap_factor = float(0.7 + 0.2 * random_generator.random())
        distance = (radius1 + radius2) * overlap_factor
        angle = 2.0 * np.pi * float(random_generator.random())
        center2 = (center1[0] + distance * float(np.cos(angle)), center1[1] + distance * float(np.sin(angle)))
    else:
        regime = "nested"
        radius1 = float(0.3 + 0.3 * random_generator.random())
        radius2 = float(0.1 + 0.2 * random_generator.random())
        center1 = (-0.2 + 0.4 * float(random_generator.random()), -0.2 + 0.4 * float(random_generator.random()))
        angle = 2.0 * np.pi * float(random_generator.random())
        offset = (radius1 - radius2) * 0.5 * float(random_generator.random())
        center2 = (center1[0] + offset * float(np.cos(angle)), center1[1] + offset * float(np.sin(angle)))

    shape = TwoCirclesSpec(
        radius1=radius1,
        center1=_clamp_center(center1),
        radius2=radius2,
        center2=_clamp_center(center2),
    )
    return regime, shape


def _build_task9_specialist_split(sample_count: int, run_config, random_generator: np.random.Generator, grid_data, measurement_points, split_name: str) -> TwoStageDatasetSplit:
    gradient_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    shape_type_rows: list[str] = []
    name_rows: list[str] = []

    for sample_index in range(sample_count):
        regime, shape = _sample_two_circle_specialist_shape(random_generator)
        mask = shape.compute_mask(grid_data.X, grid_data.Y)
        coefficient_values = compute_coefficients(mask, grid_data.X, grid_data.Y, grid_data.dA, n_max=run_config.N).coefficients
        gradient_values = compute_gradient_data(coefficient_values, measurement_points, run_config.N)
        # Task 9's specialist dataset follows the PDF and uses clean gradients.
        gradient_rows.append(measurements_to_feature_vector(gradient_values))
        coefficient_rows.append(coefficients_to_feature_vector(coefficient_values))
        mask_rows.append(mask.astype(np.float32).reshape(-1))
        shape_type_rows.append(shape.type)
        name_rows.append(f"{split_name}_{regime}_{sample_index}")

    return TwoStageDatasetSplit(
        gradient_data=np.stack(gradient_rows).astype(np.float32),
        coefficients=np.stack(coefficient_rows).astype(np.float32),
        masks=np.stack(mask_rows).astype(np.float32),
        shape_types=tuple(shape_type_rows),
        names=tuple(name_rows),
    )


def _build_task9_specialist_fixed_split(run_config) -> TwoStageDatasetSplit:
    grid_data = create_grid(GridSpec(grid_size=run_config.grid_size))
    measurement_points = measurement_points_on_unit_circle(run_config.num_measure_points)
    fixed_shapes = [
        ("fixed_separated", TwoCirclesSpec(radius1=0.22, center1=(-0.55, 0.0), radius2=0.20, center2=(0.2, 0.0))),
        ("fixed_touching", TwoCirclesSpec(radius1=0.24, center1=(-0.28, 0.0), radius2=0.22, center2=(0.20, 0.0))),
        ("fixed_overlapping", TwoCirclesSpec(radius1=0.28, center1=(-0.20, 0.0), radius2=0.25, center2=(0.18, 0.0))),
        ("fixed_nested", TwoCirclesSpec(radius1=0.42, center1=(0.05, 0.05), radius2=0.16, center2=(0.12, 0.02))),
    ]

    gradient_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    shape_type_rows: list[str] = []
    name_rows: list[str] = []

    for display_name, shape in fixed_shapes:
        mask = shape.compute_mask(grid_data.X, grid_data.Y)
        coefficient_values = compute_coefficients(mask, grid_data.X, grid_data.Y, grid_data.dA, n_max=run_config.N).coefficients
        gradient_values = compute_gradient_data(coefficient_values, measurement_points, run_config.N)
        gradient_rows.append(measurements_to_feature_vector(gradient_values))
        coefficient_rows.append(coefficients_to_feature_vector(coefficient_values))
        mask_rows.append(mask.astype(np.float32).reshape(-1))
        shape_type_rows.append(shape.type)
        name_rows.append(display_name)

    return TwoStageDatasetSplit(
        gradient_data=np.stack(gradient_rows).astype(np.float32),
        coefficients=np.stack(coefficient_rows).astype(np.float32),
        masks=np.stack(mask_rows).astype(np.float32),
        shape_types=tuple(shape_type_rows),
        names=tuple(name_rows),
    )


def build_task9_specialist_dataset(run_config):
    """Build the dedicated Task 9 two-circle specialist dataset using explicit regimes."""

    random_generator = np.random.default_rng(run_config.seed + 1000)
    grid_data = create_grid(GridSpec(grid_size=run_config.grid_size))
    measurement_points = measurement_points_on_unit_circle(run_config.num_measure_points)
    measurement_matrix = build_measurement_matrix(measurement_points, run_config.N)

    train_split = _build_task9_specialist_split(
        run_config.effective_specialist_training_samples,
        run_config,
        random_generator,
        grid_data,
        measurement_points,
        split_name="train",
    )
    validation_split = _build_task9_specialist_split(
        run_config.effective_specialist_validation_samples,
        run_config,
        random_generator,
        grid_data,
        measurement_points,
        split_name="validation",
    )
    test_split = _build_task9_specialist_split(
        max(1, run_config.effective_specialist_validation_samples),
        run_config,
        random_generator,
        grid_data,
        measurement_points,
        split_name="test",
    )
    fixed_split = _build_task9_specialist_fixed_split(run_config)

    return TwoStageDatasetBundle(
        train=train_split,
        validation=validation_split,
        test=test_split,
        fixed=fixed_split,
        measurement_points=measurement_points,
        measurement_matrix=measurement_matrix,
        matrix_condition_number=condition_number(measurement_matrix),
    )


__all__ = [
    "build_task9_general_dataset",
    "augment_task9_general_training_split",
    "augment_task9_feature_rows",
    "build_task9_specialist_dataset",
]

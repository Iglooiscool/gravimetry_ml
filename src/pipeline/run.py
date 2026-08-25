"""Orchestration for the two-model reference workflow."""

from __future__ import annotations

import torch
import numpy as np

from datasets import build_two_stage_datasets, save_two_stage_dataset
from models import (
    Stage1Regressor,
    Stage2ConvDecoder,
    Stage2CoordConvDecoder,
    Stage2MaskPredictor,
    fit_stage1_model,
    fit_stage2_model,
    predict_stage1_coefficients,
    predict_stage2_logits,
    set_torch_seed,
)
from plotting import save_condition_number_plot, save_two_stage_summary

from .artifacts import build_run_summary, save_model_weights, save_run_figures, write_run_summary
from .diagnostics import build_annulus_center_diagnostics, build_stage2_diagnostics, summarize_training_history
from .evaluation import build_run_metrics, select_stage2_threshold


def _build_stage2_model(run_config):
    stage2_config = run_config.model.stage2
    if stage2_config.model_type == "mlp":
        return Stage2MaskPredictor(
            input_dim=run_config.stage2_input_size,
            output_dim=run_config.mask_pixels,
            hidden_dims=stage2_config.hidden_layer_sizes,
            dropout_rates=stage2_config.dropout_rates,
        )
    if stage2_config.model_type == "conv_decoder":
        return Stage2ConvDecoder(
            input_dim=run_config.stage2_input_size,
            output_dim=run_config.mask_pixels,
            hidden_dims=stage2_config.hidden_layer_sizes,
            dropout_rates=stage2_config.dropout_rates,
            latent_grid_size=stage2_config.latent_grid_size,
            latent_channels=stage2_config.latent_channels,
            decoder_channels=stage2_config.decoder_channels,
        )
    if stage2_config.model_type == "coord_conv_decoder":
        return Stage2CoordConvDecoder(
            input_dim=run_config.stage2_input_size,
            output_dim=run_config.mask_pixels,
            hidden_dims=stage2_config.hidden_layer_sizes,
            dropout_rates=stage2_config.dropout_rates,
            latent_grid_size=stage2_config.latent_grid_size,
            latent_channels=stage2_config.latent_channels,
            decoder_channels=stage2_config.decoder_channels,
        )
    raise ValueError("stage2.model_type must be 'mlp', 'conv_decoder', or 'coord_conv_decoder'")


def _build_stage2_features(run_config, coefficient_features: np.ndarray, gradient_features: np.ndarray) -> np.ndarray:
    """Build Stage 2 inputs, optionally retaining the raw gradient skip path."""

    if run_config.stage2_include_gradient_features:
        return np.concatenate((coefficient_features, gradient_features), axis=1).astype(np.float32)
    return coefficient_features


def _augment_rectangle_training_rows(train_features, train_masks, train_shape_types, copies: int):
    """Duplicate rectangle rows to mirror the stronger general-model recipe."""

    if copies <= 0:
        return train_features, train_masks, train_shape_types, 0

    rectangle_indices = [index for index, shape_type in enumerate(train_shape_types) if shape_type == "rectangle"]
    if not rectangle_indices:
        return train_features, train_masks, train_shape_types, 0

    feature_rows = [train_features]
    mask_rows = [train_masks]
    augmented_shape_types = list(train_shape_types)
    for _ in range(copies):
        feature_rows.append(train_features[rectangle_indices])
        mask_rows.append(train_masks[rectangle_indices])
        augmented_shape_types.extend(train_shape_types[index] for index in rectangle_indices)

    augmented_features = np.concatenate(feature_rows, axis=0)
    augmented_masks = np.concatenate(mask_rows, axis=0)
    return augmented_features, augmented_masks, tuple(augmented_shape_types), int(augmented_features.shape[0] - train_features.shape[0])


def run_two_stage_once(run_config, device: torch.device | None = None) -> dict[str, object]:
    """Run one full two-stage experiment, save outputs, and return a summary."""

    run_output_dir = run_config.run_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_torch_seed(run_config.seed)

    dataset_bundle = build_two_stage_datasets(run_config)
    dataset_paths = save_two_stage_dataset(dataset_bundle, run_output_dir / "datasets")

    stage1_model = Stage1Regressor(
        input_dim=run_config.gradient_feature_size,
        output_dim=run_config.coefficient_size,
        hidden_dims=run_config.model.stage1.hidden_layer_sizes,
        dropout_rates=run_config.model.stage1.dropout_rates,
    )
    stage1_training = run_config.model.stage1.training
    stage1_history = fit_stage1_model(
        model=stage1_model,
        train_features=dataset_bundle.train.gradient_data,
        train_targets=dataset_bundle.train.coefficients,
        val_features=dataset_bundle.validation.gradient_data,
        val_targets=dataset_bundle.validation.coefficients,
        epochs=stage1_training.epochs,
        batch_size=stage1_training.batch_size,
        learning_rate=stage1_training.learning_rate,
        device=device,
        validation_frequency=stage1_training.validation_frequency,
        verbose=stage1_training.verbose,
        early_stopping_patience=stage1_training.early_stopping_patience,
        lr_drop_factor=stage1_training.lr_drop_factor,
        lr_drop_period=stage1_training.lr_drop_period,
        weight_decay=stage1_training.weight_decay,
        gradient_clip_norm=stage1_training.gradient_clip_norm,
    )

    predicted_train_coefficients = predict_stage1_coefficients(stage1_model, dataset_bundle.train.gradient_data, device=device, training_result=stage1_history)
    predicted_validation_coefficients = predict_stage1_coefficients(stage1_model, dataset_bundle.validation.gradient_data, device=device, training_result=stage1_history)
    predicted_test_coefficients = predict_stage1_coefficients(stage1_model, dataset_bundle.test.gradient_data, device=device, training_result=stage1_history)
    predicted_fixed_coefficients = predict_stage1_coefficients(stage1_model, dataset_bundle.fixed.gradient_data, device=device, training_result=stage1_history)

    stage2_model = _build_stage2_model(run_config)
    stage2_training = run_config.model.stage2.training
    rectangle_augmentation_copies = 2 if run_config.model.stage2.use_rectangle_edge_weighting else 0
    base_stage2_train_features = _build_stage2_features(
        run_config,
        predicted_train_coefficients,
        dataset_bundle.train.gradient_data,
    )
    augmented_train_features, augmented_train_masks, augmented_train_shape_types, rectangle_augmentation_added = _augment_rectangle_training_rows(
        base_stage2_train_features,
        dataset_bundle.train.masks,
        dataset_bundle.train.shape_types,
        rectangle_augmentation_copies,
    )
    coefficient_augmentation_copies = run_config.stage2_predicted_coefficient_augmentation_copies
    if coefficient_augmentation_copies > 0:
        coefficient_errors = predicted_train_coefficients - dataset_bundle.train.coefficients
        coefficient_error_std = np.maximum(np.std(coefficient_errors, axis=0), 1e-8)
        coefficient_rng = np.random.default_rng(run_config.seed + 2000)
        augmented_feature_rows = [augmented_train_features]
        augmented_mask_rows = [augmented_train_masks]
        augmented_shape_rows = [augmented_train_shape_types]
        for _ in range(coefficient_augmentation_copies):
            perturbations = coefficient_rng.normal(
                0.0,
                coefficient_error_std * run_config.stage2_predicted_coefficient_noise_scale,
                size=predicted_train_coefficients.shape,
            ).astype(np.float32)
            augmented_coefficients = predicted_train_coefficients + perturbations
            augmented_feature_rows.append(
                _build_stage2_features(run_config, augmented_coefficients, dataset_bundle.train.gradient_data)
            )
            augmented_mask_rows.append(dataset_bundle.train.masks)
            augmented_shape_rows.append(dataset_bundle.train.shape_types)
        augmented_train_features = np.concatenate(augmented_feature_rows, axis=0)
        augmented_train_masks = np.concatenate(augmented_mask_rows, axis=0)
        augmented_train_shape_types = tuple(shape_type for rows in augmented_shape_rows for shape_type in rows)
    stage2_validation_features = _build_stage2_features(
        run_config,
        predicted_validation_coefficients,
        dataset_bundle.validation.gradient_data,
    )
    stage2_history = fit_stage2_model(
        model=stage2_model,
        train_features=augmented_train_features,
        train_targets=augmented_train_masks,
        val_features=stage2_validation_features,
        val_targets=dataset_bundle.validation.masks,
        epochs=stage2_training.epochs,
        batch_size=stage2_training.batch_size,
        learning_rate=stage2_training.learning_rate,
        device=device,
        validation_frequency=stage2_training.validation_frequency,
        verbose=stage2_training.verbose,
        early_stopping_patience=stage2_training.early_stopping_patience,
        train_shape_types=augmented_train_shape_types,
        grid_size=run_config.grid_size,
        use_rectangle_edge_weighting=run_config.model.stage2.use_rectangle_edge_weighting,
        rectangle_edge_weight=run_config.model.stage2.rectangle_edge_weight,
        rectangle_edge_width=run_config.model.stage2.rectangle_edge_width,
        edge_weight_mode=run_config.model.stage2.edge_weight_mode,
        annulus_edge_weight=run_config.model.stage2.annulus_edge_weight,
        annulus_edge_width=run_config.model.stage2.annulus_edge_width,
        min_epochs=stage2_training.min_epochs,
        min_improvement=stage2_training.min_improvement,
        lr_drop_factor=stage2_training.lr_drop_factor,
        lr_drop_period=stage2_training.lr_drop_period,
        weight_decay=stage2_training.weight_decay,
        gradient_clip_norm=stage2_training.gradient_clip_norm,
        loss_type=stage2_training.loss_type,
        dice_loss_weight=stage2_training.dice_loss_weight,
        dice_smooth=stage2_training.dice_smooth,
        iou_loss_weight=stage2_training.iou_loss_weight,
        use_foreground_pos_weight=run_config.model.stage2.use_foreground_pos_weight,
    )

    stage2_test_features = _build_stage2_features(
        run_config,
        predicted_test_coefficients,
        dataset_bundle.test.gradient_data,
    )
    stage2_fixed_features = _build_stage2_features(
        run_config,
        predicted_fixed_coefficients,
        dataset_bundle.fixed.gradient_data,
    )
    stage2_test_true_coefficient_features = _build_stage2_features(
        run_config,
        dataset_bundle.test.coefficients,
        dataset_bundle.test.gradient_data,
    )
    stage2_fixed_true_coefficient_features = _build_stage2_features(
        run_config,
        dataset_bundle.fixed.coefficients,
        dataset_bundle.fixed.gradient_data,
    )
    predicted_test_masks = predict_stage2_logits(stage2_model, stage2_test_features, device=device, training_result=stage2_history)
    predicted_validation_masks = predict_stage2_logits(stage2_model, stage2_validation_features, device=device, training_result=stage2_history)
    predicted_fixed_masks = predict_stage2_logits(stage2_model, stage2_fixed_features, device=device, training_result=stage2_history)
    predicted_test_masks_with_true_coefficients = predict_stage2_logits(stage2_model, stage2_test_true_coefficient_features, device=device, training_result=stage2_history)
    predicted_fixed_masks_with_true_coefficients = predict_stage2_logits(stage2_model, stage2_fixed_true_coefficient_features, device=device, training_result=stage2_history)

    threshold_used, threshold_summary = select_stage2_threshold(
        run_config,
        dataset_bundle.validation.masks,
        predicted_validation_masks,
    )
    metrics, metrics_by_shape = build_run_metrics(
        dataset_bundle,
        predicted_test_coefficients,
        predicted_fixed_coefficients,
        predicted_test_masks,
        predicted_fixed_masks,
        threshold_used,
    )
    diagnostics = build_stage2_diagnostics(
        dataset_bundle.test.masks,
        dataset_bundle.fixed.masks,
        predicted_test_masks_with_true_coefficients,
        predicted_fixed_masks_with_true_coefficients,
        threshold_used,
    )
    diagnostics["annulus_center"] = build_annulus_center_diagnostics(
        predicted_fixed_masks,
        predicted_fixed_masks_with_true_coefficients,
        dataset_bundle.fixed.names,
        run_config.grid_size,
    )
    training_summary = {
        "stage1": summarize_training_history(stage1_history.history, stage1_training.epochs),
        "stage2": summarize_training_history(stage2_history.history, stage2_training.epochs),
        "stage2_train_input_source": "stage1_predicted_coefficients_plus_raw_gradients" if run_config.stage2_include_gradient_features else "stage1_predicted_coefficients",
        "stage2_validation_input_source": "stage1_predicted_coefficients_plus_raw_gradients" if run_config.stage2_include_gradient_features else "stage1_predicted_coefficients",
        "stage2_true_coefficient_input_used_for_training": False,
    }

    save_model_weights(stage1_model, stage2_model, run_output_dir)
    figure_paths = save_run_figures(
        run_config,
        dataset_bundle,
        predicted_fixed_masks,
        predicted_fixed_masks_with_true_coefficients,
        run_output_dir,
    )
    summary = build_run_summary(
        run_config,
        dataset_bundle,
        dataset_paths,
        figure_paths,
        metrics,
        metrics_by_shape,
        diagnostics,
        threshold_summary,
        stage1_history,
        stage2_history,
        {
            **training_summary,
            "rectangle_augmentation_added": rectangle_augmentation_added,
            "predicted_coefficient_augmentation_copies": coefficient_augmentation_copies,
        },
    )
    summary.update(
        {
            "stage1_input_mean": stage1_history.input_mean.tolist() if stage1_history.input_mean is not None else None,
            "stage1_input_std": stage1_history.input_std.tolist() if stage1_history.input_std is not None else None,
            "stage1_target_mean": stage1_history.target_mean.tolist() if stage1_history.target_mean is not None else None,
            "stage1_target_std": stage1_history.target_std.tolist() if stage1_history.target_std is not None else None,
            "stage2_input_mean": stage2_history.input_mean.tolist() if stage2_history.input_mean is not None else None,
            "stage2_input_std": stage2_history.input_std.tolist() if stage2_history.input_std is not None else None,
        }
    )
    write_run_summary(summary, run_output_dir)
    return summary


def run_two_stage_sweep(sweep_config, device: torch.device | None = None) -> dict[str, object]:
    """Run the two-stage experiment across several N values."""

    summary_rows: list[dict[str, object]] = []
    for run_config in sweep_config.iter_runs():
        run_summary = run_two_stage_once(run_config, device=device)
        summary_rows.append(
            {
                "N": run_config.N,
                "condition_number": run_summary["condition_number"],
                "stage2_fixed_iou": run_summary["metrics"]["stage2_fixed"]["mean_iou"],
                "stage2_test_iou": run_summary["metrics"]["stage2_test"]["mean_iou"],
            }
        )

    save_condition_number_plot(summary_rows, sweep_config.output_dir / "condition_numbers.png")
    summary = {"runs": summary_rows}
    save_two_stage_summary(summary, sweep_config.output_dir / "sweep_summary.json")
    return summary


__all__ = ["run_two_stage_once", "run_two_stage_sweep"]

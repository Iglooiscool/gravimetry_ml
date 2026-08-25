"""Main orchestration for the optional Task 9 workflow."""

from __future__ import annotations

import torch
import numpy as np

from datasets import save_two_stage_dataset
from models import Stage1Regressor, predict_stage1_coefficients, set_torch_seed
from models.task9 import (
    Task9CoefficientRouter,
    Task9CombinedModel,
    Task9GeneralMaskMLP,
    Task9TrainedHead,
    Task9TwoCircleSpecialistMLP,
    fit_task9_router,
    predict_task9_combined_logits,
    train_task9_head,
)
from plotting import save_two_stage_summary

from .artifacts import build_task9_summary, save_task9_figures, save_task9_model_weights, write_task9_summary
from .datasets import augment_task9_feature_rows, augment_task9_general_training_split, build_task9_general_dataset, build_task9_specialist_dataset
from .evaluation import build_task9_diagnostics, build_task9_metrics, build_task9_training_summary, select_task9_threshold


def _train_stage1_model(run_config, dataset_bundle, device):
    stage1_model = Stage1Regressor(
        input_dim=run_config.gradient_feature_size,
        output_dim=run_config.coefficient_size,
        hidden_dims=run_config.model.stage1.hidden_layer_sizes,
        dropout_rates=run_config.model.stage1.dropout_rates,
    )
    stage1_training = run_config.model.stage1.training
    from models import fit_stage1_model

    training_result = fit_stage1_model(
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
    )
    return stage1_model, training_result


def run_task9_once(
    run_config,
    device: torch.device | None = None,
    return_predictor: bool = False,
) -> dict[str, object] | tuple[dict[str, object], object]:
    """Run one full Task 9 experiment and save its outputs."""

    run_output_dir = run_config.run_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_torch_seed(run_config.seed)

    general_dataset = build_task9_general_dataset(run_config)
    general_dataset_paths = {key: str(value) for key, value in save_two_stage_dataset(general_dataset, run_output_dir / "datasets" / "general").items()}

    stage1_model, stage1_result = _train_stage1_model(run_config, general_dataset, device)
    predicted_train_coefficients = predict_stage1_coefficients(stage1_model, general_dataset.train.gradient_data, device=device, training_result=stage1_result)
    predicted_validation_coefficients = predict_stage1_coefficients(stage1_model, general_dataset.validation.gradient_data, device=device, training_result=stage1_result)
    predicted_test_coefficients = predict_stage1_coefficients(stage1_model, general_dataset.test.gradient_data, device=device, training_result=stage1_result)
    predicted_fixed_coefficients = predict_stage1_coefficients(stage1_model, general_dataset.fixed.gradient_data, device=device, training_result=stage1_result)

    augmented_general_train = augment_task9_general_training_split(general_dataset.train, run_config.model.general)
    augmented_predicted_train_coefficients = augment_task9_feature_rows(
        predicted_train_coefficients,
        general_dataset.train.shape_types,
        run_config.model.general,
    )
    rectangle_augmentation_added = int(augmented_general_train.gradient_data.shape[0] - general_dataset.train.gradient_data.shape[0])

    general_head = train_task9_head(
        name="task9_general",
        model=Task9GeneralMaskMLP(
            input_dim=run_config.coefficient_size,
            output_dim=run_config.mask_pixels,
            hidden_dims=run_config.model.general.hidden_layer_sizes,
            dropout_rates=run_config.model.general.dropout_rates,
        ),
        train_features=augmented_predicted_train_coefficients,
        train_targets=augmented_general_train.masks,
        val_features=predicted_validation_coefficients,
        val_targets=general_dataset.validation.masks,
        device=device,
        training_config=run_config.model.general.training,
        train_shape_types=augmented_general_train.shape_types,
        grid_size=run_config.grid_size,
        use_rectangle_edge_weighting=run_config.model.general.use_rectangle_edge_weighting,
        rectangle_edge_weight=run_config.model.general.rectangle_edge_weight,
        rectangle_edge_width=run_config.model.general.rectangle_edge_width,
        edge_weight_mode=run_config.model.general.edge_weight_mode,
    )

    specialist_dataset_paths: dict[str, str] = {}
    specialist_head = None
    if run_config.specialist_enabled:
        specialist_dataset = build_task9_specialist_dataset(run_config)
        specialist_dataset_paths = {key: str(value) for key, value in save_two_stage_dataset(specialist_dataset, run_output_dir / "datasets" / "specialist").items()}
        predicted_specialist_train_coefficients = predict_stage1_coefficients(stage1_model, specialist_dataset.train.gradient_data, device=device, training_result=stage1_result)
        predicted_specialist_validation_coefficients = predict_stage1_coefficients(stage1_model, specialist_dataset.validation.gradient_data, device=device, training_result=stage1_result)
        specialist_head = train_task9_head(
            name="task9_specialist",
            model=Task9TwoCircleSpecialistMLP(
                input_dim=run_config.coefficient_size,
                output_dim=run_config.mask_pixels,
                hidden_dims=run_config.model.specialist.hidden_layer_sizes,
                dropout_rates=run_config.model.specialist.dropout_rates,
            ),
            train_features=predicted_specialist_train_coefficients,
            train_targets=specialist_dataset.train.masks,
            val_features=predicted_specialist_validation_coefficients,
            val_targets=specialist_dataset.validation.masks,
            device=device,
            training_config=run_config.model.specialist.training,
        )

    router_model = Task9CoefficientRouter(run_config.coefficient_size)
    router_labels = np.asarray(
        [shape_type == run_config.model.specialist_shape_type for shape_type in general_dataset.train.shape_types],
        dtype=np.float32,
    )
    router_validation_labels = np.asarray(
        [shape_type == run_config.model.specialist_shape_type for shape_type in general_dataset.validation.shape_types],
        dtype=np.float32,
    )
    router_result = fit_task9_router(
        router_model,
        predicted_train_coefficients,
        router_labels,
        predicted_validation_coefficients,
        router_validation_labels,
        epochs=80,
        batch_size=64,
        learning_rate=0.001,
        device=device,
    )
    router_head = Task9TrainedHead("task9_router", router_model, router_result)

    combined_model = Task9CombinedModel(
        general=general_head,
        specialist=specialist_head,
        specialist_shape_type=run_config.model.specialist_shape_type,
        routing_mode=run_config.model.routing_mode,
        router=router_head,
    )

    from models.stage2.train import predict_stage2_logits

    general_test_logits = predict_stage2_logits(general_head.model, predicted_test_coefficients, device=device, training_result=general_head.training_result)
    general_fixed_logits = predict_stage2_logits(general_head.model, predicted_fixed_coefficients, device=device, training_result=general_head.training_result)
    general_test_true_coeff_logits = predict_stage2_logits(general_head.model, general_dataset.test.coefficients, device=device, training_result=general_head.training_result)
    general_fixed_true_coeff_logits = predict_stage2_logits(general_head.model, general_dataset.fixed.coefficients, device=device, training_result=general_head.training_result)

    combined_test_logits, test_specialist_count = predict_task9_combined_logits(combined_model, predicted_test_coefficients, general_dataset.test.shape_types, device)
    combined_validation_logits, validation_specialist_count = predict_task9_combined_logits(combined_model, predicted_validation_coefficients, general_dataset.validation.shape_types, device)
    combined_fixed_logits, fixed_specialist_count = predict_task9_combined_logits(combined_model, predicted_fixed_coefficients, general_dataset.fixed.shape_types, device)
    combined_test_true_coeff_logits, true_coeff_test_specialist_count = predict_task9_combined_logits(combined_model, general_dataset.test.coefficients, general_dataset.test.shape_types, device)
    combined_fixed_true_coeff_logits, true_coeff_fixed_specialist_count = predict_task9_combined_logits(combined_model, general_dataset.fixed.coefficients, general_dataset.fixed.shape_types, device)

    threshold_used, threshold_summary = select_task9_threshold(
        run_config,
        general_dataset.validation.masks,
        combined_validation_logits,
    )

    metrics, metrics_by_shape = build_task9_metrics(
        general_dataset,
        predicted_test_coefficients,
        predicted_fixed_coefficients,
        general_test_logits,
        general_fixed_logits,
        combined_test_logits,
        combined_fixed_logits,
        threshold_used,
    )
    diagnostics = build_task9_diagnostics(
        general_dataset,
        general_test_true_coeff_logits,
        general_fixed_true_coeff_logits,
        combined_test_true_coeff_logits,
        combined_fixed_true_coeff_logits,
        threshold_used,
        routing_counts={
            "specialist_enabled": int(run_config.specialist_enabled),
            "rectangle_augmentation_added": rectangle_augmentation_added,
            "validation_specialist_samples": validation_specialist_count,
            "test_specialist_samples": test_specialist_count,
            "fixed_specialist_samples": fixed_specialist_count,
            "test_true_coeff_specialist_samples": true_coeff_test_specialist_count,
            "fixed_true_coeff_specialist_samples": true_coeff_fixed_specialist_count,
        },
    )
    training_summary = build_task9_training_summary(stage1_result, general_head.training_result, specialist_head.training_result if specialist_head is not None else None, run_config)

    model_paths = save_task9_model_weights(stage1_model, general_head, specialist_head, run_output_dir, router_head)
    figure_paths = save_task9_figures(
        run_config,
        general_dataset,
        general_test_logits,
        combined_test_logits,
        general_fixed_logits,
        combined_fixed_logits,
        run_output_dir,
        threshold_used=threshold_used,
    )
    summary = build_task9_summary(
        run_config,
        general_dataset_paths,
        specialist_dataset_paths,
        model_paths,
        figure_paths,
        metrics,
        metrics_by_shape,
        diagnostics,
        training_summary,
        threshold_summary,
    )
    write_task9_summary(summary, run_output_dir)
    if return_predictor:
        def predict_test_logits(features):
            predicted_coefficients = predict_stage1_coefficients(
                stage1_model,
                features,
                device=device,
                training_result=stage1_result,
            )
            logits, _ = predict_task9_combined_logits(
                combined_model,
                predicted_coefficients,
                general_dataset.test.shape_types,
                device,
            )
            return logits

        return summary, predict_test_logits
    return summary


def run_task9_once_with_predictor(run_config, device: torch.device | None = None):
    """Run Task 9 and return a live oracle-routed test predictor for diagnostics."""

    return run_task9_once(run_config, device=device, return_predictor=True)


def run_task9_sweep(sweep_config, device: torch.device | None = None) -> dict[str, object]:
    """Run Task 9 across several N values and collect a compact sweep summary."""

    summary_rows: list[dict[str, object]] = []
    for run_config in sweep_config.iter_runs():
        run_summary = run_task9_once(run_config, device=device)
        summary_rows.append(
            {
                "N": run_config.N,
                "task9_combined_test_iou": run_summary["metrics"]["task9_combined_test"]["mean_iou"],
                "task9_combined_fixed_iou": run_summary["metrics"]["task9_combined_fixed"]["mean_iou"],
                "specialist_enabled": bool(run_summary["diagnostics"]["routing"]["specialist_enabled"]),
            }
        )

    summary = {"runs": summary_rows}
    save_two_stage_summary(summary, sweep_config.output_dir / "sweep_summary.json")
    return summary


__all__ = ["run_task9_once", "run_task9_once_with_predictor", "run_task9_sweep"]

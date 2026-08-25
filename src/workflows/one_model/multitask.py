"""Joint coefficient and mask training for the one-stage workflow."""

from __future__ import annotations

import json

import torch

from datasets import build_two_stage_datasets, save_two_stage_dataset
from models import evaluate_stage2_predictions, evaluate_stage2_predictions_by_shape, select_best_stage2_threshold, set_torch_seed
from models.one_model import MultiTaskGradientModel, fit_multitask_one_model, predict_multitask_one_model
from plotting import save_measurement_points_plot, save_reconstruction_examples, save_shape_gallery
from shapes import create_fixed_benchmark_shapes


def run_multitask_one_model(run_config, coefficient_loss_weight: float = 0.1, device: torch.device | None = None) -> dict[str, object]:
    """Train and evaluate a shared coefficient/mask one-stage model."""

    output_dir = run_config.run_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_torch_seed(run_config.seed)
    dataset = build_two_stage_datasets(run_config)
    dataset_paths = save_two_stage_dataset(dataset, output_dir / "datasets")
    model = MultiTaskGradientModel(
        input_dim=run_config.gradient_feature_size,
        coefficient_dim=run_config.coefficient_size,
        output_dim=run_config.mask_pixels,
        hidden_dims=run_config.model.hidden_layer_sizes,
        dropout_rates=run_config.model.dropout_rates,
        latent_grid_size=run_config.model.latent_grid_size,
        latent_channels=run_config.model.latent_channels,
        decoder_channels=run_config.model.decoder_channels,
    )
    training = run_config.model.training
    training_result = fit_multitask_one_model(
        model=model,
        gradient_features=dataset.train.gradient_data,
        target_masks=dataset.train.masks,
        target_coefficients=dataset.train.coefficients,
        validation_gradient_features=dataset.validation.gradient_data,
        validation_masks=dataset.validation.masks,
        validation_coefficients=dataset.validation.coefficients,
        epochs=training.epochs,
        batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        device=device,
        coefficient_loss_weight=coefficient_loss_weight,
        validation_frequency=training.validation_frequency,
        verbose=training.verbose,
        early_stopping_patience=training.early_stopping_patience,
        train_shape_types=dataset.train.shape_types,
        grid_size=run_config.grid_size,
        use_rectangle_edge_weighting=run_config.model.use_rectangle_edge_weighting,
        rectangle_edge_weight=run_config.model.rectangle_edge_weight,
        rectangle_edge_width=run_config.model.rectangle_edge_width,
        edge_weight_mode=run_config.model.edge_weight_mode,
        annulus_edge_weight=run_config.model.annulus_edge_weight,
        annulus_edge_width=run_config.model.annulus_edge_width,
        min_epochs=training.min_epochs,
        min_improvement=training.min_improvement,
        lr_drop_factor=training.lr_drop_factor,
        lr_drop_period=training.lr_drop_period,
        weight_decay=training.weight_decay,
        gradient_clip_norm=training.gradient_clip_norm,
        loss_type=training.loss_type,
        dice_loss_weight=training.dice_loss_weight,
        dice_smooth=training.dice_smooth,
        iou_loss_weight=training.iou_loss_weight,
    )
    validation_logits, validation_coefficients = predict_multitask_one_model(model, dataset.validation.gradient_data, device, training_result)
    test_logits, test_coefficients = predict_multitask_one_model(model, dataset.test.gradient_data, device, training_result)
    fixed_logits, fixed_coefficients = predict_multitask_one_model(model, dataset.fixed.gradient_data, device, training_result)
    threshold_summary = select_best_stage2_threshold(dataset.validation.masks, validation_logits, run_config.threshold_candidates)
    threshold = float(threshold_summary["selected_threshold"])
    summary = {
        "model_count": 1,
        "model_type": "multitask_gradient_to_mask_and_coefficients",
        "coefficient_loss_weight": coefficient_loss_weight,
        "config": {"N": run_config.N, "training_samples": run_config.training_samples, "validation_samples": run_config.validation_samples, "test_samples": run_config.test_samples, "threshold": threshold},
        "metrics": {
            "test": evaluate_stage2_predictions(dataset.test.masks, test_logits, threshold),
            "fixed": evaluate_stage2_predictions(dataset.fixed.masks, fixed_logits, threshold),
        },
        "metrics_by_shape": {
            "test": evaluate_stage2_predictions_by_shape(dataset.test.masks, test_logits, threshold, dataset.test.shape_types),
            "fixed": evaluate_stage2_predictions_by_shape(dataset.fixed.masks, fixed_logits, threshold, dataset.fixed.shape_types),
        },
        "coefficient_mae": {
            "test": float(abs(test_coefficients - dataset.test.coefficients).mean()),
            "fixed": float(abs(fixed_coefficients - dataset.fixed.coefficients).mean()),
        },
        "threshold_summary": threshold_summary,
        "dataset_paths": {key: str(value) for key, value in dataset_paths.items()},
        "training_history": training_result.history,
    }
    save_measurement_points_plot(dataset.measurement_points, output_dir / "measurement_points.png")
    save_shape_gallery(create_fixed_benchmark_shapes(), output_dir / "fixed_shapes.png", run_config.grid_size)
    save_reconstruction_examples(dataset.fixed.masks, fixed_logits, dataset.fixed.names, output_dir / "fixed_reconstructions.png", run_config.grid_size, threshold)
    torch.save(model.state_dict(), output_dir / "multitask_model_weights.pt")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary


__all__ = ["run_multitask_one_model"]

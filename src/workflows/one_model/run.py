"""Run the official direct gradient-to-mask model."""

from __future__ import annotations

import json

import torch

from datasets import build_two_stage_datasets, save_two_stage_dataset
from models import evaluate_stage2_predictions, evaluate_stage2_predictions_by_shape, select_best_stage2_threshold, set_torch_seed
from models.one_model import GradientToMaskMLP, GradientToMaskModel, fit_one_model, predict_one_model_logits
from plotting import save_measurement_points_plot, save_reconstruction_examples, save_shape_gallery
from shapes import create_fixed_benchmark_shapes


def _training_options(run_config, device: torch.device) -> dict[str, object]:
    """Translate the shared mask config into named one-model options."""

    training = run_config.model.training
    return {
        "epochs": training.epochs,
        "batch_size": training.batch_size,
        "learning_rate": training.learning_rate,
        "device": device,
        "validation_frequency": training.validation_frequency,
        "verbose": training.verbose,
        "early_stopping_patience": training.early_stopping_patience,
        "train_shape_types": None,
        "grid_size": run_config.grid_size,
        "use_rectangle_edge_weighting": run_config.model.use_rectangle_edge_weighting,
        "rectangle_edge_weight": run_config.model.rectangle_edge_weight,
        "rectangle_edge_width": run_config.model.rectangle_edge_width,
        "edge_weight_mode": run_config.model.edge_weight_mode,
        "annulus_edge_weight": run_config.model.annulus_edge_weight,
        "annulus_edge_width": run_config.model.annulus_edge_width,
        "min_epochs": training.min_epochs,
        "min_improvement": training.min_improvement,
        "lr_drop_factor": training.lr_drop_factor,
        "lr_drop_period": training.lr_drop_period,
        "weight_decay": training.weight_decay,
        "gradient_clip_norm": training.gradient_clip_norm,
        "loss_type": training.loss_type,
        "dice_loss_weight": training.dice_loss_weight,
        "dice_smooth": training.dice_smooth,
        "use_foreground_pos_weight": run_config.model.use_foreground_pos_weight,
    }


def run_one_model(run_config, device: torch.device | None = None) -> dict[str, object]:
    """Train, evaluate, and persist one direct gradient-to-mask experiment."""

    output_dir = run_config.run_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_torch_seed(run_config.seed)

    dataset = build_two_stage_datasets(run_config)
    dataset_paths = save_two_stage_dataset(dataset, output_dir / "datasets")
    model_class = GradientToMaskMLP if run_config.model.model_type == "mlp" else GradientToMaskModel
    model = model_class(
        input_dim=run_config.gradient_feature_size,
        output_dim=run_config.mask_pixels,
        hidden_dims=run_config.model.hidden_layer_sizes,
        dropout_rates=run_config.model.dropout_rates,
        latent_grid_size=run_config.model.latent_grid_size,
        latent_channels=run_config.model.latent_channels,
        decoder_channels=run_config.model.decoder_channels,
    )
    training_options = _training_options(run_config, device)
    training_options["train_shape_types"] = dataset.train.shape_types
    training_result = fit_one_model(
        model=model,
        gradient_features=dataset.train.gradient_data,
        target_masks=dataset.train.masks,
        validation_gradient_features=dataset.validation.gradient_data,
        validation_masks=dataset.validation.masks,
        **training_options,
    )

    validation_logits = predict_one_model_logits(model, dataset.validation.gradient_data, device, training_result)
    test_logits = predict_one_model_logits(model, dataset.test.gradient_data, device, training_result)
    fixed_logits = predict_one_model_logits(model, dataset.fixed.gradient_data, device, training_result)
    threshold_summary = select_best_stage2_threshold(dataset.validation.masks, validation_logits, run_config.threshold_candidates)
    threshold = float(threshold_summary["selected_threshold"])
    metrics = {
        "test": evaluate_stage2_predictions(dataset.test.masks, test_logits, threshold),
        "fixed": evaluate_stage2_predictions(dataset.fixed.masks, fixed_logits, threshold),
    }
    metrics_by_shape = {
        "test": evaluate_stage2_predictions_by_shape(dataset.test.masks, test_logits, threshold, dataset.test.shape_types),
        "fixed": evaluate_stage2_predictions_by_shape(dataset.fixed.masks, fixed_logits, threshold, dataset.fixed.shape_types),
    }

    save_measurement_points_plot(dataset.measurement_points, output_dir / "measurement_points.png")
    save_shape_gallery(create_fixed_benchmark_shapes(), output_dir / "fixed_shapes.png", run_config.grid_size)
    save_reconstruction_examples(dataset.fixed.masks, fixed_logits, dataset.fixed.names, output_dir / "fixed_reconstructions.png", run_config.grid_size, threshold)
    torch.save(model.state_dict(), output_dir / "one_model_weights.pt")
    summary = {
        "model_count": 1,
        "official_model": True,
        "config": {
            "N": run_config.N,
            "training_samples": run_config.training_samples,
            "validation_samples": run_config.validation_samples,
            "test_samples": run_config.test_samples,
            "threshold": threshold,
            "model_type": run_config.model.model_type,
        },
        "metrics": metrics,
        "metrics_by_shape": metrics_by_shape,
        "threshold_summary": threshold_summary,
        "dataset_paths": {key: str(value) for key, value in dataset_paths.items()},
        "training_history": training_result.history,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary


__all__ = ["run_one_model"]

"""Checkpoint loaders used by the active notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from models.common_train import ModelTrainingResult
from models.one_model import GradientToMaskMLP, GradientToMaskModel, predict_one_model_logits
from models import Stage1Regressor, predict_stage1_coefficients, predict_stage2_logits
from pipeline.run import _build_stage2_model, _build_stage2_features


def load_one_stage_predictor(
    output_dir: Path,
    config,
    device: torch.device,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    """Load a one-stage checkpoint and return a normalized logits predictor."""

    run_dir = output_dir / f"N_{config.N}"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    model_class = GradientToMaskMLP if config.model.model_type == "mlp" else GradientToMaskModel
    model_kwargs = {
        "input_dim": config.gradient_feature_size,
        "output_dim": config.mask_pixels,
        "hidden_dims": config.model.hidden_layer_sizes,
        "dropout_rates": config.model.dropout_rates,
    }
    if model_class is GradientToMaskModel:
        model_kwargs.update(
            {
                "latent_grid_size": config.model.latent_grid_size,
                "latent_channels": config.model.latent_channels,
                "decoder_channels": config.model.decoder_channels,
            }
        )
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(run_dir / "one_model_weights.pt", map_location=device, weights_only=True))
    training_result = ModelTrainingResult(
        history=summary["training_history"],
        input_mean=np.asarray(summary["training_input_mean"], dtype=np.float32),
        input_std=np.asarray(summary["training_input_std"], dtype=np.float32),
    )

    def predict(features: np.ndarray) -> np.ndarray:
        return predict_one_model_logits(model, features, device, training_result)

    return predict, summary


def load_two_stage_predictor(
    output_dir: Path,
    config,
    device: torch.device,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    """Load a two-stage checkpoint and expose gradient-to-mask prediction."""

    run_dir = output_dir / f"N_{config.N}"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    stage1_model = Stage1Regressor(
        input_dim=config.gradient_feature_size,
        output_dim=config.coefficient_size,
        hidden_dims=config.model.stage1.hidden_layer_sizes,
        dropout_rates=config.model.stage1.dropout_rates,
    )
    stage1_model.load_state_dict(torch.load(run_dir / "stage1_model.pt", map_location=device, weights_only=True))
    stage2_model = _build_stage2_model(config)
    stage2_model.load_state_dict(torch.load(run_dir / "stage2_model.pt", map_location=device, weights_only=True))
    stage1_result = ModelTrainingResult(
        history=summary["stage1_history"],
        input_mean=np.asarray(summary["stage1_input_mean"], dtype=np.float32),
        input_std=np.asarray(summary["stage1_input_std"], dtype=np.float32),
        target_mean=np.asarray(summary["stage1_target_mean"], dtype=np.float32),
        target_std=np.asarray(summary["stage1_target_std"], dtype=np.float32),
    )
    stage2_result = ModelTrainingResult(
        history=summary["stage2_history"],
        input_mean=np.asarray(summary["stage2_input_mean"], dtype=np.float32),
        input_std=np.asarray(summary["stage2_input_std"], dtype=np.float32),
    )

    def predict(features: np.ndarray) -> np.ndarray:
        coefficients = predict_stage1_coefficients(
            stage1_model, features, device=device, training_result=stage1_result
        )
        stage2_features = _build_stage2_features(config, coefficients, features)
        return predict_stage2_logits(stage2_model, stage2_features, device=device, training_result=stage2_result)

    return predict, summary


__all__ = ["load_one_stage_predictor", "load_two_stage_predictor"]

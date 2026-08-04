"""Training and inference helpers for the official one-model system."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..common_train import ModelTrainingResult
from ..stage2.train import fit_stage2_model, predict_stage2_logits


def fit_one_model(
    model: nn.Module,
    gradient_features: np.ndarray,
    target_masks: np.ndarray,
    validation_gradient_features: np.ndarray,
    validation_masks: np.ndarray,
    **training_options,
) -> ModelTrainingResult:
    """Train a direct gradient-to-mask model using the shared mask trainer."""

    return fit_stage2_model(
        model=model,
        train_features=gradient_features,
        train_targets=target_masks,
        val_features=validation_gradient_features,
        val_targets=validation_masks,
        **training_options,
    )


def predict_one_model_logits(
    model: nn.Module,
    gradient_features: np.ndarray,
    device: torch.device,
    training_result: ModelTrainingResult,
) -> np.ndarray:
    """Predict mask logits directly from gradient features."""

    return predict_stage2_logits(model, gradient_features, device=device, training_result=training_result)


__all__ = ["fit_one_model", "predict_one_model_logits"]

"""Training and inference helpers for the official one-model system."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common_train import ModelTrainingResult, compute_input_normalization, compute_target_normalization, normalize_values
from ..stage2.train import fit_stage2_model, predict_stage2_logits
from ..stage2.losses import WeightedBinaryMaskLoss
from ..stage2.weights import compute_shape_edge_pixel_weights


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


def fit_multitask_one_model(
    model: nn.Module,
    gradient_features: np.ndarray,
    target_masks: np.ndarray,
    target_coefficients: np.ndarray,
    validation_gradient_features: np.ndarray,
    validation_masks: np.ndarray,
    validation_coefficients: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    coefficient_loss_weight: float = 0.1,
    validation_frequency: int | None = None,
    verbose: bool = False,
    early_stopping_patience: int | None = None,
    train_shape_types: tuple[str, ...] | None = None,
    grid_size: int | None = None,
    use_rectangle_edge_weighting: bool = False,
    rectangle_edge_weight: float = 3.0,
    rectangle_edge_width: int = 2,
    edge_weight_mode: str = "rectangle",
    annulus_edge_weight: float = 1.0,
    annulus_edge_width: int = 2,
    min_epochs: int | None = None,
    min_improvement: float | None = None,
    lr_drop_factor: float | None = None,
    lr_drop_period: int | None = None,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
    loss_type: str = "bce_dice",
    dice_loss_weight: float = 1.0,
    dice_smooth: float = 1.0,
    iou_loss_weight: float = 0.0,
) -> ModelTrainingResult:
    """Train shared mask and coefficient heads with a combined objective."""

    if coefficient_loss_weight < 0:
        raise ValueError("coefficient_loss_weight must be non-negative")
    input_mean, input_std = compute_input_normalization(gradient_features)
    coefficient_mean, coefficient_std = compute_target_normalization(target_coefficients)
    normalized_train_features = normalize_values(gradient_features, input_mean, input_std)
    normalized_val_features = normalize_values(validation_gradient_features, input_mean, input_std)
    normalized_train_coefficients = normalize_values(target_coefficients, coefficient_mean, coefficient_std)
    normalized_val_coefficients = normalize_values(validation_coefficients, coefficient_mean, coefficient_std)

    train_tensors = [
        torch.tensor(normalized_train_features, dtype=torch.float32),
        torch.tensor(target_masks, dtype=torch.float32),
        torch.tensor(normalized_train_coefficients, dtype=torch.float32),
    ]
    if use_rectangle_edge_weighting:
        if train_shape_types is None or grid_size is None:
            raise ValueError("train_shape_types and grid_size are required for edge weighting")
        pixel_weights = compute_shape_edge_pixel_weights(
            masks=target_masks,
            shape_types=train_shape_types,
            grid_size=grid_size,
            edge_weight=rectangle_edge_weight,
            edge_width=rectangle_edge_width,
            edge_weight_mode=edge_weight_mode,
            annulus_edge_weight=annulus_edge_weight,
            annulus_edge_width=annulus_edge_width,
        )
        train_tensors.append(torch.tensor(pixel_weights, dtype=torch.float32))

    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        TensorDataset(
            torch.tensor(normalized_val_features, dtype=torch.float32),
            torch.tensor(validation_masks, dtype=torch.float32),
            torch.tensor(normalized_val_coefficients, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    mask_criterion = WeightedBinaryMaskLoss(
        pos_weight=None,
        loss_type=loss_type,
        dice_loss_weight=dice_loss_weight,
        dice_smooth=dice_smooth,
        iou_loss_weight=iou_loss_weight,
    )
    coefficient_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if lr_drop_factor is not None and lr_drop_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)

    model.to(device)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_mask_loss": [], "train_coefficient_loss": []}
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait_count = 0
    for epoch_index in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_mask_losses: list[float] = []
        epoch_coefficient_losses: list[float] = []
        for batch in train_loader:
            features, masks, coefficients = batch[:3]
            pixel_weights = batch[3] if len(batch) > 3 else None
            features, masks, coefficients = features.to(device), masks.to(device), coefficients.to(device)
            if pixel_weights is not None:
                pixel_weights = pixel_weights.to(device)
            optimizer.zero_grad()
            mask_logits, coefficient_predictions = model(features)
            mask_loss = mask_criterion(mask_logits, masks, pixel_weights)
            coefficient_loss = coefficient_criterion(coefficient_predictions, coefficients)
            loss = mask_loss + coefficient_loss_weight * coefficient_loss
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_mask_losses.append(float(mask_loss.detach().cpu()))
            epoch_coefficient_losses.append(float(coefficient_loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            validation_losses = []
            for features, masks, coefficients in validation_loader:
                features, masks, coefficients = features.to(device), masks.to(device), coefficients.to(device)
                mask_logits, coefficient_predictions = model(features)
                validation_losses.append(
                    float((mask_criterion(mask_logits, masks) + coefficient_loss_weight * coefficient_criterion(coefficient_predictions, coefficients)).cpu())
                )
        validation_loss = float(np.mean(validation_losses))
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(validation_loss)
        history["train_mask_loss"].append(float(np.mean(epoch_mask_losses)))
        history["train_coefficient_loss"].append(float(np.mean(epoch_coefficient_losses)))
        if verbose:
            print(f"Epoch {epoch_index + 1}/{epochs} train={history['train_loss'][-1]:.5f} val={validation_loss:.5f}")

        improved = validation_loss < best_validation_loss * (1.0 - (min_improvement or 0.0))
        if best_state is None or improved:
            best_validation_loss = min(best_validation_loss, validation_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait_count = 0
        elif min_epochs is None or epoch_index + 1 >= min_epochs:
            wait_count += 1
        if scheduler is not None:
            scheduler.step()
        if early_stopping_patience is not None and wait_count >= early_stopping_patience and (min_epochs is None or epoch_index + 1 >= min_epochs):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return ModelTrainingResult(
        history=history,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=coefficient_mean,
        target_std=coefficient_std,
    )


def predict_multitask_one_model(
    model: nn.Module,
    gradient_features: np.ndarray,
    device: torch.device,
    training_result: ModelTrainingResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mask logits and denormalized coefficient features."""

    normalized_features = normalize_values(gradient_features, training_result.input_mean, training_result.input_std)
    model.eval()
    model.to(device)
    with torch.no_grad():
        mask_logits, coefficient_predictions = model(torch.tensor(normalized_features, dtype=torch.float32, device=device))
    coefficients = coefficient_predictions.cpu().numpy() * training_result.target_std + training_result.target_mean
    return mask_logits.cpu().numpy(), coefficients


__all__ = ["fit_one_model", "predict_one_model_logits", "fit_multitask_one_model", "predict_multitask_one_model"]

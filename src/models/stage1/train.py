"""Stage 1 training and inference helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..common_train import (
    ModelTrainingResult,
    _evaluate_model,
    _to_loader,
    compute_input_normalization,
    compute_target_normalization,
    denormalize_values,
    normalize_values,
    predict_tensor,
)


def stop_if_overfitting(
    validation_loss: float,
    best_validation_loss: float,
    wait_count: int,
    patience: int | None,
) -> tuple[bool, float, int]:
    """Simple early stopping used by Stage 1."""

    if validation_loss < best_validation_loss:
        return False, validation_loss, 0

    updated_wait_count = wait_count + 1
    should_stop = patience is not None and updated_wait_count >= patience
    return should_stop, best_validation_loss, updated_wait_count


def _fit_stage1_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    validation_frequency: int | None = None,
    verbose: bool = True,
    early_stopping_patience: int | None = None,
) -> ModelTrainingResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = _to_loader(train_features, train_targets, batch_size=batch_size, shuffle=True)
    val_loader = _to_loader(val_features, val_targets, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "validation_steps": []}
    best_validation_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    wait_count = 0
    global_step = 0
    stop_training = False

    for epoch_index in range(epochs):
        model.train()
        train_losses: list[float] = []
        validated_this_epoch = False

        def run_validation() -> None:
            nonlocal best_validation_loss, best_state_dict, wait_count, stop_training, validated_this_epoch
            current_validation_loss = _evaluate_model(model, val_loader, criterion, device)
            history["val_loss"].append(current_validation_loss)
            history["validation_steps"].append(global_step)
            validated_this_epoch = True

            if verbose:
                latest_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
                print(
                    f"Epoch {epoch_index + 1}/{epochs} | step {global_step} | "
                    f"train_loss={latest_train_loss:.6f} | val_loss={current_validation_loss:.6f}"
                )

            stop_training, updated_best_validation_loss, updated_wait_count = stop_if_overfitting(
                validation_loss=current_validation_loss,
                best_validation_loss=best_validation_loss,
                wait_count=wait_count,
                patience=early_stopping_patience,
            )
            if updated_best_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation_loss = updated_best_validation_loss
            wait_count = updated_wait_count

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            global_step += 1

            if validation_frequency is not None and validation_frequency > 0 and global_step % validation_frequency == 0:
                run_validation()
                if stop_training:
                    break
                model.train()

        if not validated_this_epoch:
            run_validation()

        history["train_loss"].append(float(np.mean(train_losses)))
        if stop_training:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return ModelTrainingResult(history=history)


def fit_stage1_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    validation_frequency: int | None = None,
    verbose: bool = True,
    early_stopping_patience: int | None = 10,
) -> ModelTrainingResult:
    """Train Stage 1 with MSE on normalized inputs and targets."""

    input_mean, input_std = compute_input_normalization(train_features)
    target_mean, target_std = compute_target_normalization(train_targets)

    normalized_train_features = normalize_values(train_features, input_mean, input_std)
    normalized_val_features = normalize_values(val_features, input_mean, input_std)
    normalized_train_targets = normalize_values(train_targets, target_mean, target_std)
    normalized_val_targets = normalize_values(val_targets, target_mean, target_std)

    training_result = _fit_stage1_model(
        model=model,
        train_features=normalized_train_features,
        train_targets=normalized_train_targets,
        val_features=normalized_val_features,
        val_targets=normalized_val_targets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        validation_frequency=validation_frequency,
        verbose=verbose,
        early_stopping_patience=early_stopping_patience,
    )
    training_result.input_mean = input_mean
    training_result.input_std = input_std
    training_result.target_mean = target_mean
    training_result.target_std = target_std
    return training_result


def predict_stage1_coefficients(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    training_result: ModelTrainingResult,
) -> np.ndarray:
    """Predict de-normalized Stage 1 coefficient features from raw gradients."""

    if training_result.input_mean is None or training_result.input_std is None:
        raise ValueError("Stage 1 input normalization values are missing")
    if training_result.target_mean is None or training_result.target_std is None:
        raise ValueError("Stage 1 target normalization values are missing")

    normalized_features = normalize_values(features, training_result.input_mean, training_result.input_std)
    normalized_predictions = predict_tensor(model, normalized_features, device=device)
    return denormalize_values(normalized_predictions, training_result.target_mean, training_result.target_std)


__all__ = ["fit_stage1_model", "predict_stage1_coefficients", "stop_if_overfitting"]

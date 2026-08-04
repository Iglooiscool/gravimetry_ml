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


def _coefficient_features_to_complex_tensor(features: torch.Tensor) -> torch.Tensor:
    """Convert concatenated real/imag coefficient features into a complex tensor."""

    half_index = features.shape[-1] // 2
    return torch.complex(features[..., :half_index], features[..., half_index:])


def _complex_measurements_to_feature_tensor(values: torch.Tensor) -> torch.Tensor:
    """Flatten complex measurements into concatenated real/imag features."""

    return torch.cat((values.real, values.imag), dim=-1)


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
    lr_drop_factor: float | None = None,
    lr_drop_period: int | None = None,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
    measurement_loss_weight: float = 0.0,
    input_mean: np.ndarray | None = None,
    input_std: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    measurement_matrix: np.ndarray | None = None,
) -> ModelTrainingResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if lr_drop_factor is not None and lr_drop_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)
    train_loader = _to_loader(train_features, train_targets, batch_size=batch_size, shuffle=True)
    val_loader = _to_loader(val_features, val_targets, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()

    use_measurement_loss = measurement_loss_weight > 0.0
    if use_measurement_loss:
        if input_mean is None or input_std is None or target_mean is None or target_std is None or measurement_matrix is None:
            raise ValueError("Stage 1 measurement loss requires normalization values and the measurement matrix")
        input_mean_tensor = torch.tensor(input_mean, dtype=torch.float32, device=device)
        input_std_tensor = torch.tensor(input_std, dtype=torch.float32, device=device)
        target_mean_tensor = torch.tensor(target_mean, dtype=torch.float32, device=device)
        target_std_tensor = torch.tensor(target_std, dtype=torch.float32, device=device)
        measurement_matrix_tensor = torch.tensor(measurement_matrix, dtype=torch.complex64, device=device)

    history = {"train_loss": [], "val_loss": [], "validation_steps": []}
    best_validation_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    wait_count = 0
    global_step = 0
    for epoch_index in range(epochs):
        model.train()
        train_losses: list[float] = []
        validated_this_epoch = False
        epoch_best_validation_loss = float("inf")
        previous_best_validation_loss = best_validation_loss

        def run_validation() -> None:
            nonlocal best_validation_loss, best_state_dict, epoch_best_validation_loss, validated_this_epoch
            current_validation_loss = _evaluate_model(model, val_loader, criterion, device)
            history["val_loss"].append(current_validation_loss)
            history["validation_steps"].append(global_step)
            validated_this_epoch = True
            epoch_best_validation_loss = min(epoch_best_validation_loss, current_validation_loss)

            if verbose:
                latest_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
                print(
                    f"Epoch {epoch_index + 1}/{epochs} | step {global_step} | "
                    f"train_loss={latest_train_loss:.6f} | val_loss={current_validation_loss:.6f}"
                )

            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            coefficient_loss = criterion(predictions, batch_targets)
            loss = coefficient_loss
            if use_measurement_loss:
                denormalized_predictions = predictions * target_std_tensor + target_mean_tensor
                predicted_coefficients = _coefficient_features_to_complex_tensor(denormalized_predictions)
                reconstructed_measurements = predicted_coefficients @ measurement_matrix_tensor.transpose(0, 1)
                reconstructed_features = _complex_measurements_to_feature_tensor(reconstructed_measurements)
                normalized_reconstructed_features = (reconstructed_features - input_mean_tensor) / input_std_tensor
                measurement_loss = criterion(normalized_reconstructed_features, batch_features)
                loss = coefficient_loss + float(measurement_loss_weight) * measurement_loss
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            global_step += 1

            if validation_frequency is not None and validation_frequency > 0 and global_step % validation_frequency == 0:
                run_validation()
                model.train()

        if not validated_this_epoch:
            run_validation()

        history["train_loss"].append(float(np.mean(train_losses)))
        improved_this_epoch = epoch_best_validation_loss < previous_best_validation_loss
        if improved_this_epoch:
            wait_count = 0
        else:
            wait_count += 1
        if early_stopping_patience is not None and wait_count >= early_stopping_patience:
            break
        if scheduler is not None:
            scheduler.step()

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
    lr_drop_factor: float | None = None,
    lr_drop_period: int | None = None,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
    measurement_loss_weight: float = 0.0,
    measurement_matrix: np.ndarray | None = None,
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
        lr_drop_factor=lr_drop_factor,
        lr_drop_period=lr_drop_period,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        measurement_loss_weight=measurement_loss_weight,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        measurement_matrix=measurement_matrix,
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

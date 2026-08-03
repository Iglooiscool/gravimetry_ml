"""Task 9 head training wrappers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .combined import Task9TrainedHead
from ..common_train import ModelTrainingResult, _evaluate_model, _to_loader, compute_input_normalization, normalize_values
from ..stage2.losses import WeightedBinaryMaskLoss
from ..stage2.train import stop_if_safe
from ..stage2.weights import compute_shape_edge_pixel_weights


def _fit_task9_head(
    *,
    model: nn.Module,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    criterion: nn.Module,
    device: torch.device,
    validation_frequency: int | None = None,
    verbose: bool = True,
    early_stopping_patience: int | None = None,
    train_sample_weights: np.ndarray | None = None,
    min_epochs: int | None = None,
    min_improvement: float | None = None,
    lr_drop_factor: float | None = None,
    lr_drop_period: int | None = None,
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
) -> ModelTrainingResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if lr_drop_factor is not None and lr_drop_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)
    train_loader = _to_loader(train_features, train_targets, batch_size=batch_size, shuffle=True, sample_weights=train_sample_weights)
    val_loader = _to_loader(val_features, val_targets, batch_size=batch_size, shuffle=False)

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
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch_index + 1}/{epochs} | step {global_step} | "
                    f"train_loss={latest_train_loss:.6f} | val_loss={current_validation_loss:.6f} | lr={current_lr:.6f}"
                )

            if current_validation_loss < best_validation_loss:
                best_validation_loss = current_validation_loss
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        for batch in train_loader:
            batch_features, batch_targets = batch[:2]
            batch_sample_weights = batch[2] if len(batch) > 2 else None
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            if batch_sample_weights is not None:
                batch_sample_weights = batch_sample_weights.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            if batch_sample_weights is None:
                loss = criterion(predictions, batch_targets)
            else:
                loss = criterion(predictions, batch_targets, batch_sample_weights)
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
        meets_epoch_floor = min_epochs is None or (epoch_index + 1) >= min_epochs
        effective_min_improvement = 0.0 if min_improvement is None else float(min_improvement)
        improved_this_epoch = False
        if epoch_best_validation_loss < previous_best_validation_loss:
            if previous_best_validation_loss == float("inf"):
                improved_this_epoch = True
            elif epoch_best_validation_loss < previous_best_validation_loss * (1.0 - effective_min_improvement):
                improved_this_epoch = True
        if improved_this_epoch:
            wait_count = 0
        elif meets_epoch_floor:
            wait_count += 1
        if scheduler is not None:
            scheduler.step()
        if early_stopping_patience is not None and wait_count >= early_stopping_patience and meets_epoch_floor:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return ModelTrainingResult(history=history)


def train_task9_head(
    *,
    name: str,
    model,
    train_features,
    train_targets,
    val_features,
    val_targets,
    device,
    training_config,
    train_shape_types=None,
    grid_size=None,
    use_rectangle_edge_weighting: bool = False,
    rectangle_edge_weight: float = 3.0,
    rectangle_edge_width: int = 2,
    edge_weight_mode: str = "rectangle",
) -> Task9TrainedHead:
    """Train one Task 9 head with a Task 9-owned training path."""

    input_mean, input_std = compute_input_normalization(train_features)
    normalized_train_features = normalize_values(train_features, input_mean, input_std)
    normalized_val_features = normalize_values(val_features, input_mean, input_std)

    train_sample_weights = None
    if use_rectangle_edge_weighting:
        if train_shape_types is None:
            raise ValueError("train_shape_types are required when rectangle edge weighting is enabled")
        if grid_size is None:
            raise ValueError("grid_size is required when rectangle edge weighting is enabled")
        train_sample_weights = compute_shape_edge_pixel_weights(
            masks=train_targets,
            shape_types=train_shape_types,
            grid_size=grid_size,
            edge_weight=rectangle_edge_weight,
            edge_width=rectangle_edge_width,
            edge_weight_mode=edge_weight_mode,
        )

    positive_fraction = float(np.mean(train_targets))
    positive_fraction = min(max(positive_fraction, 1e-4), 1.0 - 1e-4)
    pos_weight = (1.0 - positive_fraction) / positive_fraction

    training_result = _fit_task9_head(
        model=model,
        train_features=normalized_train_features,
        train_targets=train_targets,
        val_features=normalized_val_features,
        val_targets=val_targets,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        criterion=WeightedBinaryMaskLoss(
            pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device),
            loss_type=training_config.loss_type,
            dice_loss_weight=training_config.dice_loss_weight,
            dice_smooth=training_config.dice_smooth,
        ),
        device=device,
        validation_frequency=training_config.validation_frequency,
        verbose=training_config.verbose,
        early_stopping_patience=training_config.early_stopping_patience,
        min_epochs=training_config.min_epochs,
        min_improvement=training_config.min_improvement,
        lr_drop_factor=training_config.lr_drop_factor,
        lr_drop_period=training_config.lr_drop_period,
        weight_decay=training_config.weight_decay,
        gradient_clip_norm=training_config.gradient_clip_norm,
        train_sample_weights=train_sample_weights,
    )
    training_result.input_mean = input_mean
    training_result.input_std = input_std
    return Task9TrainedHead(name=name, model=model, training_result=training_result)


__all__ = ["train_task9_head"]

"""Shared training helpers used by both stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelTrainingResult:
    """Training history plus any normalization values learned during fitting."""

    history: dict[str, list[float]]
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None
    target_mean: np.ndarray | None = None
    target_std: np.ndarray | None = None


def set_torch_seed(seed: int) -> None:
    """Keep seeding in one helper so runs are repeatable."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    tensors = [
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    ]
    if sample_weights is not None:
        tensors.append(torch.tensor(sample_weights, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Compute the mean validation loss for the current model state."""

    model.eval()
    validation_losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch_features, batch_targets = batch[:2]
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            validation_losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(validation_losses))


def compute_input_normalization(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute z-score normalization values from training inputs."""

    input_mean = train_values.mean(axis=0)
    input_std = train_values.std(axis=0)
    input_std = np.where(input_std == 0, 1.0, input_std)
    return input_mean, input_std


def compute_target_normalization(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute safe z-score normalization values for targets."""

    target_mean = train_values.mean(axis=0)
    target_std = train_values.std(axis=0)
    target_std = np.where(target_std == 0, 1.0, target_std)
    return target_mean, target_std


def normalize_values(values: np.ndarray, mean_values: np.ndarray, std_values: np.ndarray) -> np.ndarray:
    """Apply z-score style normalization to an array."""

    return (values - mean_values) / std_values


def denormalize_values(values: np.ndarray, mean_values: np.ndarray, std_values: np.ndarray) -> np.ndarray:
    """Undo z-score style normalization on an array."""

    return values * std_values + mean_values


def predict_tensor(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Run inference on a full NumPy feature array and return NumPy predictions."""

    model.eval()
    model.to(device)
    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32, device=device)
        return model(tensor).detach().cpu().numpy()


__all__ = [
    "ModelTrainingResult",
    "set_torch_seed",
    "compute_input_normalization",
    "compute_target_normalization",
    "normalize_values",
    "denormalize_values",
    "predict_tensor",
    "_to_loader",
    "_evaluate_model",
]

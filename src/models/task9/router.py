"""Measurement-only router for the experimental three-model workflow."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common_train import ModelTrainingResult, compute_input_normalization, normalize_values


class Task9CoefficientRouter(nn.Module):
    """Binary classifier deciding whether to use the two-circle specialist."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def fit_task9_router(
    model: Task9CoefficientRouter,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> ModelTrainingResult:
    """Train the router using Stage 1-predicted coefficients only."""

    input_mean, input_std = compute_input_normalization(train_features)
    train_features = normalize_values(train_features, input_mean, input_std)
    validation_features = normalize_values(validation_features, input_mean, input_std)
    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    validation_x = torch.tensor(validation_features, dtype=torch.float32)
    validation_y = torch.tensor(validation_labels, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(validation_x.to(device)), validation_y.to(device))
        val_loss = float(validation_loss.detach().cpu())
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return ModelTrainingResult(history=history, input_mean=input_mean, input_std=input_std)


def predict_task9_router(
    model: Task9CoefficientRouter,
    features: np.ndarray,
    *,
    training_result: ModelTrainingResult,
    device: torch.device,
) -> np.ndarray:
    """Predict specialist probabilities from predicted coefficient features."""

    normalized = normalize_values(features, training_result.input_mean, training_result.input_std)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(normalized, dtype=torch.float32).to(device))
    return logits.detach().cpu().numpy()


__all__ = ["Task9CoefficientRouter", "fit_task9_router", "predict_task9_router"]

"""One-stage mixture-of-experts model with a learned annulus gate."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..common_train import ModelTrainingResult, compute_input_normalization, normalize_values
from ..stage2.losses import WeightedBinaryMaskLoss
from ..stage2.weights import compute_shape_edge_pixel_weights
from .model import GradientToMaskModel


class AnnulusRouterOneStage(nn.Module):
    """Direct gradient-to-mask experts combined by a learned gate."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 1024),
        dropout_rates: tuple[float, ...] = (0.1, 0.1),
        latent_grid_size: int = 16,
        latent_channels: int = 160,
        decoder_channels: tuple[int, ...] = (160, 128, 96, 64, 32),
        router_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        expert_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "dropout_rates": dropout_rates,
            "latent_grid_size": latent_grid_size,
            "latent_channels": latent_channels,
            "decoder_channels": decoder_channels,
        }
        self.general_expert = GradientToMaskModel(**expert_kwargs)
        self.annulus_expert = GradientToMaskModel(**expert_kwargs)
        self.router = nn.Sequential(
            nn.Linear(input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, 1),
        )

    def forward_with_router(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixed logits, both expert logits, and router logits."""

        general_logits = self.general_expert(inputs)
        annulus_logits = self.annulus_expert(inputs)
        router_logits = self.router(inputs).squeeze(-1)
        annulus_probability = torch.sigmoid(router_logits).unsqueeze(-1)
        mixed_logits = (1.0 - annulus_probability) * general_logits + annulus_probability * annulus_logits
        return mixed_logits, general_logits, annulus_logits, router_logits

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the differentiably gated mask logits."""

        return self.forward_with_router(inputs)[0]


def fit_annulus_router(
    model: AnnulusRouterOneStage,
    gradient_features: np.ndarray,
    target_masks: np.ndarray,
    shape_types: tuple[str, ...],
    validation_gradient_features: np.ndarray,
    validation_masks: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    router_loss_weight: float = 0.25,
    specialist_loss_weight: float = 1.0,
    annulus_edge_weight: float = 1.0,
    annulus_edge_width: int = 3,
    grid_size: int = 32,
) -> ModelTrainingResult:
    """Train the gated experts with mask and learned-routing objectives."""

    input_mean, input_std = compute_input_normalization(gradient_features)
    train_features = normalize_values(gradient_features, input_mean, input_std)
    validation_features = normalize_values(validation_gradient_features, input_mean, input_std)
    router_targets = np.asarray([shape == "annulus" for shape in shape_types], dtype=np.float32)
    pixel_weights = compute_shape_edge_pixel_weights(
        target_masks,
        shape_types,
        grid_size,
        edge_weight=1.0,
        edge_width=annulus_edge_width,
        edge_weight_mode="all",
        annulus_edge_weight=annulus_edge_weight,
        annulus_edge_width=annulus_edge_width,
    )
    loader = DataLoader(
        TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(target_masks, dtype=torch.float32),
            torch.tensor(router_targets, dtype=torch.float32),
            torch.tensor(pixel_weights, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    mask_criterion = WeightedBinaryMaskLoss(pos_weight=None, loss_type="bce_dice", dice_loss_weight=1.0, dice_smooth=1.0)
    positive_count = max(float(router_targets.sum()), 1.0)
    negative_count = max(float(len(router_targets) - router_targets.sum()), 1.0)
    router_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_x = torch.tensor(validation_features, dtype=torch.float32).to(device)
    validation_y = torch.tensor(validation_masks, dtype=torch.float32).to(device)
    model.to(device)
    best_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_mask_loss": [], "train_router_loss": []}
    for _ in range(epochs):
        model.train()
        losses, mask_losses, router_losses = [], [], []
        for batch_x, batch_y, batch_router_y, batch_weights in loader:
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_router_y, batch_weights = batch_router_y.to(device), batch_weights.to(device)
            mixed, _general, annulus, router_logits = model.forward_with_router(batch_x)
            mask_loss = mask_criterion(mixed, batch_y, batch_weights)
            annulus_indices = batch_router_y.bool()
            specialist_loss = (
                mask_criterion(annulus[annulus_indices], batch_y[annulus_indices], batch_weights[annulus_indices])
                if annulus_indices.any()
                else torch.zeros((), device=device)
            )
            router_loss = router_criterion(router_logits, batch_router_y)
            loss = mask_loss + specialist_loss_weight * specialist_loss + router_loss_weight * router_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mask_losses.append(float(mask_loss.detach().cpu()))
            router_losses.append(float(router_loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = mask_criterion(model(validation_x), validation_y)
        val_value = float(validation_loss.detach().cpu())
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_value)
        history["train_mask_loss"].append(float(np.mean(mask_losses)))
        history["train_router_loss"].append(float(np.mean(router_losses)))
        if val_value < best_loss:
            best_loss = val_value
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return ModelTrainingResult(history=history, input_mean=input_mean, input_std=input_std)


def predict_annulus_router(
    model: AnnulusRouterOneStage,
    gradient_features: np.ndarray,
    *,
    device: torch.device,
    training_result: ModelTrainingResult,
    return_router_logits: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Predict mask logits and optionally return learned router logits."""

    normalized = normalize_values(gradient_features, training_result.input_mean, training_result.input_std)
    model.eval()
    with torch.no_grad():
        mixed, _general, _annulus, router_logits = model.forward_with_router(torch.tensor(normalized, dtype=torch.float32).to(device))
    mixed_array = mixed.detach().cpu().numpy()
    if return_router_logits:
        return mixed_array, router_logits.detach().cpu().numpy()
    return mixed_array


__all__ = ["AnnulusRouterOneStage", "fit_annulus_router", "predict_annulus_router"]

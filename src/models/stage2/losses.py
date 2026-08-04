"""Loss functions used by Stage 2 training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WeightedBinaryMaskLoss(nn.Module):
    """Stage 2 BCE, BCE+Dice, or MSE loss with optional per-pixel weights."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None,
        loss_type: str = "bce",
        dice_loss_weight: float = 0.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        if loss_type not in {"bce", "bce_dice", "mse"}:
            raise ValueError("loss_type must be 'bce', 'bce_dice', or 'mse'")
        self.loss_type = loss_type
        self.dice_loss_weight = float(dice_loss_weight)
        self.dice_smooth = float(dice_smooth)

    def _compute_dice_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(predictions)
        if pixel_weights is not None:
            intersection = (pixel_weights * probabilities * targets).sum(dim=1)
            denominator = (pixel_weights * probabilities).sum(dim=1) + (pixel_weights * targets).sum(dim=1)
        else:
            intersection = (probabilities * targets).sum(dim=1)
            denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice_score = (2.0 * intersection + self.dice_smooth) / (denominator + self.dice_smooth)
        return 1.0 - dice_score.mean()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            probabilities = torch.sigmoid(predictions)
            squared_error = (probabilities - targets) ** 2
            if pixel_weights is not None:
                squared_error = squared_error * pixel_weights
            return squared_error.mean()

        loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        if pixel_weights is not None:
            loss = loss * pixel_weights
        bce_loss = loss.mean()
        if self.loss_type == "bce":
            return bce_loss
        dice_loss = self._compute_dice_loss(predictions, targets, pixel_weights)
        return bce_loss + self.dice_loss_weight * dice_loss


class SigmoidMSEMaskLoss(nn.Module):
    """Task 9 PDF loss: sigmoid mask predictions trained with MSE."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(predictions)
        return ((probabilities - targets) ** 2).mean()


__all__ = ["SigmoidMSEMaskLoss", "WeightedBinaryMaskLoss"]

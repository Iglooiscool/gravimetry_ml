"""Training settings shared by the stage-specific config modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageTrainingConfig:
    """Supported training options for one stage of the pipeline."""

    epochs: int
    batch_size: int
    learning_rate: float
    validation_frequency: int | None = None
    verbose: bool = True
    early_stopping_patience: int | None = None
    min_epochs: int | None = None
    min_improvement: float | None = None
    lr_drop_factor: float | None = None
    lr_drop_period: int | None = None
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    measurement_loss_weight: float = 0.0
    loss_type: str = "bce"
    dice_loss_weight: float = 0.0
    dice_smooth: float = 1.0


__all__ = ["StageTrainingConfig"]

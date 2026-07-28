"""Stage 1 model settings."""

from __future__ import annotations

from dataclasses import dataclass

from .training import StageTrainingConfig


@dataclass(frozen=True)
class Stage1ModelConfig:
    """Architecture and training settings for Stage 1."""

    hidden_layer_sizes: tuple[int, ...] = (128, 256, 128)
    dropout_rates: tuple[float, ...] = (0.2, 0.2, 0.2)
    training: StageTrainingConfig = StageTrainingConfig(
        epochs=200,
        batch_size=32,
        learning_rate=0.0005,
        validation_frequency=10,
        verbose=True,
        early_stopping_patience=10,
    )


__all__ = ["Stage1ModelConfig"]

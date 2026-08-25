"""Stage 2 model settings."""

from __future__ import annotations

from dataclasses import dataclass

from .training import StageTrainingConfig


@dataclass(frozen=True)
class Stage2ModelConfig:
    """Architecture and training settings for Stage 2."""

    hidden_layer_sizes: tuple[int, ...] = (512, 1024, 2048)
    dropout_rates: tuple[float, ...] = (0.2, 0.2, 0.2)
    model_type: str = "mlp"
    latent_grid_size: int = 8
    latent_channels: int = 64
    decoder_channels: tuple[int, ...] = (64, 32, 16)
    use_rectangle_edge_weighting: bool = False
    use_foreground_pos_weight: bool = True
    rectangle_edge_weight: float = 3.0
    rectangle_edge_width: int = 2
    edge_weight_mode: str = "rectangle"
    annulus_edge_weight: float = 1.0
    annulus_edge_width: int = 2
    training: StageTrainingConfig = StageTrainingConfig(
        epochs=150,
        batch_size=64,
        learning_rate=0.001,
        validation_frequency=30,
        verbose=True,
        early_stopping_patience=20,
        min_epochs=40,
        min_improvement=0.005,
        lr_drop_factor=0.5,
        lr_drop_period=50,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
        loss_type="bce",
        dice_loss_weight=0.0,
        dice_smooth=1.0,
    )
__all__ = ["Stage2ModelConfig"]

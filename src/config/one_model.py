"""Configuration for the official one-model workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .stage2 import Stage2ModelConfig
from .training import StageTrainingConfig


@dataclass(frozen=True)
class OneModelRunConfig:
    """Settings for direct gradient-to-mask reconstruction."""

    N: int = 8
    training_samples: int = 10_000
    validation_samples: int = 2_000
    test_samples: int = 500
    rho: float = 0.8
    grid_size: int = 32
    threshold: float = 0.5
    use_validation_threshold_sweep: bool = True
    threshold_candidates: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
    noise_sigma: float = 0.01
    noise_mode: str = "absolute"
    training_noise_replicas: int = 1
    seed: int = 42
    training_shape_weights: tuple[tuple[str, float], ...] | None = (
        ("rectangle", 0.25),
        ("two_circles", 0.45),
        ("annulus", 0.10),
        ("ellipse", 0.15),
        ("circle", 0.05),
    )
    model: Stage2ModelConfig = field(
        default_factory=lambda: Stage2ModelConfig(
            hidden_layer_sizes=(512, 1024, 2048),
            dropout_rates=(0.2, 0.2, 0.2),
            model_type="mlp",
            latent_grid_size=16,
            latent_channels=160,
            decoder_channels=(160, 128, 96, 64, 32),
            use_rectangle_edge_weighting=False,
            use_foreground_pos_weight=False,
            rectangle_edge_weight=4.0,
            rectangle_edge_width=3,
            edge_weight_mode="rectangle",
            annulus_edge_weight=1.0,
            annulus_edge_width=3,
            training=StageTrainingConfig(
                epochs=150,
                batch_size=64,
                learning_rate=0.001,
                validation_frequency=30,
                verbose=False,
                early_stopping_patience=20,
                min_epochs=40,
                min_improvement=0.001,
                lr_drop_factor=0.5,
                lr_drop_period=50,
                weight_decay=0.0001,
                gradient_clip_norm=1.0,
                loss_type="mse",
                dice_loss_weight=0.0,
                dice_smooth=1.0,
            ),
        )
    )
    output_dir: Path = Path("outputs/one_model")

    @property
    def num_measure_points(self) -> int:
        return self.N + 1

    @property
    def mask_pixels(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def coefficient_size(self) -> int:
        return 2 * (self.N + 1)

    @property
    def gradient_feature_size(self) -> int:
        return 2 * self.num_measure_points

    @property
    def run_output_dir(self) -> Path:
        return self.output_dir / f"N_{self.N}"


__all__ = ["OneModelRunConfig"]

"""Canonical settings for the three supported model-count workflows.

The active notebooks should use these factories instead of duplicating model
architectures and training settings inline. Change the values in this module
when the official experiment definition changes; the typed run configurations
remain responsible for validating and carrying those settings through the
workflow.
"""

from __future__ import annotations

from pathlib import Path

from .model_stack import TwoStageStackConfig
from .one_model import OneModelRunConfig
from .runs import TwoStageRunConfig
from .stage1 import Stage1ModelConfig
from .stage2 import Stage2ModelConfig
from .task9 import Task9RunConfig, Task9StackConfig
from .training import StageTrainingConfig

OFFICIAL_TRAIN_SIGMAS = (0.0, 0.001, 0.0025, 0.005, 0.01)
OFFICIAL_SHAPE_WEIGHTS = (
    ("rectangle", 0.25),
    ("two_circles", 0.45),
    ("annulus", 0.10),
    ("ellipse", 0.15),
    ("circle", 0.05),
)


def _sigma_label(sigma: float) -> str:
    """Return the repository directory label for one noise level."""

    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0.0:
        return "000"
    return f"{sigma:g}".replace("0.", "").replace(".", "")


def _official_stage1_config() -> Stage1ModelConfig:
    """Build the Stage 1 settings shared by the official two-stage model."""

    return Stage1ModelConfig(
        hidden_layer_sizes=(256, 512, 256),
        dropout_rates=(0.2, 0.2, 0.2),
        training=StageTrainingConfig(
            epochs=200,
            batch_size=64,
            learning_rate=0.0005,
            validation_frequency=20,
            verbose=False,
            early_stopping_patience=20,
            min_epochs=40,
            min_improvement=0.001,
            lr_drop_factor=0.5,
            lr_drop_period=60,
            weight_decay=0.0001,
            gradient_clip_norm=1.0,
            loss_type="mse",
        ),
    )


def _official_mask_config(edge_weight_mode: str = "all") -> Stage2ModelConfig:
    """Build the coordinate-convolution mask settings used by one/two-stage."""

    return Stage2ModelConfig(
        hidden_layer_sizes=(512, 1024),
        dropout_rates=(0.1, 0.1),
        model_type="coord_conv_decoder",
        latent_grid_size=16,
        latent_channels=160,
        decoder_channels=(160, 128, 96, 64, 32),
        use_rectangle_edge_weighting=True,
        use_foreground_pos_weight=False,
        rectangle_edge_weight=4.0,
        rectangle_edge_width=3,
        edge_weight_mode=edge_weight_mode,
        annulus_edge_weight=1.0,
        annulus_edge_width=3,
        training=StageTrainingConfig(
            epochs=170,
            batch_size=96,
            learning_rate=0.0005,
            validation_frequency=60,
            verbose=False,
            early_stopping_patience=25,
            min_epochs=50,
            min_improvement=0.001,
            lr_drop_factor=0.5,
            lr_drop_period=80,
            weight_decay=0.00025,
            gradient_clip_norm=0.8,
            loss_type="bce_dice",
            dice_loss_weight=1.0,
            dice_smooth=1.0,
        ),
    )


def official_one_stage_config(
    train_sigma: float,
    *,
    output_root: Path = Path("output"),
    coefficient_order: int = 10,
    training_samples: int = 20_000,
    validation_samples: int = 2_000,
    test_samples: int = 1_000,
    seed: int = 42,
) -> OneModelRunConfig:
    """Return the official direct gradient-to-mask run configuration."""

    output_dir = output_root / "stage_one" / "runs" / f"train_sigma{_sigma_label(train_sigma)}"
    return OneModelRunConfig(
        N=coefficient_order,
        training_samples=training_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        noise_sigma=train_sigma,
        noise_mode="absolute",
        seed=seed,
        training_noise_replicas=2,
        training_shape_weights=OFFICIAL_SHAPE_WEIGHTS,
        use_validation_threshold_sweep=True,
        model=_official_mask_config(),
        output_dir=output_dir,
    )


def official_two_stage_config(
    train_sigma: float,
    *,
    output_root: Path = Path("output"),
    coefficient_order: int = 10,
    training_samples: int = 20_000,
    validation_samples: int = 2_000,
    test_samples: int = 1_000,
    seed: int = 42,
) -> TwoStageRunConfig:
    """Return the official gradient-to-coefficients-to-mask configuration."""

    output_dir = output_root / "stage_two" / "runs" / f"train_sigma{_sigma_label(train_sigma)}"
    return TwoStageRunConfig(
        N=coefficient_order,
        training_samples=training_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        noise_sigma=train_sigma,
        noise_mode="absolute",
        seed=seed,
        training_noise_replicas=2,
        training_shape_weights=OFFICIAL_SHAPE_WEIGHTS,
        use_validation_threshold_sweep=True,
        stage2_predicted_coefficient_augmentation_copies=2,
        stage2_predicted_coefficient_noise_scale=0.5,
        stage2_include_gradient_features=False,
        model=TwoStageStackConfig(
            stage1=_official_stage1_config(),
            stage2=_official_mask_config(edge_weight_mode="rectangle"),
        ),
        output_dir=output_dir,
    )


def official_three_stage_config(
    train_sigma: float,
    *,
    output_root: Path = Path("output"),
    coefficient_order: int = 10,
    training_samples: int = 10_000,
    validation_samples: int = 2_000,
    test_samples: int = 1_000,
    seed: int = 42,
) -> Task9RunConfig:
    """Return the official predicted-router three-stage configuration."""

    output_dir = output_root / "stage_three" / "runs" / f"train_sigma{_sigma_label(train_sigma)}"
    return Task9RunConfig(
        N=coefficient_order,
        training_samples=training_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        noise_sigma=train_sigma,
        noise_mode="absolute",
        seed=seed,
        use_validation_threshold_sweep=True,
        model=Task9StackConfig(routing_mode="predicted_router"),
        output_dir=output_dir,
    )


__all__ = [
    "OFFICIAL_SHAPE_WEIGHTS",
    "OFFICIAL_TRAIN_SIGMAS",
    "official_one_stage_config",
    "official_three_stage_config",
    "official_two_stage_config",
]

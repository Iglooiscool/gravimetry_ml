"""Configuration for the experimental three-model workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .stage1 import Stage1ModelConfig
from .training import StageTrainingConfig


@dataclass(frozen=True)
class Task9GeneralMLPConfig:
    """Configuration for the general Task 9 mask MLP."""

    hidden_layer_sizes: tuple[int, ...] = (512, 1024, 2048)
    dropout_rates: tuple[float, ...] = (0.2, 0.2, 0.2)
    use_rectangle_edge_weighting: bool = False
    rectangle_edge_weight: float = 3.0
    rectangle_edge_width: int = 2
    edge_weight_mode: str = "rectangle"
    enable_rectangle_edge_augmentation: bool = True
    rectangle_augmentation_copies: int = 2
    training: StageTrainingConfig = StageTrainingConfig(
        epochs=150,
        batch_size=64,
        learning_rate=0.001,
        validation_frequency=30,
        verbose=True,
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
    )


@dataclass(frozen=True)
class Task9SpecialistMLPConfig:
    """Configuration for the Task 9 two-circle specialist MLP."""

    hidden_layer_sizes: tuple[int, ...] = (1024, 2048, 4096)
    dropout_rates: tuple[float, ...] = (0.3, 0.3, 0.3)
    training: StageTrainingConfig = StageTrainingConfig(
        epochs=200,
        batch_size=32,
        learning_rate=0.0005,
        validation_frequency=30,
        verbose=True,
        early_stopping_patience=None,
        min_epochs=None,
        min_improvement=None,
        lr_drop_factor=0.5,
        lr_drop_period=50,
        weight_decay=0.0001,
        gradient_clip_norm=None,
        loss_type="mse",
        dice_loss_weight=0.0,
        dice_smooth=1.0,
    )


@dataclass(frozen=True)
class Task9StackConfig:
    """Configuration for the optional Task 9 three-stage workflow."""

    stage1: Stage1ModelConfig = field(default_factory=Stage1ModelConfig)
    general: Task9GeneralMLPConfig = field(default_factory=Task9GeneralMLPConfig)
    specialist: Task9SpecialistMLPConfig = field(default_factory=Task9SpecialistMLPConfig)
    enable_specialist: bool = True
    specialist_min_n: int = 4
    specialist_shape_type: str = "two_circles"
    routing_mode: str = "true_shape_type"


@dataclass(frozen=True)
class Task9RunConfig:
    """Settings for one full Task 9 run."""

    N: int
    training_samples: int
    validation_samples: int
    test_samples: int = 500
    specialist_training_samples: int | None = None
    specialist_validation_samples: int | None = None
    rho: float = 0.8
    grid_size: int = 32
    threshold: float = 0.5
    use_validation_threshold_sweep: bool = False
    threshold_candidates: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    noise_sigma: float = 0.01
    seed: int = 42
    model: Task9StackConfig = field(default_factory=Task9StackConfig)
    output_dir: Path = Path("outputs/three_models")

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

    @property
    def effective_specialist_training_samples(self) -> int:
        return self.specialist_training_samples or self.training_samples

    @property
    def effective_specialist_validation_samples(self) -> int:
        return self.specialist_validation_samples or self.validation_samples

    @property
    def specialist_enabled(self) -> bool:
        return self.model.enable_specialist and self.N >= self.model.specialist_min_n


@dataclass(frozen=True)
class Task9SweepConfig:
    """Settings for a sweep of Task 9 runs across several N values."""

    n_values: tuple[int, ...] = (2, 4, 6, 8, 10)
    training_sizes: tuple[int, ...] = (2000, 4000, 6000, 8000, 10000)
    validation_sizes: tuple[int, ...] = (500, 800, 1200, 1600, 2000)
    test_samples: int = 500
    specialist_training_samples: int | None = None
    specialist_validation_samples: int | None = None
    rho: float = 0.8
    grid_size: int = 32
    threshold: float = 0.5
    use_validation_threshold_sweep: bool = False
    threshold_candidates: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    noise_sigma: float = 0.01
    seed: int = 42
    model: Task9StackConfig = field(default_factory=Task9StackConfig)
    output_dir: Path = Path("outputs/three_models")

    def validate(self) -> None:
        if not (len(self.n_values) == len(self.training_sizes) == len(self.validation_sizes)):
            raise ValueError("n_values, training_sizes, and validation_sizes must have the same length")

    def iter_runs(self) -> list[Task9RunConfig]:
        self.validate()
        run_configs: list[Task9RunConfig] = []
        for index, n_value in enumerate(self.n_values):
            run_configs.append(
                Task9RunConfig(
                    N=n_value,
                    training_samples=self.training_sizes[index],
                    validation_samples=self.validation_sizes[index],
                    test_samples=self.test_samples,
                    specialist_training_samples=self.specialist_training_samples,
                    specialist_validation_samples=self.specialist_validation_samples,
                    rho=self.rho,
                    grid_size=self.grid_size,
                    threshold=self.threshold,
                    use_validation_threshold_sweep=self.use_validation_threshold_sweep,
                    threshold_candidates=self.threshold_candidates,
                    noise_sigma=self.noise_sigma,
                    seed=self.seed + index,
                    model=self.model,
                    output_dir=self.output_dir,
                )
            )
        return run_configs


ThreeModelsConfig = Task9StackConfig
ThreeModelsRunConfig = Task9RunConfig
ThreeModelsSweepConfig = Task9SweepConfig


__all__ = [
    "Task9GeneralMLPConfig",
    "Task9SpecialistMLPConfig",
    "Task9StackConfig",
    "Task9RunConfig",
    "Task9SweepConfig",
    "ThreeModelsConfig",
    "ThreeModelsRunConfig",
    "ThreeModelsSweepConfig",
]

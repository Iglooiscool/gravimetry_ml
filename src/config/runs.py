"""Purpose: keep the run and sweep settings in one place.

This file holds the main experiment settings for one run, a sweep of runs, and
the small Task 2 export helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from measurements import GridSpec
from shapes import ShapeSamplingConfig

from .model_stack import TwoStageStackConfig


@dataclass(frozen=True)
class TwoStageRunConfig:
    """Settings for one full two-stage run."""

    N: int
    training_samples: int
    validation_samples: int
    test_samples: int = 500
    rho: float = 0.8
    grid_size: int = 32
    threshold: float = 0.5
    use_validation_threshold_sweep: bool = False
    threshold_candidates: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
    noise_level: float = 0.01
    seed: int = 42
    training_shape_weights: tuple[tuple[str, float], ...] | None = None
    model: TwoStageStackConfig = field(default_factory=TwoStageStackConfig)
    output_dir: Path = Path("outputs/two_stage")

    # Purpose:
    # Return the number of unit-circle measurement points for this N value.
    #
    # Inputs:
    # - none beyond the current run config
    #
    # Returns:
    # - The number of boundary measurement points, equal to N + 1
    @property
    def num_measure_points(self) -> int:
        return self.N + 1

    # Purpose:
    # Return the number of pixels in one flattened output mask.
    #
    # Inputs:
    # - none beyond the current run config
    #
    # Returns:
    # - The flattened mask length
    @property
    def mask_pixels(self) -> int:
        return self.grid_size * self.grid_size

    # Purpose:
    # Return the flattened coefficient feature length after the real/imag split.
    #
    # Inputs:
    # - none beyond the current run config
    #
    # Returns:
    # - The coefficient feature vector length
    @property
    def coefficient_size(self) -> int:
        return 2 * (self.N + 1)

    # Purpose:
    # Return the gradient feature length after real/imag split.
    #
    # Inputs:
    # - none beyond the current run config
    #
    # Returns:
    # - The gradient feature vector length
    @property
    def gradient_feature_size(self) -> int:
        return 2 * self.num_measure_points

    # Purpose:
    # Keep the old property name working while notebooks move to the clearer
    # gradient-based name.
    #
    # Inputs:
    # - none beyond the current run config
    #
    # Returns:
    # - The same value as gradient_feature_size
    @property
    def measurement_feature_size(self) -> int:
        return self.gradient_feature_size

    # Purpose:
    # Build the output folder path for this specific N run.
    #
    # Inputs:
    # - none beyond the current run config
    #
    # Returns:
    # - The N-specific output folder path
    @property
    def run_output_dir(self) -> Path:
        return self.output_dir / f"N_{self.N}"


@dataclass(frozen=True)
class TwoStageSweepConfig:
    """Settings for a sweep of two-stage runs across several N values."""

    n_values: tuple[int, ...] = (2, 4, 6, 8, 10)
    training_sizes: tuple[int, ...] = (2000, 4000, 6000, 8000, 10000)
    validation_sizes: tuple[int, ...] = (500, 800, 1200, 1600, 2000)
    test_samples: int = 500
    rho: float = 0.8
    grid_size: int = 32
    threshold: float = 0.5
    use_validation_threshold_sweep: bool = False
    threshold_candidates: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
    noise_level: float = 0.01
    seed: int = 42
    training_shape_weights: tuple[tuple[str, float], ...] | None = None
    model: TwoStageStackConfig = field(default_factory=TwoStageStackConfig)
    output_dir: Path = Path("outputs/two_stage")

    # Purpose:
    # Check that the sweep arrays all have matching lengths.
    #
    # Inputs:
    # - none beyond the current sweep config
    #
    # Returns:
    # - None. Raises ValueError if the arrays do not line up.
    def validate(self) -> None:
        if not (len(self.n_values) == len(self.training_sizes) == len(self.validation_sizes)):
            raise ValueError("n_values, training_sizes, and validation_sizes must have the same length")

    # Purpose:
    # Expand the sweep arrays into concrete per-N run configs.
    #
    # Inputs:
    # - none beyond the current sweep config
    #
    # Returns:
    # - A list of run configs, one for each N value
    def iter_runs(self) -> list[TwoStageRunConfig]:
        self.validate()
        run_configs: list[TwoStageRunConfig] = []
        for index, n_value in enumerate(self.n_values):
            run_configs.append(
                TwoStageRunConfig(
                    N=n_value,
                    training_samples=self.training_sizes[index],
                    validation_samples=self.validation_sizes[index],
                    test_samples=self.test_samples,
                    rho=self.rho,
                    grid_size=self.grid_size,
                    threshold=self.threshold,
                    use_validation_threshold_sweep=self.use_validation_threshold_sweep,
                    threshold_candidates=self.threshold_candidates,
                    noise_level=self.noise_level,
                    seed=self.seed + index,
                    training_shape_weights=self.training_shape_weights,
                    model=self.model,
                    output_dir=self.output_dir,
                )
            )
        return run_configs


@dataclass(frozen=True)
class Task2GenerateConfig:
    """Settings for the small Task 2 coefficient export helper."""

    output_dir: Path = Path("data/processed")
    samples_per_shape: int = 32
    shape_types: tuple[str, ...] = ("ellipse", "rectangle", "annulus", "two_circles")
    n_max: int = 5
    seed: int = 42
    grid: GridSpec = GridSpec()
    sampling: ShapeSamplingConfig = ShapeSamplingConfig()

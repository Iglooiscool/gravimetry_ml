"""Canonical PDF sweep settings shared by the three final notebooks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalSweepSpec:
    """Task 7 sweep settings used by every final model showcase."""

    n_values: tuple[int, ...] = (2, 4, 6, 8, 10)
    training_sizes: tuple[int, ...] = (2_000, 4_000, 6_000, 8_000, 10_000)
    validation_sizes: tuple[int, ...] = (500, 800, 1_200, 1_600, 2_000)
    test_samples: int = 500
    noise_sigma: float = 0.001
    noise_mode: str = "absolute"
    seed: int = 42

    def validate(self) -> None:
        if not (len(self.n_values) == len(self.training_sizes) == len(self.validation_sizes)):
            raise ValueError("Task 7 sweep arrays must have matching lengths")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")


FINAL_SWEEP = FinalSweepSpec()
FINAL_SWEEP.validate()

__all__ = ["FINAL_SWEEP", "FinalSweepSpec"]

"""Dataset container types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TwoStageDatasetSplit:
    """One dataset split with Stage 1 inputs, coefficients, masks, and names."""

    gradient_data: np.ndarray
    coefficients: np.ndarray
    masks: np.ndarray
    shape_types: tuple[str, ...]
    names: tuple[str, ...]


@dataclass
class TwoStageDatasetBundle:
    """The full train/validation/test/fixed dataset bundle for one run."""

    train: TwoStageDatasetSplit
    validation: TwoStageDatasetSplit
    test: TwoStageDatasetSplit
    fixed: TwoStageDatasetSplit
    measurement_points: np.ndarray
    measurement_matrix: np.ndarray
    matrix_condition_number: float

    def splits(self) -> dict[str, TwoStageDatasetSplit]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "fixed": self.fixed,
        }


__all__ = ["TwoStageDatasetSplit", "TwoStageDatasetBundle"]

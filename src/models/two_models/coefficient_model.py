"""The first model in the two-model system."""

from ..stage1.model import Stage1Regressor


class GradientToCoefficientModel(Stage1Regressor):
    """Predict shape coefficients from noisy gradient features."""


__all__ = ["GradientToCoefficientModel"]

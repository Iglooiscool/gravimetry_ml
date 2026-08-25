"""The general mask model in the three-model system."""

from ..task9.general_mlp import Task9GeneralMaskMLP


class CoefficientToGeneralMaskModel(Task9GeneralMaskMLP):
    """Predict masks from coefficients for all supported shape types."""


__all__ = ["CoefficientToGeneralMaskModel"]

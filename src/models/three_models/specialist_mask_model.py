"""The specialist mask model in the three-model system."""

from ..task9.specialist_mlp import Task9TwoCircleSpecialistMLP


class CoefficientToSpecialistMaskModel(Task9TwoCircleSpecialistMLP):
    """Predict masks from coefficients for two-circle examples."""


__all__ = ["CoefficientToSpecialistMaskModel"]

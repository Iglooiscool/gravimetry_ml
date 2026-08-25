"""The two-model coefficient-to-mask reference implementation."""

from .coefficient_model import GradientToCoefficientModel
from .mask_model import CoefficientToMaskModel
from .system import TwoModelSystem
from .training import fit_coefficient_model, fit_mask_model, predict_coefficients, predict_mask_logits

__all__ = [
    "GradientToCoefficientModel",
    "CoefficientToMaskModel",
    "TwoModelSystem",
    "fit_coefficient_model",
    "fit_mask_model",
    "predict_coefficients",
    "predict_mask_logits",
]

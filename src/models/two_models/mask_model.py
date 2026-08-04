"""The second model in the two-model system."""

from ..stage2.model import Stage2CoordConvDecoder


class CoefficientToMaskModel(Stage2CoordConvDecoder):
    """Predict mask logits from predicted shape coefficients."""


__all__ = ["CoefficientToMaskModel"]

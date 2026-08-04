"""Stage 2 model and training API."""

from .losses import WeightedBinaryMaskLoss
from .model import Stage2ConvDecoder, Stage2CoordConvDecoder, Stage2MaskPredictor
from .train import fit_stage2_model, predict_stage2_logits, stop_if_safe
from .weights import compute_rectangle_edge_pixel_weights, compute_shape_edge_pixel_weights

__all__ = [
    "Stage2ConvDecoder",
    "Stage2CoordConvDecoder",
    "Stage2MaskPredictor",
    "WeightedBinaryMaskLoss",
    "compute_rectangle_edge_pixel_weights",
    "compute_shape_edge_pixel_weights",
    "fit_stage2_model",
    "predict_stage2_logits",
    "stop_if_safe",
]

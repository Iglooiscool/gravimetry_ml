"""Purpose: expose the public model, metric, and training API."""

from .metrics import binary_iou, evaluate_regression_predictions, evaluate_stage2_predictions, evaluate_stage2_predictions_by_shape, mean_absolute_error, mean_squared_error, select_best_stage2_threshold
from .common_train import ModelTrainingResult, compute_input_normalization, compute_target_normalization, denormalize_values, normalize_values, predict_tensor, set_torch_seed
from .stage1 import Stage1Regressor, fit_stage1_model, predict_stage1_coefficients, stop_if_overfitting
from .stage2 import Stage2ConvDecoder, Stage2CoordConvDecoder, Stage2MaskPredictor, compute_rectangle_edge_pixel_weights, compute_shape_edge_pixel_weights, fit_stage2_model, predict_stage2_logits, stop_if_safe

__all__ = [
    "Stage1Regressor",
    "Stage2ConvDecoder",
    "Stage2CoordConvDecoder",
    "Stage2MaskPredictor",
    "ModelTrainingResult",
    "compute_rectangle_edge_pixel_weights",
    "compute_shape_edge_pixel_weights",
    "compute_input_normalization",
    "compute_target_normalization",
    "normalize_values",
    "denormalize_values",
    "fit_stage1_model",
    "fit_stage2_model",
    "predict_tensor",
    "predict_stage1_coefficients",
    "predict_stage2_logits",
    "set_torch_seed",
    "stop_if_safe",
    "stop_if_overfitting",
    "binary_iou",
    "evaluate_regression_predictions",
    "evaluate_stage2_predictions",
    "evaluate_stage2_predictions_by_shape",
    "mean_squared_error",
    "mean_absolute_error",
    "select_best_stage2_threshold",
]

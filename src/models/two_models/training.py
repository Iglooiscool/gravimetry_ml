"""Named training helpers for the two-model system."""

from ..stage1.train import fit_stage1_model, predict_stage1_coefficients
from ..stage2.train import fit_stage2_model, predict_stage2_logits

fit_coefficient_model = fit_stage1_model
fit_mask_model = fit_stage2_model
predict_coefficients = predict_stage1_coefficients
predict_mask_logits = predict_stage2_logits

__all__ = ["fit_coefficient_model", "fit_mask_model", "predict_coefficients", "predict_mask_logits"]

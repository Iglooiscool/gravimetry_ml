"""Named training helpers for the three-model system."""

from ..stage1.train import fit_stage1_model, predict_stage1_coefficients
from ..task9.train import train_task9_head

fit_coefficient_model = fit_stage1_model
fit_mask_model = train_task9_head
predict_coefficients = predict_stage1_coefficients

__all__ = ["fit_coefficient_model", "fit_mask_model", "predict_coefficients"]

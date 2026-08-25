"""Combined Task 9 model routing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import nn

from ..common_train import ModelTrainingResult
from ..stage2.train import predict_stage2_logits
from .router import predict_task9_router


@dataclass
class Task9TrainedHead:
    """A trained Task 9 head plus its normalization state."""

    name: str
    model: nn.Module
    training_result: ModelTrainingResult


@dataclass
class Task9CombinedModel:
    """General Task 9 model plus an optional specialist override."""

    general: Task9TrainedHead
    specialist: Task9TrainedHead | None = None
    specialist_shape_type: str = "two_circles"
    routing_mode: str = "true_shape_type"
    router: Task9TrainedHead | None = None


def specialist_indices_for_shape_types(
    shape_types: tuple[str, ...] | None,
    specialist_shape_type: str,
    routing_mode: str,
) -> np.ndarray:
    """Return the indices that should use the specialist model."""

    if shape_types is None or routing_mode != "true_shape_type":
        return np.array([], dtype=int)
    return np.array([index for index, shape_type in enumerate(shape_types) if shape_type == specialist_shape_type], dtype=int)


def combine_task9_logits(
    general_logits: np.ndarray,
    specialist_logits: np.ndarray | None,
    shape_types: tuple[str, ...] | None,
    specialist_shape_type: str,
    routing_mode: str,
    specialist_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Overlay specialist predictions onto the general predictions."""

    combined_logits = general_logits.copy()
    if specialist_indices is None:
        specialist_indices = specialist_indices_for_shape_types(shape_types, specialist_shape_type, routing_mode)
    if specialist_logits is None or specialist_indices.size == 0:
        return combined_logits, 0
    combined_logits[specialist_indices] = specialist_logits
    return combined_logits, int(specialist_indices.size)


def predict_task9_combined_logits(
    combined_model: Task9CombinedModel,
    features: np.ndarray,
    shape_types: tuple[str, ...] | None,
    device,
) -> tuple[np.ndarray, int]:
    """Run the general head and optionally replace routed samples with specialist logits."""

    general_logits = predict_stage2_logits(
        combined_model.general.model,
        features,
        device=device,
        training_result=combined_model.general.training_result,
    )
    specialist_indices = specialist_indices_for_shape_types(
        shape_types, combined_model.specialist_shape_type, combined_model.routing_mode
    )
    if combined_model.routing_mode == "predicted_router" and combined_model.router is not None:
        router_logits = predict_task9_router(
            combined_model.router.model,
            features,
            training_result=combined_model.router.training_result,
            device=device,
        )
        specialist_indices = np.flatnonzero(router_logits >= 0.0)
    specialist_logits = None
    if combined_model.specialist is not None and specialist_indices.size > 0:
        specialist_logits = predict_stage2_logits(
            combined_model.specialist.model,
            features[specialist_indices],
            device=device,
            training_result=combined_model.specialist.training_result,
        )
    return combine_task9_logits(
        general_logits=general_logits,
        specialist_logits=specialist_logits,
        shape_types=shape_types,
        specialist_shape_type=combined_model.specialist_shape_type,
        routing_mode=combined_model.routing_mode,
        specialist_indices=specialist_indices,
    )


__all__ = [
    "Task9TrainedHead",
    "Task9CombinedModel",
    "specialist_indices_for_shape_types",
    "combine_task9_logits",
    "predict_task9_combined_logits",
]

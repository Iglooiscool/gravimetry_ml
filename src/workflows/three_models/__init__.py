"""Experimental three-model workflow entry points.

The implementation lives in :mod:`workflows.task9.run`; these functions make
the three-stage showcase API explicit and document the predictor variant used
for test-noise evaluation.
"""

from __future__ import annotations

import torch

from workflows.task9 import run_task9_once, run_task9_once_with_predictor, run_task9_sweep


def run_three_models(run_config, device: torch.device | None = None) -> dict[str, object]:
    """Train and evaluate one experimental three-stage configuration."""

    return run_task9_once(run_config, device=device)


def run_three_models_with_predictor(run_config, device: torch.device | None = None):
    """Train the three-stage workflow and return its live prediction callable."""

    return run_task9_once_with_predictor(run_config, device=device)


def run_three_models_sweep(sweep_config, device: torch.device | None = None) -> dict[str, object]:
    """Run the experimental three-stage workflow over a sweep configuration."""

    return run_task9_sweep(sweep_config, device=device)

__all__ = ["run_three_models", "run_three_models_with_predictor", "run_three_models_sweep"]

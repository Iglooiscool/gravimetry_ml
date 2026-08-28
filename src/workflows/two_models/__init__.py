"""Reference two-model workflow entry points.

The implementation lives in :mod:`pipeline.run`; these functions provide the
named workflow API used by notebooks without maintaining a second runner.
"""

from __future__ import annotations

import torch

from pipeline.run import run_two_stage_once, run_two_stage_sweep


def run_two_models(run_config, device: torch.device | None = None) -> dict[str, object]:
    """Train and evaluate one connected two-model configuration."""

    return run_two_stage_once(run_config, device=device)


def run_two_models_sweep(sweep_config, device: torch.device | None = None) -> dict[str, object]:
    """Run the connected two-model workflow over a sweep configuration."""

    return run_two_stage_sweep(sweep_config, device=device)

__all__ = ["run_two_models", "run_two_models_sweep"]

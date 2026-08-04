"""Named entry points for the experimental three-model workflow."""

from workflows.task9 import run_task9_once, run_task9_sweep

run_three_models = run_task9_once
run_three_models_sweep = run_task9_sweep

__all__ = ["run_three_models", "run_three_models_sweep"]

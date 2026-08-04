"""Named entry points for the reference two-model workflow."""

from pipeline.run import run_two_stage_once, run_two_stage_sweep

run_two_models = run_two_stage_once
run_two_models_sweep = run_two_stage_sweep

__all__ = ["run_two_models", "run_two_models_sweep"]

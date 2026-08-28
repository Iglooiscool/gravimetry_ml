"""Purpose: provide a simple command-line entry point for running the model.

This file lets you run one experiment or a full sweep without going through a
notebook, using the config objects defined in src/config.
"""

from __future__ import annotations

import argparse
import json

from config import Task9SweepConfig, TwoStageSweepConfig
from config.official import (
    official_one_stage_config,
    official_three_stage_config,
    official_two_stage_config,
)
from workflows.one_model import run_one_model
from workflows.two_models import run_two_models, run_two_models_sweep
from workflows.three_models import run_three_models, run_three_models_sweep


# Purpose:
# Parse command-line arguments for the simple run/sweep entry point.
#
# Inputs:
# - none
#
# Returns:
# - An argparse Namespace containing the chosen mode and settings
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a gravimetry model-count workflow")
    parser.add_argument("mode", choices=("run", "sweep"), help="Run one experiment or a full sweep")
    parser.add_argument("--model-count", type=int, choices=(1, 2, 3), default=1, help="Number of neural models to use")
    parser.add_argument("--coefficient-order", "--N", dest="coefficient_order", type=int, default=10, help="Coefficient order for single-run mode")
    parser.add_argument("--train-samples", type=int, default=4000, help="Training sample count for single-run mode")
    parser.add_argument("--validation-samples", type=int, default=1000, help="Validation sample count for single-run mode")
    parser.add_argument("--noise-sigma", type=float, default=0.0, help="Training gradient noise sigma for single-run mode")
    return parser.parse_args()


# Purpose:
# Execute the selected mode and print the resulting summary.
#
# Inputs:
# - none. Reads command-line arguments from the shell.
#
# Returns:
# - None. Prints the summary JSON to stdout.
def main() -> None:
    args = parse_args()
    if args.model_count == 1:
        if args.mode == "run":
            summary = run_one_model(
                official_one_stage_config(
                    args.noise_sigma,
                    coefficient_order=args.coefficient_order,
                    training_samples=args.train_samples,
                    validation_samples=args.validation_samples,
                )
            )
        else:
            raise ValueError("One-model sweeps are not currently supported")
    elif args.model_count == 2:
        if args.mode == "run":
            summary = run_two_models(
                official_two_stage_config(
                    args.noise_sigma,
                    coefficient_order=args.coefficient_order,
                    training_samples=args.train_samples,
                    validation_samples=args.validation_samples,
                )
            )
        else:
            summary = run_two_models_sweep(TwoStageSweepConfig())
    else:
        if args.mode == "run":
            summary = run_three_models(
                official_three_stage_config(
                    args.noise_sigma,
                    coefficient_order=args.coefficient_order,
                    training_samples=args.train_samples,
                    validation_samples=args.validation_samples,
                )
            )
        else:
            summary = run_three_models_sweep(Task9SweepConfig())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

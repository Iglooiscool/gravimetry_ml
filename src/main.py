"""Purpose: provide a simple command-line entry point for running the model.

This file lets you run one experiment or a full sweep without going through a
notebook, using the config objects defined in src/config.
"""

from __future__ import annotations

import argparse
import json

from config import Task9RunConfig, Task9SweepConfig, TwoStageRunConfig, TwoStageSweepConfig
from pipeline import run_two_stage_once, run_two_stage_sweep
from workflows.task9 import run_task9_once, run_task9_sweep


# Purpose:
# Parse command-line arguments for the simple run/sweep entry point.
#
# Inputs:
# - none
#
# Returns:
# - An argparse Namespace containing the chosen mode and settings
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gravimetry workflows")
    parser.add_argument("mode", choices=("run", "sweep"), help="Run one experiment or a full sweep")
    parser.add_argument("--workflow", choices=("two_stage", "task9"), default="two_stage", help="Workflow to execute")
    parser.add_argument("--N", type=int, default=8, help="Coefficient order for single-run mode")
    parser.add_argument("--train-samples", type=int, default=4000, help="Training sample count for single-run mode")
    parser.add_argument("--validation-samples", type=int, default=1000, help="Validation sample count for single-run mode")
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
    if args.workflow == "two_stage":
        if args.mode == "run":
            summary = run_two_stage_once(
                TwoStageRunConfig(
                    N=args.N,
                    training_samples=args.train_samples,
                    validation_samples=args.validation_samples,
                )
            )
        else:
            summary = run_two_stage_sweep(TwoStageSweepConfig())
    else:
        if args.mode == "run":
            summary = run_task9_once(
                Task9RunConfig(
                    N=args.N,
                    training_samples=args.train_samples,
                    validation_samples=args.validation_samples,
                )
            )
        else:
            summary = run_task9_sweep(Task9SweepConfig())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

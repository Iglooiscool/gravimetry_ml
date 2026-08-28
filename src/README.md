# Source Layout

## `data_pipeline`

The final experiment entry point for dataset construction. `datasets.builder`
contains the implementation: clean shapes become coefficients and gradients,
then Gaussian noise is applied only while constructing training gradients.

## `workflows`

The active notebooks call the workflow entry points directly:

- `workflows.one_model`: direct gradient-to-mask training and evaluation.
- `workflows.two_models`: connected gradient-to-coefficient-to-mask workflow.
- `workflows.three_models`: experimental Task 8 plus Task 9 workflow.

These modules coordinate dataset construction, training, evaluation, and artifact
writing. They are the implementation entry points; there is no duplicate
`final_models` facade.

## `config/official.py`

This module is the single source of truth for the supported one-, two-, and
three-model experiment presets. Use `official_one_stage_config`,
`official_two_stage_config`, and `official_three_stage_config` when creating a
run outside the notebooks.

The `experiments` package contains shared notebook utilities for checkpoint
loading, model introspection, test-time noise sweeps, and CSV/JSON artifact
export. It does not alter the official training-only noise protocol.

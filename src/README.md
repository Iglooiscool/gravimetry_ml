# Source Layout

## `data_pipeline`

The final experiment entry point for dataset construction. `datasets.builder`
contains the implementation: clean shapes become coefficients and gradients,
then Gaussian noise is applied only while constructing training gradients.

## `final_models`

Stable public entry points for the three final showcases:

- `final_models.one_stage`: direct gradient-to-mask model.
- `final_models.two_stage`: connected gradient-to-coefficient-to-mask model.
- `final_models.three_stage`: Task 8 plus Task 9 general and specialist stages.

## Existing Modules

The existing `models`, `workflows`, `measurements`, and `shapes` packages remain
the tested implementation layer while the final notebooks use the explicit
`final_models` interface.

The `experiments` package contains shared notebook utilities for checkpoint
loading, model introspection, test-time noise sweeps, and CSV/JSON artifact
export. It does not alter the official training-only noise protocol.

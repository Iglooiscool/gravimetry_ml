# Active Experiment Notebooks

Exactly five notebooks are maintained here. Historical educational, scratch,
and diagnostic notebooks are preserved under `notebooks_archive/`.

- `00_project_notes_and_methodology.ipynb`: problem, data, noise, splits, and metrics.
- `01_one_stage_pipeline.ipynb`: selected direct gradient-to-mask pipeline.
- `02_two_stage_pipeline.ipynb`: coefficient prediction followed by mask reconstruction.
- `03_three_stage_pipeline.ipynb`: experimental general/specialist workflow and routing limitation.
- `04_one_stage_annulus_router.ipynb`: standalone learned annulus-specialist one-stage candidate.

The previous reporting and sampling ablation notebooks remain under
`notebooks_archive/`.

All active notebooks use clean validation/test data for the official metric.
Their supplementary robustness curves evaluate a fixed model at test noise
levels `0.0`, `0.001`, `0.0025`, `0.005`, and `0.01`.

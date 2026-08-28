# minimal-gravimetry-ml

This project reconstructs simple binary shapes from a small number of noisy
boundary measurements.

## Model Versions

The reusable code is organized by the number of neural models in the system.

### One model

The selected one-stage model maps gradient features directly to mask logits with
a coordinate-aware decoder:

```text
gradient features -> mask
```

Its public implementation is in `src/models/one_model/`. A flat MLP remains
available as a baseline in the experimental notebook.

### Two models

The reference two-model system separates coefficient prediction from mask
reconstruction:

```text
gradient features -> coefficients -> mask
```

Its public implementation is in `src/models/two_models/`.

### Three models

The supported three-model system contains a coefficient model, a general mask
model, and a two-circle specialist mask model:

```text
gradient features -> coefficients -> general or specialist mask
```

Its public implementation is in `src/models/three_models/`. It uses the
predicted coefficient router; oracle shape routing is retained only for
diagnostics and tests.

## Repository Layout

```text
minimal-gravimetry-ml/
  notebooks/              Educational and reference notebooks
  src/
    config/               Shared and official model-count configuration
    datasets/             Synthetic dataset generation and persistence
    measurements/         Measurement and coefficient calculations
    models/
      shared/             Reusable training, loss, and normalization logic
      one_model/          Official direct gradient-to-mask model
      two_models/         Coefficient model plus mask model
      three_models/       General and specialist mask models
      stage1/             Low-level compatibility component
      stage2/             Low-level decoder and mask-training component
      task9/              Low-level compatibility component for three models
    workflows/
      one_model/          Official workflow
      three_models/       Official three-model workflow
      task9/              Three-model dataset, routing, and artifact helpers
    main.py               Command-line entry point
  tests/                  Automated checks
```

## Running Models

Run the official one-model workflow:

```bash
python src/main.py run --model-count 1 --coefficient-order 8 --train-samples 10000 --validation-samples 2000
```

Run the two-model reference workflow:

```bash
python src/main.py run --model-count 2 --coefficient-order 8 --train-samples 4000 --validation-samples 1000
```

Run the experimental three-model workflow:

```bash
python src/main.py run --model-count 3 --coefficient-order 8 --train-samples 4000 --validation-samples 1000
```

Historical artifacts remain under `outputs/`. The active notebooks write raw
runs under `output/stage_one/runs/`, `output/stage_two/runs/`, and
`output/stage_three/runs/`. Grouped research artifacts are written beside those
run folders, and curated copies are published under `expected_output/`.

Official run artifacts are organized by training-noise condition and coefficient order:

```text
output/stage_one/runs/train_sigma001/N_10/summary.json
output/stage_two/runs/train_sigma001/N_10/fixed_reconstructions.png
```

The active model notebooks compare training sigma values `0.0`, `0.001`,
`0.0025`, `0.005`, and `0.01`. Each `N_10` folder contains the official run
summary, clean fixed-benchmark results, saved datasets, and model weights.
Test-noise robustness sweeps are not part of the official outputs.

Official architecture and training presets are centralized in
`src/config/official.py`. The active notebooks and CLI use these factories so
that changing an official model setting does not require editing multiple
notebooks.

## Active Notebooks

The active notebook set is intentionally limited to four documented workflows:

- `00_project_notes_and_methodology.ipynb`
- `01_one_stage_pipeline.ipynb`
- `02_two_stage_pipeline.ipynb`
- `03_three_stage_pipeline.ipynb`
Earlier educational and scratch notebooks are preserved under
`notebooks_archive/`.

The annulus-router candidate and historical experiments are preserved under
`notebooks_archive/`.

## Development

The shared synthetic-data noise model follows Tasks 5, 7, and 8: Gaussian
noise is added independently to the real and imaginary parts of the clean
gradient, with `noise_sigma` as the absolute standard deviation. Noise is not
scaled by the signal magnitude.

Install development dependencies, then run:

```bash
pytest
ruff check
python -m compileall src
```

The tests add `src/` to `sys.path`, so the source packages can be imported
directly during local development.

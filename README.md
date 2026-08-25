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

The experimental three-model system contains a coefficient model, a general
mask model, and a two-circle specialist mask model:

```text
gradient features -> coefficients -> general or specialist mask
```

Its public implementation is in `src/models/three_models/`. Specialist routing
is experimental and is not the official model path.

## Repository Layout

```text
minimal-gravimetry-ml/
  notebooks/              Educational and reference notebooks
  src/
    config/               Model-count and training configuration
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
      three_models/       Experimental three-model workflow
      task9/              Existing implementation used by three_models
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

Historical artifacts remain under `outputs/`. The five active notebooks write
new, traceable artifacts under `output/`.

Noise-sweep artifacts are organized by training and test noise:

```text
output/one_stage/train_sigma001/test_sigma005/metrics.csv
output/two_stage/train_sigma001/test_sigma005/metrics_by_shape.csv
```

The active model notebooks compare training sigma values `0.0`, `0.001`,
`0.0025`, `0.005`, and `0.01`, then evaluate each trained model at the same
five test sigma values.

## Active Notebooks

The active notebook set is intentionally limited to five documented workflows:

- `00_project_notes_and_methodology.ipynb`
- `01_one_stage_pipeline.ipynb`
- `02_two_stage_pipeline.ipynb`
- `03_three_stage_pipeline.ipynb`
- `04_one_stage_annulus_router.ipynb`

Earlier educational and scratch notebooks are preserved under
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

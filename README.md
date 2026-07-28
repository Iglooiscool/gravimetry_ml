# minimal-gravimetry-ml

This project reconstructs simple binary shapes from a small number of noisy boundary measurements.

The main idea is a two-stage PyTorch pipeline:

1. Stage 1 takes gradient-based boundary data and predicts shape coefficients.
2. Stage 2 takes those coefficients and predicts a binary mask of the shape.

## What is in this repo

- `docs/tasks/` holds the task PDFs that define the project steps.
- `notebooks/` holds task notebooks that explain the ideas and show experiments.
- `src/` holds the real reusable code.
- `tests/` holds automated checks for the reusable code.

## How the code is organized

The real code lives in `src/`.

### Main entry points

- `src/main.py`
  Runs the project from the command line.

- `src/pipeline.py`
  Runs one experiment or a full sweep in Python code.

### Config

- `src/config/models.py`
  Model layout settings such as hidden layer sizes and dropout.

- `src/config/runs.py`
  Run settings such as sample counts, epochs, learning rate, and output folders.

### Shapes

- `src/shapes/base.py`
  Base shape type shared by all shapes.

- `src/shapes/shapes.py`
  Concrete shapes, fixed benchmark shapes, and random shape sampling.

### Measurements and math

- `src/measurements.py`
  Grid creation, coefficient computation, unit-circle measurement points, gradient data, measurement matrix, and Gaussian noise.

### Models

- `src/models/two_stage.py`
  Stage 1 and Stage 2 network definitions.

- `src/models/training.py`
  Training loops, normalization helpers, early stopping, and prediction helpers.

- `src/models/metrics.py`
  MSE, MAE, IoU, and evaluation helpers.

### Data and outputs

- `src/datasets.py`
  Synthetic dataset generation for train, validation, test, and fixed benchmark splits.

- `src/plotting.py`
  Plots and JSON summary writers.

## Repository layout

```text
minimal-gravimetry-ml/
  docs/
  notebooks/
  outputs/
  src/
    __init__.py
    main.py
    measurements.py
    datasets.py
    pipeline.py
    plotting.py
    config/
      __init__.py
      models.py
      runs.py
    shapes/
      __init__.py
      base.py
      shapes.py
    models/
      __init__.py
      two_stage.py
      training.py
      metrics.py
  tests/
  .dockerignore
  .gitignore
  docker-compose.yml
  Dockerfile
  pyproject.toml
```

## Running the project

Run one experiment from the command line:

```bash
python src/main.py run --N 8 --train-samples 4000 --validation-samples 1000
```

Run a sweep:

```bash
python src/main.py sweep
```

If you want to work through the project step by step, open the notebooks in `notebooks/` instead.

## Docker-only setup

Build image:

```bash
docker compose build
```

Start Jupyter Lab:

```bash
docker compose up
```

Open in browser:

```text
http://localhost:8888
```

Stop services:

```bash
docker compose down
```

Notes:

- The container runs Jupyter Lab with no token/password for local development convenience.
- Keep this setup for local machine use only.

## Development note

- Tests add `src/` to `sys.path`, so the source files and folders can be imported directly.
- Task 8 now defines the real Stage 1 setup used by the reusable code: gradient data goes into the DNN, not the older measurement-only path.

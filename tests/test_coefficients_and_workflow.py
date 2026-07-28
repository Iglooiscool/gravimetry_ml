import numpy as np

from config import Task2GenerateConfig
from pipeline import generate_task2_dataset
from measurements import GridSpec, compute_coefficients, create_grid
from shapes import EllipseSpec


def test_coefficients_shape_and_finite_values(tmp_path):
    grid = create_grid(GridSpec(grid_size=120))
    shape = EllipseSpec(a=0.6, b=0.3, theta=np.pi / 4, center=(0.1, 0.1))
    mask = shape.compute_mask(grid.X, grid.Y)
    result = compute_coefficients(mask, grid.X, grid.Y, grid.dA, n_max=5)

    assert result.coefficients.shape == (6,)
    assert np.isfinite(result.real).all()
    assert np.isfinite(result.imag).all()
    assert np.isfinite(result.magnitudes).all()


def test_task2_dataset_workflow_writes_outputs(tmp_path):
    cfg = Task2GenerateConfig(output_dir=tmp_path, samples_per_shape=2, n_max=3, seed=7)
    paths = generate_task2_dataset(cfg)

    assert paths["metadata"].exists()
    assert paths["summary"].exists()

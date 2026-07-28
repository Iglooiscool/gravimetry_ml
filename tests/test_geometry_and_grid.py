import numpy as np

from measurements import GridSpec, create_grid
from shapes import AnnulusSpec, CircleSpec, EllipseSpec, RectangleSpec, TwoCirclesSpec


def test_create_grid_shapes_and_spacing():
    grid = create_grid(GridSpec(grid_size=100))
    assert grid.X.shape == (100, 100)
    assert grid.Y.shape == (100, 100)
    assert grid.dx > 0
    assert grid.dA > 0


def test_ellipse_mask_contains_points():
    grid = create_grid(GridSpec(grid_size=100))
    shape = EllipseSpec(a=0.6, b=0.3, theta=np.pi / 4, center=(0.1, 0.1))
    mask = shape.compute_mask(grid.X, grid.Y)
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


def test_other_shapes_compute_masks():
    grid = create_grid(GridSpec(grid_size=80))
    shapes = [
        RectangleSpec(width=0.8, height=0.4, theta=np.pi / 6, center=(0.0, 0.0)),
        CircleSpec(radius=0.4, center=(0.0, 0.0)),
        AnnulusSpec(outer_radius=0.7, inner_radius=0.3, center=(0.0, 0.0)),
        TwoCirclesSpec(radius1=0.3, center1=(-0.4, 0.3), radius2=0.25, center2=(0.4, -0.2)),
    ]
    for shape in shapes:
        mask = shape.compute_mask(grid.X, grid.Y)
        assert mask.dtype == np.bool_
        assert mask.sum() > 0

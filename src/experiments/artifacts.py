"""Organize completed model runs into readable research artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

from .evaluation import sigma_label


def _copy(source: Path, destination: Path) -> None:
    """Copy one file when it exists, creating its parent directory."""

    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def publish_run_artifacts(
    stage_name: str,
    run_output_dir: Path,
    training_sigma: float,
    *,
    grouped_root: Path,
) -> Path:
    """Copy one completed run into the grouped research artifact layout."""

    if not stage_name:
        raise ValueError("stage_name must not be empty")
    label = sigma_label(training_sigma)
    stage_root = grouped_root / stage_name
    for source_name, destination_name in (
        ("fixed_reconstructions.png", f"fixed_reconstruction_sigma_{label}.png"),
        ("fixed_shapes.png", f"fixed_shapes_sigma_{label}.png"),
        ("measurement_points.png", f"measurement_points_sigma_{label}.png"),
    ):
        _copy(run_output_dir / source_name, stage_root / "fixed_reconstructions" / destination_name)

    _copy(run_output_dir / "summary.json", stage_root / "summary_json" / f"summary_sigma_{label}.json")

    for source in sorted(run_output_dir.glob("*.pt")):
        _copy(source, stage_root / "models" / f"{source.stem}_sigma_{label}{source.suffix}")

    return stage_root


def publish_expected_output(stage_name: str, *, output_root: Path = Path("output")) -> Path:
    """Copy grouped artifacts into the presentation-facing expected-output folder."""

    source = output_root / stage_name
    destination = output_root.parent / "expected_output" / stage_name
    if not source.exists():
        raise FileNotFoundError(f"Grouped output does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "runs" or child.is_file():
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
    return destination


__all__ = ["publish_expected_output", "publish_run_artifacts"]

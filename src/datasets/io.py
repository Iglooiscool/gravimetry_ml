"""Dataset persistence helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import TwoStageDatasetBundle


def save_two_stage_dataset(dataset_bundle: TwoStageDatasetBundle, output_dir: Path) -> dict[str, Path]:
    """Save dataset arrays in a simple ``.npz`` format."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for split_name, dataset_split in dataset_bundle.splits().items():
        split_path = output_dir / f"{split_name}.npz"
        np.savez_compressed(
            split_path,
            gradient_data=dataset_split.gradient_data,
            coefficients=dataset_split.coefficients,
            masks=dataset_split.masks,
            shape_types=np.array(dataset_split.shape_types),
            names=np.array(dataset_split.names),
        )
        output_paths[split_name] = split_path
    return output_paths


__all__ = ["save_two_stage_dataset"]

"""Small plotting and path helpers shared by active notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by locating ``pyproject.toml``."""

    root = (start or Path.cwd()).resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    if not (root / "pyproject.toml").exists():
        raise RuntimeError("Could not locate project root")
    return root


def plot_noise_curve(results: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save a clearly labelled overall IoU versus test-noise plot."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(results["noise_sigma"], results["mean_iou"], marker="o", linewidth=2, label="Test IoU")
    axis.set_title(title)
    axis.set_xlabel("Test gradient noise sigma")
    axis.set_ylabel("Mean IoU")
    axis.set_xticks(results["noise_sigma"].tolist())
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.show()


def plot_shape_curves(results: pd.DataFrame, output_path: Path, title: str) -> None:
    """Save per-shape IoU curves for a noise sweep."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    for shape, group in results.groupby("shape"):
        axis.plot(group["noise_sigma"], group["mean_iou"], marker="o", label=shape)
    axis.set_title(title)
    axis.set_xlabel("Test gradient noise sigma")
    axis.set_ylabel("Mean IoU")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.show()


def plot_history(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    """Save available training and validation loss histories."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    for key in ("train_loss", "val_loss"):
        values = history.get(key)
        if values:
            axis.plot(range(1, len(values) + 1), values, label=key.replace("_", " ").title())
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.show()


__all__ = ["find_project_root", "plot_history", "plot_noise_curve", "plot_shape_curves"]

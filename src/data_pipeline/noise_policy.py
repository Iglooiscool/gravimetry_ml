"""Noise policy used by the final experiments.

The project specification adds independent Gaussian noise to the real and
imaginary gradient components of training samples only. Validation, random
test, and fixed test gradients remain clean.
"""

from __future__ import annotations


TRAINING_ONLY_NOISE = True

__all__ = ["TRAINING_ONLY_NOISE"]

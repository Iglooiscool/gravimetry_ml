"""Purpose: define the base interface shared by all shape specifications.

This file contains the abstract shape record used throughout the project so the
rest of the pipeline can work with different shapes through one common API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShapeSpec(ABC):
    """Base shape type shared by all concrete shape classes."""

    @property
    @abstractmethod
    def type(self) -> str:
        """Stable shape type identifier."""

    @abstractmethod
    def validate(self) -> None:
        """Raise ``ValueError`` if parameters are invalid."""

    @abstractmethod
    def compute_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return a boolean mask for points inside the shape."""

    # Purpose:
    # Export a shape as a plain dictionary so notebooks and JSON summaries can
    # store it without needing custom serialization code.
    #
    # Inputs:
    # - none beyond the current shape instance
    #
    # Returns:
    # - A dictionary containing the dataclass fields plus the stable shape type
    def to_record(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type
        return data

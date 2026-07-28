"""Combined stage configuration for the two-stage model stack."""

from __future__ import annotations

from dataclasses import dataclass, field

from .stage1 import Stage1ModelConfig
from .stage2 import Stage2ModelConfig


@dataclass(frozen=True)
class TwoStageStackConfig:
    """Model settings for the connected Stage 1 -> Stage 2 stack."""

    stage1: Stage1ModelConfig = field(default_factory=Stage1ModelConfig)
    stage2: Stage2ModelConfig = field(default_factory=Stage2ModelConfig)


__all__ = ["TwoStageStackConfig"]

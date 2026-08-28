"""Purpose: expose the public config API for runs and stage settings."""

from .model_stack import TwoModelsConfig, TwoStageStackConfig
from .runs import Task2GenerateConfig, TwoModelsRunConfig, TwoModelsSweepConfig, TwoStageRunConfig, TwoStageSweepConfig
from .stage1 import Stage1ModelConfig
from .stage2 import Stage2ModelConfig
from .task9 import (
    Task9GeneralMLPConfig,
    Task9RunConfig,
    Task9SpecialistMLPConfig,
    Task9StackConfig,
    Task9SweepConfig,
    ThreeModelsConfig,
    ThreeModelsRunConfig,
    ThreeModelsSweepConfig,
)
from .training import StageTrainingConfig
from .one_model import OneModelRunConfig
from .official import (
    OFFICIAL_SHAPE_WEIGHTS,
    OFFICIAL_TRAIN_SIGMAS,
    official_one_stage_config,
    official_three_stage_config,
    official_two_stage_config,
)

__all__ = [
    "Stage1ModelConfig",
    "Stage2ModelConfig",
    "StageTrainingConfig",
    "Task9GeneralMLPConfig",
    "Task9SpecialistMLPConfig",
    "Task9StackConfig",
    "Task9RunConfig",
    "Task9SweepConfig",
    "ThreeModelsConfig",
    "ThreeModelsRunConfig",
    "ThreeModelsSweepConfig",
    "TwoStageStackConfig",
    "TwoModelsConfig",
    "Task2GenerateConfig",
    "TwoStageRunConfig",
    "TwoStageSweepConfig",
    "TwoModelsRunConfig",
    "TwoModelsSweepConfig",
    "OneModelRunConfig",
    "OFFICIAL_SHAPE_WEIGHTS",
    "OFFICIAL_TRAIN_SIGMAS",
    "official_one_stage_config",
    "official_three_stage_config",
    "official_two_stage_config",
]

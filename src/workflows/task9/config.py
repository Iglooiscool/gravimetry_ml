"""Re-export Task 9 config objects from the main config package."""

from config.task9 import Task9GeneralMLPConfig, Task9RunConfig, Task9SpecialistMLPConfig, Task9StackConfig, Task9SweepConfig

__all__ = [
    "Task9GeneralMLPConfig",
    "Task9SpecialistMLPConfig",
    "Task9StackConfig",
    "Task9RunConfig",
    "Task9SweepConfig",
]

"""Official one-model workflow entry points."""

from .run import run_one_model
from .multitask import run_multitask_one_model

__all__ = ["run_one_model", "run_multitask_one_model"]

"""Model introspection helpers for reproducible notebook reports."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import nn


def parameter_count(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts for a PyTorch model."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def layer_table(model: nn.Module, dummy_input: torch.Tensor) -> pd.DataFrame:
    """Run a dummy input through leaf modules and return a shape table."""

    rows: list[dict[str, Any]] = []
    hooks: list[Any] = []

    def capture(name: str, module: nn.Module):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                output_shape = tuple(output.shape)
            else:
                output_shape = type(output).__name__
            rows.append(
                {
                    "layer": name or "root",
                    "type": module.__class__.__name__,
                    "output_shape": output_shape,
                    "parameters": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
                    "trainable": sum(
                        parameter.numel()
                        for parameter in module.parameters(recurse=False)
                        if parameter.requires_grad
                    ),
                }
            )

        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(capture(name, module)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return pd.DataFrame(rows)


__all__ = ["layer_table", "parameter_count"]

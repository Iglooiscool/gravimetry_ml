"""Pipeline diagnostics and training-history summaries."""

from __future__ import annotations

from models import evaluate_stage2_predictions


def summarize_training_history(history: dict[str, list[float]], requested_epochs: int) -> dict[str, float | int | bool | None]:
    """Create a compact training summary from the stored history lists."""

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    validation_steps = history.get("validation_steps", [])
    best_validation_loss = min(val_loss) if val_loss else None
    best_validation_step = validation_steps[val_loss.index(best_validation_loss)] if val_loss else None
    epochs_completed = len(train_loss)
    return {
        "epochs_requested": requested_epochs,
        "epochs_completed": epochs_completed,
        "stopped_early": epochs_completed < requested_epochs,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_validation_loss": val_loss[-1] if val_loss else None,
        "best_validation_loss": best_validation_loss,
        "best_validation_step": best_validation_step,
        "validation_checks": len(val_loss),
    }


def build_stage2_diagnostics(
    true_test_masks,
    true_fixed_masks,
    predicted_test_masks_with_true_coefficients,
    predicted_fixed_masks_with_true_coefficients,
    threshold_used: float,
) -> dict[str, object]:
    """Build diagnostic comparisons separate from the main pipeline runner."""

    return {
        "stage2_with_true_coefficients": {
            "test": evaluate_stage2_predictions(
                true_test_masks,
                predicted_test_masks_with_true_coefficients,
                threshold_used,
            ),
            "fixed": evaluate_stage2_predictions(
                true_fixed_masks,
                predicted_fixed_masks_with_true_coefficients,
                threshold_used,
            ),
        }
    }


__all__ = ["build_stage2_diagnostics", "summarize_training_history"]

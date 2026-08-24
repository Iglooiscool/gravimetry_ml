"""Run the two-stage absolute-noise sweep sequentially with bounded CPU memory."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from config import (
    Stage1ModelConfig,
    Stage2ModelConfig,
    StageTrainingConfig,
    TwoStageRunConfig,
    TwoStageStackConfig,
)
from pipeline import run_two_stage_once


SIGMAS = (0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01)


def build_config(sigma: float, output_dir: Path, small: bool) -> TwoStageRunConfig:
    samples = 2000 if small else 10000
    validation = 500 if small else 2000
    stage1 = Stage1ModelConfig(
        hidden_layer_sizes=(128, 256, 128),
        dropout_rates=(0.2, 0.2, 0.2),
        training=StageTrainingConfig(
            epochs=120 if small else 200,
            batch_size=32,
            learning_rate=0.0005,
            validation_frequency=10,
            verbose=True,
            early_stopping_patience=10,
        ),
    )
    stage2 = Stage2ModelConfig(
        hidden_layer_sizes=(512, 1024),
        dropout_rates=(0.1, 0.1),
        model_type="coord_conv_decoder",
        latent_grid_size=16,
        latent_channels=160,
        decoder_channels=(160, 128, 96, 64, 32),
        use_rectangle_edge_weighting=True,
        use_foreground_pos_weight=False,
        rectangle_edge_weight=4.0,
        rectangle_edge_width=3,
        edge_weight_mode="rectangle",
        annulus_edge_width=3,
        training=StageTrainingConfig(
            epochs=100 if small else 170,
            batch_size=96,
            learning_rate=0.0005,
            validation_frequency=30,
            verbose=True,
            early_stopping_patience=24,
            min_epochs=30 if small else 50,
            min_improvement=0.002,
            lr_drop_factor=0.5,
            lr_drop_period=80,
            weight_decay=0.00025,
            gradient_clip_norm=0.8,
            loss_type="bce_dice",
            dice_loss_weight=1.0,
        ),
    )
    return TwoStageRunConfig(
        N=8,
        training_samples=samples,
        validation_samples=validation,
        test_samples=500,
        rho=0.8,
        grid_size=32,
        noise_sigma=sigma,
        noise_mode="absolute",
        seed=42,
        stage2_predicted_coefficient_augmentation_copies=2,
        stage2_predicted_coefficient_noise_scale=1.0,
        model=TwoStageStackConfig(stage1=stage1, stage2=stage2),
        output_dir=output_dir / f"sigma_{sigma:g}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/two_models_noise_sweep"))
    parser.add_argument("--small", action="store_true", help="Use a low-memory smoke sweep.")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU thread limit.")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sigma in SIGMAS:
        print(f"\n=== absolute sigma={sigma:g} ===", flush=True)
        summary = run_two_stage_once(build_config(sigma, args.output_dir, args.small))
        (args.output_dir / f"sigma_{sigma:g}" / "sweep_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        del summary
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

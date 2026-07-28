"""Helpers for running a small, explicit three-trial tuning loop."""

from __future__ import annotations

from dataclasses import replace

from config import TwoStageRunConfig
from pipeline import run_two_stage_once


def build_three_trial_tuning_configs(base_run_config: TwoStageRunConfig) -> dict[str, TwoStageRunConfig]:
    """Build baseline, Stage 1, and Stage 2 trial configs from one base run config."""

    baseline_config = replace(
        base_run_config,
        output_dir=base_run_config.output_dir / "baseline",
    )

    stage1_variant = replace(
        baseline_config.model.stage1,
        hidden_layer_sizes=(256, 256, 128),
        dropout_rates=(0.15, 0.15, 0.1),
        training=replace(
            baseline_config.model.stage1.training,
            learning_rate=0.0003,
            early_stopping_patience=15,
        ),
    )
    stage1_config = replace(
        baseline_config,
        model=replace(
            baseline_config.model,
            stage1=stage1_variant,
        ),
        output_dir=base_run_config.output_dir / "stage1_tuned",
    )

    stage2_variant = replace(
        baseline_config.model.stage2,
        hidden_layer_sizes=(768, 1536, 2048),
        dropout_rates=(0.15, 0.15, 0.1),
        training=replace(
            baseline_config.model.stage2.training,
            learning_rate=0.0007,
            min_improvement=0.003,
            lr_drop_period=60,
            weight_decay=0.0002,
            gradient_clip_norm=0.8,
        ),
    )
    stage2_config = replace(
        baseline_config,
        model=replace(
            baseline_config.model,
            stage2=stage2_variant,
        ),
        output_dir=base_run_config.output_dir / "stage2_tuned",
    )

    return {
        "baseline": baseline_config,
        "stage1_tuned": stage1_config,
        "stage2_tuned": stage2_config,
    }


def run_three_trial_tuning(
    base_run_config: TwoStageRunConfig,
    device=None,
) -> dict[str, dict[str, object]]:
    """Run the three standard tuning trials and return their summaries."""

    trial_configs = build_three_trial_tuning_configs(base_run_config)
    results: dict[str, dict[str, object]] = {}
    for trial_name, trial_config in trial_configs.items():
        trial_summary = run_two_stage_once(trial_config, device=device)
        results[trial_name] = {
            "config": trial_summary["config"],
            "metrics": trial_summary["metrics"],
            "metrics_by_shape": trial_summary["metrics_by_shape"],
            "diagnostics": trial_summary["diagnostics"],
            "training_summary": trial_summary["training_summary"],
            "summary_path": str(trial_config.run_output_dir / "summary.json"),
        }
    return results


def build_stage2_upgrade_tuning_configs(base_run_config: TwoStageRunConfig) -> dict[str, TwoStageRunConfig]:
    """Build three stronger Stage 2 trials around the current best Stage 1 settings."""

    stage1_best = replace(
        base_run_config.model.stage1,
        hidden_layer_sizes=(256, 256, 128),
        dropout_rates=(0.15, 0.15, 0.1),
        training=replace(
            base_run_config.model.stage1.training,
            learning_rate=0.0003,
            early_stopping_patience=15,
        ),
    )

    decoder_baseline = replace(
        base_run_config.model.stage2,
        model_type="conv_decoder",
        hidden_layer_sizes=(256, 512),
        dropout_rates=(0.15, 0.15),
        latent_grid_size=8,
        latent_channels=64,
        decoder_channels=(64, 32, 16),
        use_rectangle_edge_weighting=True,
        edge_weight_mode="rectangle",
        training=replace(
            base_run_config.model.stage2.training,
            batch_size=128,
            learning_rate=0.0007,
            validation_frequency=60,
            lr_drop_period=60,
            weight_decay=0.0002,
            gradient_clip_norm=0.8,
        ),
    )
    decoder_edges_all = replace(
        decoder_baseline,
        edge_weight_mode="all",
        rectangle_edge_weight=4.0,
        rectangle_edge_width=2,
    )
    decoder_disconnected_focus = replace(
        decoder_baseline,
        edge_weight_mode="all",
        rectangle_edge_weight=3.5,
        rectangle_edge_width=3,
        latent_channels=96,
        decoder_channels=(96, 64, 32),
        training=replace(
            decoder_baseline.training,
            learning_rate=0.0006,
            weight_decay=0.0003,
            min_improvement=0.003,
            lr_drop_period=70,
        ),
    )

    return {
        "decoder_baseline": replace(
            base_run_config,
            model=replace(base_run_config.model, stage1=stage1_best, stage2=decoder_baseline),
            output_dir=base_run_config.output_dir / "decoder_baseline",
        ),
        "decoder_edges_all": replace(
            base_run_config,
            model=replace(base_run_config.model, stage1=stage1_best, stage2=decoder_edges_all),
            output_dir=base_run_config.output_dir / "decoder_edges_all",
        ),
        "decoder_disconnected_focus": replace(
            base_run_config,
            model=replace(base_run_config.model, stage1=stage1_best, stage2=decoder_disconnected_focus),
            output_dir=base_run_config.output_dir / "decoder_disconnected_focus",
        ),
    }


def run_stage2_upgrade_tuning(
    base_run_config: TwoStageRunConfig,
    device=None,
) -> dict[str, dict[str, object]]:
    """Run three stronger Stage 2 trials and return their summaries."""

    trial_configs = build_stage2_upgrade_tuning_configs(base_run_config)
    results: dict[str, dict[str, object]] = {}
    for trial_name, trial_config in trial_configs.items():
        trial_summary = run_two_stage_once(trial_config, device=device)
        results[trial_name] = {
            "config": trial_summary["config"],
            "metrics": trial_summary["metrics"],
            "metrics_by_shape": trial_summary["metrics_by_shape"],
            "diagnostics": trial_summary["diagnostics"],
            "training_summary": trial_summary["training_summary"],
            "summary_path": str(trial_config.run_output_dir / "summary.json"),
        }
    return results


def build_stage2_loss_upgrade_configs(base_run_config: TwoStageRunConfig) -> dict[str, TwoStageRunConfig]:
    """Build three Stage 2 loss trials from the current best decoder direction."""

    stage1_best = replace(
        base_run_config.model.stage1,
        hidden_layer_sizes=(256, 256, 128),
        dropout_rates=(0.15, 0.15, 0.1),
        training=replace(
            base_run_config.model.stage1.training,
            learning_rate=0.0003,
            early_stopping_patience=15,
        ),
    )
    stage2_best_decoder = replace(
        base_run_config.model.stage2,
        model_type="conv_decoder",
        hidden_layer_sizes=(256, 512),
        dropout_rates=(0.15, 0.15),
        latent_grid_size=8,
        latent_channels=96,
        decoder_channels=(96, 64, 32),
        use_rectangle_edge_weighting=True,
        rectangle_edge_weight=3.5,
        rectangle_edge_width=3,
        edge_weight_mode="all",
        training=replace(
            base_run_config.model.stage2.training,
            batch_size=128,
            learning_rate=0.0006,
            validation_frequency=60,
            lr_drop_period=70,
            weight_decay=0.0003,
            gradient_clip_norm=0.8,
            min_improvement=0.003,
            loss_type="bce",
            dice_loss_weight=0.0,
            dice_smooth=1.0,
        ),
    )
    stage2_bce_dice_light = replace(
        stage2_best_decoder,
        training=replace(
            stage2_best_decoder.training,
            loss_type="bce_dice",
            dice_loss_weight=0.5,
            dice_smooth=1.0,
        ),
    )
    stage2_bce_dice_strong = replace(
        stage2_best_decoder,
        training=replace(
            stage2_best_decoder.training,
            loss_type="bce_dice",
            dice_loss_weight=1.0,
            dice_smooth=1.0,
        ),
    )

    return {
        "decoder_bce_baseline": replace(
            base_run_config,
            model=replace(base_run_config.model, stage1=stage1_best, stage2=stage2_best_decoder),
            output_dir=base_run_config.output_dir / "decoder_bce_baseline",
        ),
        "decoder_bce_dice_light": replace(
            base_run_config,
            model=replace(base_run_config.model, stage1=stage1_best, stage2=stage2_bce_dice_light),
            output_dir=base_run_config.output_dir / "decoder_bce_dice_light",
        ),
        "decoder_bce_dice_strong": replace(
            base_run_config,
            model=replace(base_run_config.model, stage1=stage1_best, stage2=stage2_bce_dice_strong),
            output_dir=base_run_config.output_dir / "decoder_bce_dice_strong",
        ),
    }


def run_stage2_loss_upgrade_tuning(
    base_run_config: TwoStageRunConfig,
    device=None,
) -> dict[str, dict[str, object]]:
    """Run three Stage 2 loss-upgrade trials and return their summaries."""

    trial_configs = build_stage2_loss_upgrade_configs(base_run_config)
    results: dict[str, dict[str, object]] = {}
    for trial_name, trial_config in trial_configs.items():
        trial_summary = run_two_stage_once(trial_config, device=device)
        results[trial_name] = {
            "config": trial_summary["config"],
            "metrics": trial_summary["metrics"],
            "metrics_by_shape": trial_summary["metrics_by_shape"],
            "diagnostics": trial_summary["diagnostics"],
            "training_summary": trial_summary["training_summary"],
            "summary_path": str(trial_config.run_output_dir / "summary.json"),
        }
    return results


def build_stage2_threshold_sampling_configs(base_run_config: TwoStageRunConfig) -> dict[str, TwoStageRunConfig]:
    """Build threshold-sweep and shape-balancing trials from the best current decoder setup."""

    loss_configs = build_stage2_loss_upgrade_configs(base_run_config)
    baseline = loss_configs["decoder_bce_dice_strong"]
    threshold_sweep = replace(
        baseline,
        use_validation_threshold_sweep=True,
        output_dir=base_run_config.output_dir / "decoder_bce_dice_threshold_sweep",
    )
    balanced_threshold_sweep = replace(
        threshold_sweep,
        training_shape_weights=(
            ("rectangle", 0.3),
            ("two_circles", 0.3),
            ("annulus", 0.2),
            ("ellipse", 0.1),
            ("circle", 0.1),
        ),
        output_dir=base_run_config.output_dir / "decoder_bce_dice_balanced_threshold_sweep",
    )
    return {
        "decoder_bce_dice_baseline": baseline,
        "decoder_bce_dice_threshold_sweep": threshold_sweep,
        "decoder_bce_dice_balanced_threshold_sweep": balanced_threshold_sweep,
    }


def run_stage2_threshold_sampling_tuning(
    base_run_config: TwoStageRunConfig,
    device=None,
) -> dict[str, dict[str, object]]:
    """Run three threshold-sweep and shape-balancing trials."""

    trial_configs = build_stage2_threshold_sampling_configs(base_run_config)
    results: dict[str, dict[str, object]] = {}
    for trial_name, trial_config in trial_configs.items():
        trial_summary = run_two_stage_once(trial_config, device=device)
        results[trial_name] = {
            "config": trial_summary["config"],
            "metrics": trial_summary["metrics"],
            "metrics_by_shape": trial_summary["metrics_by_shape"],
            "diagnostics": trial_summary["diagnostics"],
            "training_summary": trial_summary["training_summary"],
            "threshold_summary": trial_summary["threshold_summary"],
            "summary_path": str(trial_config.run_output_dir / "summary.json"),
        }
    return results


__all__ = [
    "build_three_trial_tuning_configs",
    "run_three_trial_tuning",
    "build_stage2_upgrade_tuning_configs",
    "build_stage2_loss_upgrade_configs",
    "build_stage2_threshold_sampling_configs",
    "run_stage2_upgrade_tuning",
    "run_stage2_loss_upgrade_tuning",
    "run_stage2_threshold_sampling_tuning",
]

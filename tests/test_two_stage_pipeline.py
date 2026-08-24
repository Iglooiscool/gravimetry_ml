import numpy as np

from config import OneModelRunConfig, Stage1ModelConfig, Stage2ModelConfig, StageTrainingConfig, TwoStageRunConfig, TwoStageStackConfig, TwoStageSweepConfig
from datasets import build_two_stage_datasets
from measurements import (
    build_measurement_matrix,
    coefficient_features_to_complex,
    compute_gradient_data,
    measurement_points_on_unit_circle,
    measurements_to_feature_vector,
)
import torch

from models import Stage1Regressor, Stage2ConvDecoder, Stage2MaskPredictor, binary_iou, compute_rectangle_edge_pixel_weights, compute_shape_edge_pixel_weights, evaluate_stage2_predictions_by_shape, fit_stage2_model, predict_stage2_logits, select_best_stage2_threshold, stop_if_overfitting, stop_if_safe
from pipeline import run_two_stage_once
from shapes import CircleSpec, create_fixed_benchmark_shapes, sample_random_shape
from workflows.one_model import run_one_model
from tuning import build_stage2_loss_upgrade_configs, build_stage2_threshold_sampling_configs, build_stage2_upgrade_tuning_configs, build_three_trial_tuning_configs, run_stage2_loss_upgrade_tuning, run_stage2_threshold_sampling_tuning, run_stage2_upgrade_tuning, run_three_trial_tuning


def test_fixed_benchmark_shapes_exist():
    shapes = create_fixed_benchmark_shapes()
    assert len(shapes) == 5
    assert shapes[0][1].type == "ellipse"
    assert shapes[1][1].type == "circle"
    assert shapes[-1][1].type == "two_circles"


def test_measurement_matrix_shape_matches_prompt():
    points = measurement_points_on_unit_circle(3)
    matrix = build_measurement_matrix(points, n_max=2)
    assert matrix.shape == (3, 3)
    assert np.isfinite(np.abs(matrix)).all()


def test_two_stage_dataset_shapes_are_consistent():
    cfg = TwoStageRunConfig(N=2, training_samples=8, validation_samples=4, test_samples=3, grid_size=16)
    dataset = build_two_stage_datasets(cfg)
    assert dataset.train.gradient_data.shape == (8, 6)
    assert dataset.train.coefficients.shape == (8, 6)
    assert dataset.train.masks.shape == (8, 256)
    assert dataset.fixed.masks.shape[0] == 5


def test_two_stage_fused_input_size_includes_raw_gradients():
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=8,
        validation_samples=4,
        stage2_include_gradient_features=True,
    )

    assert cfg.stage2_input_size == cfg.coefficient_size + cfg.gradient_feature_size


def test_noise_is_applied_only_to_training_gradients():
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=32,
        validation_samples=16,
        test_samples=16,
        grid_size=16,
        noise_sigma=0.01,
        seed=11,
    )
    dataset = build_two_stage_datasets(cfg)

    for split in (dataset.validation, dataset.test, dataset.fixed):
        coefficients = coefficient_features_to_complex(split.coefficients)
        clean_gradients = np.stack(
            [
                measurements_to_feature_vector(
                    compute_gradient_data(row, dataset.measurement_points, cfg.N)
                )
                for row in coefficients
            ]
        )
        assert np.allclose(split.gradient_data, clean_gradients)

    train_coefficients = coefficient_features_to_complex(dataset.train.coefficients)
    clean_train_gradients = np.stack(
        [
            measurements_to_feature_vector(
                compute_gradient_data(row, dataset.measurement_points, cfg.N)
            )
            for row in train_coefficients
        ]
    )
    assert not np.allclose(dataset.train.gradient_data, clean_train_gradients)


def test_training_noise_replicas_only_expand_training_split():
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=8,
        validation_samples=4,
        test_samples=3,
        grid_size=16,
        noise_sigma=0.01,
        training_noise_replicas=3,
        seed=13,
    )
    dataset = build_two_stage_datasets(cfg)

    assert dataset.train.gradient_data.shape[0] == 24
    assert dataset.train.coefficients.shape[0] == 24
    assert dataset.validation.gradient_data.shape[0] == 4
    assert dataset.test.gradient_data.shape[0] == 3
    assert dataset.fixed.gradient_data.shape[0] == 5


def test_two_stage_dataset_can_be_limited_to_selected_shapes():
    cfg = TwoStageRunConfig(N=2, training_samples=8, validation_samples=4, test_samples=3, grid_size=16)
    dataset = build_two_stage_datasets(cfg, allowed_shape_types=("circle", "rectangle"))
    assert set(dataset.train.shape_types).issubset({"circle", "rectangle"})
    assert set(dataset.validation.shape_types).issubset({"circle", "rectangle"})
    assert set(dataset.test.shape_types).issubset({"circle", "rectangle"})
    assert set(dataset.fixed.shape_types).issubset({"circle", "rectangle"})


def test_one_model_mlp_workflow_constructs_without_decoder_arguments(tmp_path):
    cfg = OneModelRunConfig(
        N=2,
        training_samples=8,
        validation_samples=4,
        test_samples=3,
        grid_size=8,
        noise_sigma=0.01,
        output_dir=tmp_path / "one_model_mlp",
        model=Stage2ModelConfig(
            hidden_layer_sizes=(16,),
            dropout_rates=(0.0,),
            model_type="mlp",
            training=StageTrainingConfig(
                epochs=1,
                batch_size=4,
                learning_rate=0.001,
                validation_frequency=1,
                verbose=False,
                early_stopping_patience=None,
            ),
        ),
    )
    summary = run_one_model(cfg, device=torch.device("cpu"))
    assert summary["model_count"] == 1
    assert "test" in summary["metrics"]


def test_two_stage_dataset_can_bias_training_shapes_only():
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=40,
        validation_samples=20,
        test_samples=20,
        grid_size=16,
        seed=7,
        training_shape_weights=(("rectangle", 0.5), ("two_circles", 0.5)),
    )
    dataset = build_two_stage_datasets(cfg)
    assert set(dataset.train.shape_types).issubset({"rectangle", "two_circles"})
    assert len(set(dataset.validation.shape_types)) >= 2


def test_circle_sampling_returns_distinct_circle_type():
    rng = np.random.default_rng(7)
    shape = sample_random_shape(rng, "circle")
    assert isinstance(shape, CircleSpec)
    assert shape.type == "circle"


def test_two_stage_models_output_shapes():
    stage1 = Stage1Regressor(input_dim=6, output_dim=6)
    stage2 = Stage2MaskPredictor(input_dim=6, output_dim=256, hidden_dims=(32, 64), dropout_rates=(0.1, 0.2))
    conv_stage2 = Stage2ConvDecoder(
        input_dim=6,
        output_dim=256,
        hidden_dims=(32, 64),
        dropout_rates=(0.1, 0.2),
        latent_grid_size=8,
        latent_channels=16,
        decoder_channels=(16, 8),
    )
    assert tuple(stage1.network[-1].weight.shape) == (6, 128)
    assert tuple(stage2.network[-1].weight.shape) == (256, 64)
    assert any(isinstance(layer, torch.nn.Dropout) for layer in stage2.network)
    assert conv_stage2(torch.randn(3, 6)).shape == (3, 256)


def test_stage2_training_stores_input_normalization_and_predicts_logits():
    rng = np.random.default_rng(5)
    train_features = rng.normal(size=(12, 6)).astype(np.float32)
    train_targets = (rng.random(size=(12, 16)) > 0.5).astype(np.float32)
    val_features = rng.normal(size=(6, 6)).astype(np.float32)
    val_targets = (rng.random(size=(6, 16)) > 0.5).astype(np.float32)

    model = Stage2MaskPredictor(input_dim=6, output_dim=16, hidden_dims=(8,), dropout_rates=(0.1,))
    result = fit_stage2_model(
        model=model,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        device=torch.device("cpu"),
        validation_frequency=1,
        verbose=False,
        early_stopping_patience=None,
        min_epochs=1,
        min_improvement=0.005,
        lr_drop_factor=0.5,
        lr_drop_period=1,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
    )

    assert result.input_mean is not None
    assert result.input_std is not None
    logits = predict_stage2_logits(model, val_features, device=torch.device("cpu"), training_result=result)
    assert logits.shape == (6, 16)


def test_stop_if_safe_does_not_stop_before_min_epochs():
    should_stop, best_validation_loss, wait_count = stop_if_safe(
        validation_loss=0.60,
        best_validation_loss=0.50,
        wait_count=19,
        patience=20,
        epoch_index=9,
        min_epochs=40,
        min_improvement=0.005,
    )
    assert should_stop is False
    assert best_validation_loss == 0.50
    assert wait_count == 19


def test_stop_if_safe_requires_minimum_relative_improvement():
    should_stop, best_validation_loss, wait_count = stop_if_safe(
        validation_loss=0.498,
        best_validation_loss=0.50,
        wait_count=3,
        patience=20,
        epoch_index=49,
        min_epochs=40,
        min_improvement=0.005,
    )
    assert should_stop is False
    assert best_validation_loss == 0.50
    assert wait_count == 4


def test_stop_if_safe_resets_wait_count_after_sufficient_improvement():
    should_stop, best_validation_loss, wait_count = stop_if_safe(
        validation_loss=0.497,
        best_validation_loss=0.50,
        wait_count=3,
        patience=20,
        epoch_index=49,
        min_epochs=40,
        min_improvement=0.005,
    )
    assert should_stop is False
    assert best_validation_loss == 0.497
    assert wait_count == 0


def test_evaluate_stage2_predictions_by_shape_groups_metrics_correctly():
    y_true = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    logits = np.array(
        [
            [10, -10, -10, -10],
            [10, 10, -10, -10],
            [-10, -10, 10, -10],
        ],
        dtype=np.float32,
    )
    metrics = evaluate_stage2_predictions_by_shape(
        y_true=y_true,
        logits=logits,
        threshold=0.5,
        shape_types=("circle", "circle", "rectangle"),
    )

    assert metrics["circle"]["sample_count"] == 2
    assert metrics["circle"]["mean_iou"] == 1.0
    assert metrics["rectangle"]["sample_count"] == 1
    assert metrics["rectangle"]["mean_iou"] == 0.5


def test_select_best_stage2_threshold_returns_best_validation_iou():
    y_true = np.array([[1, 1, 0, 0]], dtype=np.float32)
    logits = np.array([[2.0, -0.2, -1.0, -2.0]], dtype=np.float32)
    selection = select_best_stage2_threshold(y_true, logits, (0.3, 0.5, 0.7))

    assert selection["selected_threshold"] == 0.3
    assert len(selection["candidates"]) == 3


def test_rectangle_edge_pixel_weights_are_higher_on_rectangle_boundaries():
    grid_size = 6
    rectangle_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    circle_like_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    masks = np.stack([rectangle_mask.reshape(-1), circle_like_mask.reshape(-1)])

    pixel_weights = compute_rectangle_edge_pixel_weights(
        masks=masks,
        shape_types=("rectangle", "circle"),
        grid_size=grid_size,
        edge_weight=3.0,
        edge_width=0,
    )

    assert pixel_weights.shape == masks.shape
    assert np.max(pixel_weights[0]) == 3.0
    assert np.min(pixel_weights[0]) == 1.0
    assert np.all(pixel_weights[1] == 1.0)


def test_shape_edge_pixel_weights_can_apply_to_all_shapes():
    grid_size = 6
    rectangle_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    rectangle_mask[2:4, 2:4] = 1.0
    circle_like_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    circle_like_mask[1:5, 2:4] = 1.0
    masks = np.stack([rectangle_mask.reshape(-1), circle_like_mask.reshape(-1)])

    pixel_weights = compute_shape_edge_pixel_weights(
        masks=masks,
        shape_types=("rectangle", "two_circles"),
        grid_size=grid_size,
        edge_weight=4.0,
        edge_width=0,
        edge_weight_mode="all",
    )

    assert np.max(pixel_weights[0]) == 4.0
    assert np.max(pixel_weights[1]) == 4.0


def test_stage2_training_supports_rectangle_edge_weighting():
    rng = np.random.default_rng(11)
    train_features = rng.normal(size=(8, 6)).astype(np.float32)
    train_targets = np.zeros((8, 16), dtype=np.float32)
    train_targets[:, 5:11] = 1.0
    val_features = rng.normal(size=(4, 6)).astype(np.float32)
    val_targets = np.zeros((4, 16), dtype=np.float32)
    val_targets[:, 5:11] = 1.0

    model = Stage2MaskPredictor(input_dim=6, output_dim=16, hidden_dims=(8,), dropout_rates=(0.1,))
    result = fit_stage2_model(
        model=model,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        device=torch.device("cpu"),
        validation_frequency=1,
        verbose=False,
        early_stopping_patience=None,
        train_shape_types=("rectangle", "circle", "rectangle", "ellipse", "rectangle", "circle", "annulus", "rectangle"),
        grid_size=4,
        use_rectangle_edge_weighting=True,
        rectangle_edge_weight=3.0,
        rectangle_edge_width=1,
        min_epochs=1,
        min_improvement=0.005,
        lr_drop_factor=0.5,
        lr_drop_period=1,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
    )

    assert result.input_mean is not None
    assert result.input_std is not None


def test_stage2_training_supports_bce_dice_loss():
    rng = np.random.default_rng(17)
    train_features = rng.normal(size=(8, 6)).astype(np.float32)
    train_targets = (rng.random(size=(8, 16)) > 0.5).astype(np.float32)
    val_features = rng.normal(size=(4, 6)).astype(np.float32)
    val_targets = (rng.random(size=(4, 16)) > 0.5).astype(np.float32)

    model = Stage2MaskPredictor(input_dim=6, output_dim=16, hidden_dims=(8,), dropout_rates=(0.1,))
    result = fit_stage2_model(
        model=model,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        device=torch.device("cpu"),
        validation_frequency=1,
        verbose=False,
        early_stopping_patience=None,
        loss_type="bce_dice",
        dice_loss_weight=0.5,
        dice_smooth=1.0,
    )

    assert result.history["train_loss"]
    assert result.history["val_loss"]


def test_binary_iou_simple_case():
    y_true = np.array([1, 1, 0, 0], dtype=np.float32)
    y_pred = np.array([1, 0, 0, 0], dtype=np.float32)
    assert binary_iou(y_true, y_pred) == 0.5


def test_stop_if_overfitting_resets_wait_count_on_improvement():
    should_stop, best_validation_loss, wait_count = stop_if_overfitting(
        validation_loss=0.25,
        best_validation_loss=0.50,
        wait_count=3,
        patience=10,
    )
    assert should_stop is False
    assert best_validation_loss == 0.25
    assert wait_count == 0


def test_stop_if_overfitting_triggers_after_patience_is_reached():
    should_stop, best_validation_loss, wait_count = stop_if_overfitting(
        validation_loss=0.50,
        best_validation_loss=0.25,
        wait_count=9,
        patience=10,
    )
    assert should_stop is True
    assert best_validation_loss == 0.25
    assert wait_count == 10


def test_two_stage_smoke_run_writes_outputs(tmp_path):
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=12,
        validation_samples=6,
        test_samples=4,
        grid_size=16,
        model=TwoStageStackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(128, 256, 128),
                dropout_rates=(0.2, 0.2, 0.2),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=10,
                ),
            ),
            stage2=Stage2ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=None,
                    min_epochs=1,
                    min_improvement=0.005,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                ),
            ),
        ),
        output_dir=tmp_path,
    )
    summary = run_two_stage_once(cfg)
    assert "metrics" in summary
    assert "metrics_by_shape" in summary
    assert "diagnostics" in summary
    assert "threshold_summary" in summary
    assert "training_summary" in summary
    assert "stage2_test" in summary["metrics_by_shape"]
    assert "stage2_with_true_coefficients" in summary["diagnostics"]
    assert "stage1" in summary["training_summary"]
    assert (tmp_path / "N_2" / "summary.json").exists()


def test_two_stage_smoke_run_supports_epoch_end_validation_only(tmp_path):
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=12,
        validation_samples=6,
        test_samples=4,
        grid_size=16,
        model=TwoStageStackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(128, 256, 128),
                dropout_rates=(0.2, 0.2, 0.2),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=None,
                    verbose=False,
                    early_stopping_patience=10,
                ),
            ),
            stage2=Stage2ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=None,
                    verbose=False,
                    early_stopping_patience=None,
                    min_epochs=1,
                    min_improvement=0.005,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                ),
            ),
        ),
        output_dir=tmp_path,
    )
    summary = run_two_stage_once(cfg)
    assert "metrics" in summary


def test_two_stage_smoke_run_supports_validation_threshold_sweep(tmp_path):
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=12,
        validation_samples=6,
        test_samples=4,
        grid_size=16,
        use_validation_threshold_sweep=True,
        threshold_candidates=(0.4, 0.5, 0.6),
        model=TwoStageStackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=10,
                ),
            ),
            stage2=Stage2ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=None,
                    min_epochs=1,
                    min_improvement=0.005,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                ),
            ),
        ),
        output_dir=tmp_path,
    )
    summary = run_two_stage_once(cfg)
    assert summary["threshold_summary"]["selection_mode"] == "validation_sweep"
    assert summary["threshold_summary"]["selected_threshold"] in {0.4, 0.5, 0.6}


def test_build_three_trial_tuning_configs_creates_distinct_variants(tmp_path):
    cfg = TwoStageRunConfig(output_dir=tmp_path, N=2, training_samples=12, validation_samples=6, test_samples=4)
    trial_configs = build_three_trial_tuning_configs(cfg)

    assert set(trial_configs) == {"baseline", "stage1_tuned", "stage2_tuned"}
    assert trial_configs["baseline"].output_dir == tmp_path / "baseline"
    assert trial_configs["stage1_tuned"].model.stage1.training.learning_rate == 0.0003
    assert trial_configs["stage2_tuned"].model.stage2.training.weight_decay == 0.0002


def test_build_stage2_upgrade_tuning_configs_create_decoder_trials(tmp_path):
    cfg = TwoStageRunConfig(output_dir=tmp_path, N=2, training_samples=12, validation_samples=6, test_samples=4)
    trial_configs = build_stage2_upgrade_tuning_configs(cfg)

    assert set(trial_configs) == {"decoder_baseline", "decoder_edges_all", "decoder_disconnected_focus"}
    assert trial_configs["decoder_baseline"].model.stage2.model_type == "conv_decoder"
    assert trial_configs["decoder_edges_all"].model.stage2.edge_weight_mode == "all"
    assert trial_configs["decoder_disconnected_focus"].model.stage2.latent_channels == 96


def test_build_stage2_loss_upgrade_configs_create_loss_trials(tmp_path):
    cfg = TwoStageRunConfig(output_dir=tmp_path, N=2, training_samples=12, validation_samples=6, test_samples=4)
    trial_configs = build_stage2_loss_upgrade_configs(cfg)

    assert set(trial_configs) == {"decoder_bce_baseline", "decoder_bce_dice_light", "decoder_bce_dice_strong"}
    assert trial_configs["decoder_bce_baseline"].model.stage2.training.loss_type == "bce"
    assert trial_configs["decoder_bce_dice_light"].model.stage2.training.dice_loss_weight == 0.5
    assert trial_configs["decoder_bce_dice_strong"].model.stage2.training.dice_loss_weight == 1.0


def test_build_stage2_threshold_sampling_configs_create_expected_trials(tmp_path):
    cfg = TwoStageRunConfig(output_dir=tmp_path, N=2, training_samples=12, validation_samples=6, test_samples=4)
    trial_configs = build_stage2_threshold_sampling_configs(cfg)

    assert set(trial_configs) == {
        "decoder_bce_dice_baseline",
        "decoder_bce_dice_threshold_sweep",
        "decoder_bce_dice_balanced_threshold_sweep",
    }
    assert trial_configs["decoder_bce_dice_threshold_sweep"].use_validation_threshold_sweep is True
    assert trial_configs["decoder_bce_dice_balanced_threshold_sweep"].training_shape_weights is not None


def test_run_three_trial_tuning_returns_compact_trial_summaries(tmp_path):
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=12,
        validation_samples=6,
        test_samples=4,
        grid_size=16,
        model=TwoStageStackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=10,
                ),
            ),
            stage2=Stage2ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=20,
                    min_epochs=1,
                    min_improvement=0.005,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                ),
            ),
        ),
        output_dir=tmp_path,
    )

    results = run_three_trial_tuning(cfg)

    assert set(results) == {"baseline", "stage1_tuned", "stage2_tuned"}
    for summary in results.values():
        assert "metrics" in summary
        assert "metrics_by_shape" in summary
        assert "training_summary" in summary
        assert "summary_path" in summary


def test_run_stage2_upgrade_tuning_returns_compact_trial_summaries(tmp_path):
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=12,
        validation_samples=6,
        test_samples=4,
        grid_size=16,
        model=TwoStageStackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=10,
                ),
            ),
            stage2=Stage2ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                model_type="mlp",
                latent_grid_size=4,
                latent_channels=16,
                decoder_channels=(16, 8),
                use_rectangle_edge_weighting=True,
                edge_weight_mode="rectangle",
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=20,
                    min_epochs=1,
                    min_improvement=0.005,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                ),
            ),
        ),
        output_dir=tmp_path,
    )

    results = run_stage2_upgrade_tuning(cfg)

    assert set(results) == {"decoder_baseline", "decoder_edges_all", "decoder_disconnected_focus"}
    for summary in results.values():
        assert "metrics" in summary
        assert "metrics_by_shape" in summary
        assert "training_summary" in summary
        assert "summary_path" in summary


def test_run_stage2_loss_upgrade_tuning_returns_compact_trial_summaries(tmp_path):
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=12,
        validation_samples=6,
        test_samples=4,
        grid_size=16,
        model=TwoStageStackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=10,
                ),
            ),
            stage2=Stage2ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                model_type="mlp",
                latent_grid_size=4,
                latent_channels=16,
                decoder_channels=(16, 8),
                use_rectangle_edge_weighting=True,
                edge_weight_mode="rectangle",
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=20,
                    min_epochs=1,
                    min_improvement=0.005,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                    loss_type="bce",
                    dice_loss_weight=0.0,
                    dice_smooth=1.0,
                ),
            ),
        ),
        output_dir=tmp_path,
    )

    results = run_stage2_loss_upgrade_tuning(cfg)

    assert set(results) == {"decoder_bce_baseline", "decoder_bce_dice_light", "decoder_bce_dice_strong"}
    for summary in results.values():
        assert "metrics" in summary
        assert "metrics_by_shape" in summary
        assert "training_summary" in summary
        assert "summary_path" in summary


def test_run_stage2_threshold_sampling_tuning_returns_compact_trial_summaries(tmp_path):
    cfg = TwoStageRunConfig(
        N=2,
        training_samples=12,
        validation_samples=6,
        test_samples=4,
        grid_size=16,
        model=TwoStageStackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=10,
                ),
            ),
            stage2=Stage2ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                model_type="mlp",
                latent_grid_size=4,
                latent_channels=16,
                decoder_channels=(16, 8),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=20,
                    min_epochs=1,
                    min_improvement=0.005,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                    loss_type="bce_dice",
                    dice_loss_weight=1.0,
                    dice_smooth=1.0,
                ),
            ),
        ),
        output_dir=tmp_path,
    )

    results = run_stage2_threshold_sampling_tuning(cfg)

    assert set(results) == {
        "decoder_bce_dice_baseline",
        "decoder_bce_dice_threshold_sweep",
        "decoder_bce_dice_balanced_threshold_sweep",
    }
    for summary in results.values():
        assert "metrics" in summary
        assert "threshold_summary" in summary
        assert "summary_path" in summary


def test_two_stage_sweep_config_creates_run_configs(tmp_path):
    sweep = TwoStageSweepConfig(
        n_values=(2, 4),
        training_sizes=(10, 12),
        validation_sizes=(4, 5),
        output_dir=tmp_path,
    )
    runs = sweep.iter_runs()
    assert [run.N for run in runs] == [2, 4]

import numpy as np

from config import Stage1ModelConfig, StageTrainingConfig, Task9GeneralMLPConfig, Task9RunConfig, Task9SpecialistMLPConfig, Task9StackConfig
from models.stage2 import SigmoidMSEMaskLoss
from models.task9 import combine_task9_logits
from workflows.task9 import build_task9_general_dataset, build_task9_specialist_dataset, run_task9_once
from workflows.task9.datasets import augment_task9_feature_rows, augment_task9_general_training_split


def test_build_task9_specialist_dataset_contains_only_two_circles():
    cfg = Task9RunConfig(
        N=4,
        training_samples=8,
        validation_samples=4,
        specialist_training_samples=6,
        specialist_validation_samples=3,
        test_samples=3,
        grid_size=16,
    )

    specialist_dataset = build_task9_specialist_dataset(cfg)

    assert set(specialist_dataset.train.shape_types) == {"two_circles"}
    assert set(specialist_dataset.validation.shape_types) == {"two_circles"}
    assert set(specialist_dataset.test.shape_types) == {"two_circles"}
    assert specialist_dataset.fixed.names == (
        "fixed_separated",
        "fixed_touching",
        "fixed_overlapping",
        "fixed_nested",
    )


def test_task9_uses_sigmoid_mse_mask_loss():
    import torch

    loss = SigmoidMSEMaskLoss()
    predictions = torch.tensor([[0.0, 2.0]])
    targets = torch.tensor([[0.0, 1.0]])
    expected = (((torch.sigmoid(predictions) - targets) ** 2).mean()).item()
    assert loss(predictions, targets).item() == expected


def test_task9_general_rectangle_augmentation_duplicates_rectangle_rows():
    cfg = Task9RunConfig(
        N=4,
        training_samples=16,
        validation_samples=4,
        test_samples=4,
        grid_size=16,
    )

    general_dataset = build_task9_general_dataset(cfg)
    augmented_split = augment_task9_general_training_split(general_dataset.train, cfg.model.general)
    augmented_features = augment_task9_feature_rows(general_dataset.train.coefficients, general_dataset.train.shape_types, cfg.model.general)

    rectangle_count = sum(shape_type == "rectangle" for shape_type in general_dataset.train.shape_types)
    expected_added = rectangle_count * cfg.model.general.rectangle_augmentation_copies
    assert augmented_split.gradient_data.shape[0] == general_dataset.train.gradient_data.shape[0] + expected_added
    assert augmented_features.shape[0] == general_dataset.train.coefficients.shape[0] + expected_added
    assert sum(shape_type == "rectangle" for shape_type in augmented_split.shape_types) == rectangle_count * (1 + cfg.model.general.rectangle_augmentation_copies)


def test_combine_task9_logits_uses_specialist_for_two_circles():
    general_logits = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    specialist_logits = np.array([[9.0, 9.0]], dtype=np.float32)
    combined_logits, specialist_count = combine_task9_logits(
        general_logits=general_logits,
        specialist_logits=specialist_logits,
        shape_types=("circle", "two_circles", "rectangle"),
        specialist_shape_type="two_circles",
        routing_mode="true_shape_type",
    )

    assert specialist_count == 1
    assert np.allclose(combined_logits[0], general_logits[0])
    assert np.allclose(combined_logits[1], specialist_logits[0])
    assert np.allclose(combined_logits[2], general_logits[2])


def test_run_task9_once_writes_outputs(tmp_path):
    cfg = Task9RunConfig(
        N=4,
        training_samples=12,
        validation_samples=6,
        specialist_training_samples=8,
        specialist_validation_samples=4,
        test_samples=4,
        grid_size=16,
        model=Task9StackConfig(
            stage1=Stage1ModelConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=2,
                ),
            ),
            general=Task9GeneralMLPConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                use_rectangle_edge_weighting=True,
                enable_rectangle_edge_augmentation=True,
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=None,
                    min_epochs=1,
                    min_improvement=0.001,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    gradient_clip_norm=1.0,
                    loss_type="bce",
                ),
            ),
            specialist=Task9SpecialistMLPConfig(
                hidden_layer_sizes=(32, 32),
                dropout_rates=(0.1, 0.1),
                training=StageTrainingConfig(
                    epochs=1,
                    batch_size=4,
                    learning_rate=0.0005,
                    validation_frequency=1,
                    verbose=False,
                    early_stopping_patience=None,
                    lr_drop_factor=0.5,
                    lr_drop_period=1,
                    weight_decay=0.0001,
                    loss_type="bce",
                ),
            ),
        ),
        output_dir=tmp_path,
    )

    summary = run_task9_once(cfg)

    assert "metrics" in summary
    assert "diagnostics" in summary
    assert "training_summary" in summary
    assert "figure_paths" in summary
    assert summary["diagnostics"]["routing"]["specialist_enabled"] == 1
    assert summary["diagnostics"]["routing"]["rectangle_augmentation_added"] >= 0
    assert (tmp_path / "N_4" / "summary.json").exists()
    assert (tmp_path / "N_4" / "task9_general_model.pt").exists()
    assert (tmp_path / "N_4" / "task9_specialist_model.pt").exists()
    assert (tmp_path / "N_4" / "test_best_reconstructions.png").exists()
    assert (tmp_path / "N_4" / "test_worst_reconstructions.png").exists()
    assert (tmp_path / "N_4" / "test_random_reconstructions.png").exists()
    assert (tmp_path / "N_4" / "test_general_vs_combined.png").exists()
    assert (tmp_path / "N_4" / "fixed_general_vs_combined.png").exists()

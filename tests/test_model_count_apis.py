import torch

from config import OneModelRunConfig, Stage2ModelConfig, StageTrainingConfig
from models.one_model import GradientToMaskMLP, GradientToMaskModel
from models.three_models import CoefficientToGeneralMaskModel, CoefficientToSpecialistMaskModel, ThreeModelSystem
from models.two_models import CoefficientToMaskModel, GradientToCoefficientModel, TwoModelSystem


def test_model_count_packages_expose_distinct_model_roles():
    one_model = GradientToMaskModel(input_dim=6, output_dim=256, latent_grid_size=4, latent_channels=8, decoder_channels=(8, 4))
    one_model_mlp = GradientToMaskMLP(input_dim=6, output_dim=256, hidden_dims=(8,), dropout_rates=(0.1,))
    coefficient_model = GradientToCoefficientModel(input_dim=6, output_dim=6, hidden_dims=(8,), dropout_rates=(0.1,))
    mask_model = CoefficientToMaskModel(input_dim=6, output_dim=256, latent_grid_size=4, latent_channels=8, decoder_channels=(8, 4))
    general_model = CoefficientToGeneralMaskModel(input_dim=6, output_dim=256, hidden_dims=(8,), dropout_rates=(0.1,))
    specialist_model = CoefficientToSpecialistMaskModel(input_dim=6, output_dim=256, hidden_dims=(8,), dropout_rates=(0.1,))

    assert one_model(torch.randn(2, 6)).shape == (2, 256)
    assert one_model_mlp(torch.randn(2, 6)).shape == (2, 256)
    assert coefficient_model(torch.randn(2, 6)).shape == (2, 6)
    assert mask_model(torch.randn(2, 6)).shape == (2, 256)
    assert general_model(torch.randn(2, 6)).shape == (2, 256)
    assert specialist_model(torch.randn(2, 6)).shape == (2, 256)
    assert isinstance(TwoModelSystem(coefficient_model, mask_model), TwoModelSystem)
    assert isinstance(ThreeModelSystem(coefficient_model, general_model, specialist_model), ThreeModelSystem)


def test_one_model_config_uses_official_defaults():
    config = OneModelRunConfig()

    assert config.N == 8
    assert config.training_samples == 10_000
    assert config.validation_samples == 2_000
    assert config.model.model_type == "mlp"
    assert config.model.hidden_layer_sizes == (512, 1024, 2048)
    assert config.model.dropout_rates == (0.2, 0.2, 0.2)
    assert config.model.training.loss_type == "mse"
    assert config.model.use_rectangle_edge_weighting is False
    assert config.output_dir.name == "one_model"
    assert config.training_shape_weights == (
        ("rectangle", 0.25),
        ("two_circles", 0.45),
        ("annulus", 0.10),
        ("ellipse", 0.15),
        ("circle", 0.05),
    )


def test_one_model_config_supports_small_training_settings():
    config = OneModelRunConfig(
        N=2,
        training_samples=8,
        validation_samples=4,
        grid_size=16,
        model=Stage2ModelConfig(
            hidden_layer_sizes=(8,),
            dropout_rates=(0.1,),
            model_type="coord_conv_decoder",
            latent_grid_size=4,
            latent_channels=8,
            decoder_channels=(8, 4),
            training=StageTrainingConfig(epochs=1, batch_size=4, learning_rate=0.001, validation_frequency=1, verbose=False),
        ),
    )

    assert config.gradient_feature_size == 6
    assert config.mask_pixels == 256

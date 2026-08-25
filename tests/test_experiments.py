import numpy as np
import torch

from experiments import feature_matrix_with_noise
from models.stage2.losses import WeightedBinaryMaskLoss
from models.one_model import AnnulusRouterOneStage, fit_annulus_router, predict_annulus_router


def test_feature_matrix_noise_is_shape_stable_and_reproducible():
    features = np.zeros((3, 8), dtype=np.float32)
    first = feature_matrix_with_noise(features, sigma=0.01, seed=7)
    second = feature_matrix_with_noise(features, sigma=0.01, seed=7)

    assert first.shape == features.shape
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, features)


def test_feature_matrix_zero_noise_is_unchanged():
    features = np.arange(16, dtype=np.float32).reshape(2, 8)

    np.testing.assert_array_equal(feature_matrix_with_noise(features, sigma=0.0, seed=7), features)


def test_soft_iou_loss_is_differentiable():
    criterion = WeightedBinaryMaskLoss(
        pos_weight=None,
        loss_type="bce_dice_iou",
        dice_loss_weight=1.0,
        iou_loss_weight=0.25,
    )
    logits = torch.zeros((2, 16), requires_grad=True)
    targets = torch.zeros((2, 16))
    targets[:, :4] = 1.0

    loss = criterion(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_annulus_router_forward_and_training_smoke():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(6, 8)).astype(np.float32)
    masks = (rng.random(size=(6, 16)) > 0.75).astype(np.float32)
    shape_types = ("annulus", "circle", "annulus", "ellipse", "circle", "rectangle")
    model = AnnulusRouterOneStage(
        input_dim=8,
        output_dim=16,
        hidden_dims=(8,),
        dropout_rates=(0.0,),
        latent_grid_size=2,
        latent_channels=4,
        decoder_channels=(4, 2),
        router_hidden_dim=4,
    )
    result = fit_annulus_router(
        model,
        features,
        masks,
        shape_types,
        features,
        masks,
        epochs=1,
        batch_size=3,
        learning_rate=0.001,
        device=torch.device("cpu"),
        grid_size=4,
    )
    logits, router_logits = predict_annulus_router(
        model,
        features,
        device=torch.device("cpu"),
        training_result=result,
        return_router_logits=True,
    )
    assert logits.shape == masks.shape
    assert router_logits.shape == (len(features),)

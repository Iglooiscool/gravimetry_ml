import numpy as np
import pytest

from measurements import add_gaussian_noise


def test_complex_noise_uses_absolute_sigma_for_both_components():
    values = np.array([1.0 + 2.0j, 100.0 + 200.0j], dtype=np.complex128)
    sigma = 0.01
    expected_rng = np.random.default_rng(12)
    expected_real_noise = expected_rng.normal(0.0, sigma, size=values.shape)
    expected_imag_noise = expected_rng.normal(0.0, sigma, size=values.shape)
    expected = values + expected_real_noise + 1j * expected_imag_noise

    actual = add_gaussian_noise(values, sigma, np.random.default_rng(12))

    assert np.allclose(actual, expected)
    assert np.allclose(actual - values, expected - values)


def test_noise_is_not_scaled_by_signal_magnitude():
    small_values = np.zeros(20_000, dtype=np.float64)
    large_values = np.zeros(20_000, dtype=np.float64)
    sigma = 0.01

    small_noise = add_gaussian_noise(small_values, sigma, np.random.default_rng(4))
    large_noise = add_gaussian_noise(large_values, sigma, np.random.default_rng(4))

    assert np.array_equal(small_noise, large_noise)
    assert abs(float(np.std(small_noise)) - sigma) < 0.0003


def test_zero_noise_returns_the_input():
    values = np.array([1.0 + 2.0j, -3.0 + 0.5j])

    actual = add_gaussian_noise(values, 0.0, np.random.default_rng(5))

    assert np.array_equal(actual, values)


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError, match="sigma"):
        add_gaussian_noise(np.zeros(2), -0.01, np.random.default_rng(6))

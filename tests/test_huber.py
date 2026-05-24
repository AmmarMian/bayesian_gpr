import numpy as np
import pytest
from bayesian_gpr.huber import huber, huber_total


def test_huber_quadratic_core():
    delta = 1.0
    r = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    np.testing.assert_allclose(huber(r, delta), 0.5 * r**2)


def test_huber_linear_tails():
    delta = 1.0
    r = np.array([2.0, -2.0, 3.0])
    expected = delta * (np.abs(r) - 0.5 * delta)
    np.testing.assert_allclose(huber(r, delta), expected)


def test_huber_continuous_at_delta():
    delta = 1.5
    r_lo = np.array([delta - 1e-10])
    r_hi = np.array([delta + 1e-10])
    assert abs(huber(r_lo, delta).item() - huber(r_hi, delta).item()) < 1e-6


def test_huber_total_gaussian_case():
    # All residuals = 1 < delta=2, so each element = 0.5*1^2 = 0.5; 9 elements total
    D_obs = np.ones((3, 3))
    D_sim = np.zeros((3, 3))
    result = huber_total(D_obs, D_sim, delta=2.0)
    assert isinstance(result, float)
    assert result == pytest.approx(4.5)


def test_huber_zero_residual():
    r = np.zeros((5, 5))
    np.testing.assert_allclose(huber(r, delta=1.0), 0.0)

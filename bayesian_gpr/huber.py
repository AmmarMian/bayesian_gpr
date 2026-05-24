"""Huber loss function. Returns a cost value, not a log-density."""

import numpy as np


def huber(r: np.ndarray, delta: float) -> np.ndarray:
    """Element-wise Huber loss: r**2/2 if |r|<=delta, else delta*(|r| - delta/2)."""
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))


def huber_total(D_obs: np.ndarray, D_sim: np.ndarray, delta: float) -> float:
    """Sum of element-wise Huber loss on the residual.

    Returns a cost (not a log-density).
    """
    return float(huber(D_obs - D_sim, delta).sum())

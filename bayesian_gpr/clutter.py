"""Low-rank clutter generator.

Generates a random rank-k matrix L = U @ V.T to model horizontally-layered
ground clutter. This is a *generator* (one random draw), not a fitter.
Stella's MCMC code can write a proposal that moves U and V if she wants to
sample over clutter as part of the posterior.
"""
from __future__ import annotations

import numpy as np

from .scene import SceneGrid


def low_rank_clutter(
    grid: SceneGrid,
    rank: int = 3,
    snr_db: float | None = None,
    target_image: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a rank-k clutter matrix L = U @ V.T.

    Parameters
    ----------
    grid         : scene grid (provides Nt, Nx)
    rank         : number of rank-1 components (higher = richer clutter texture)
    snr_db       : if given together with target_image, scale L so that
                   SNR = 20·log10(max|signal| / rms(clutter)) = snr_db.
                   Positive → hyperbolas visible above clutter.
                   Negative → clutter rms exceeds signal peak (hyperbolas buried).
    target_image : reference image for SNR scaling (typically the clean B-scan)
    rng          : random generator; uses default_rng() if None

    Returns
    -------
    L : ndarray of shape (Nt, Nx)
    """
    if rng is None:
        rng = np.random.default_rng()
    U = rng.standard_normal((grid.Nt, rank))
    V = rng.standard_normal((grid.Nx, rank))
    L = U @ V.T
    if snr_db is not None and target_image is not None:
        signal_peak = np.max(np.abs(target_image))
        L_rms = np.linalg.norm(L, "fro") / np.sqrt(L.size)
        if L_rms > 0 and signal_peak > 0:
            # SNR = 20·log10(signal_peak / clutter_rms)
            # → scale clutter so rms(L) = signal_peak · 10^(-snr_db/20)
            clutter_rms_target = signal_peak * 10 ** (-snr_db / 20.0)
            L = L * (clutter_rms_target / L_rms)
    return L

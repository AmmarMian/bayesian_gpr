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
                   ||L||_F / ||target_image||_F = 10**(-snr_db/20).
                   Higher snr_db → weaker clutter relative to targets.
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
        target_norm = np.linalg.norm(target_image, "fro")
        L_norm = np.linalg.norm(L, "fro")
        if L_norm > 0 and target_norm > 0:
            L = L * (target_norm * 10 ** (-snr_db / 20.0) / L_norm)
    return L

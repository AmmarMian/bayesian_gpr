"""Visualisation utilities for GPR B-scans and scene ground truth."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .scene import Atom, Scene, SceneGrid, SoilParams, WaveletParams


def plot_bscan(
    image: np.ndarray,
    grid: SceneGrid,
    ax: plt.Axes | None = None,
    title: str = "",
    cmap: str = "gray",
) -> plt.Axes:
    """Plot a B-scan image with physical axis labels (x in m, t in ns)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(
        image,
        cmap=cmap,
        aspect="auto",
        extent=[0, grid.x_extent, grid.t_max * 1e9, 0],
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (ns)")
    if title:
        ax.set_title(title)
    return ax


def overlay_truth_hyperbolas(
    ax: plt.Axes,
    atoms: list[Atom],
    grid: SceneGrid,
    soil: SoilParams,
    wavelet: WaveletParams,
) -> None:
    """Overlay ground-truth hyperbola loci as dashed coloured lines."""
    colors = cm.tab10.colors
    x = grid.x_axis
    for i, atom in enumerate(atoms):
        p = atom.depth + atom.R
        t_apex = 2.0 * atom.depth / soil.v
        t_ns = ((2.0 / soil.v) * (np.sqrt(p**2 + (x - atom.x0) ** 2) - p) + t_apex) * 1e9
        mask = (t_ns >= 0) & (t_ns <= grid.t_max * 1e9)
        ax.plot(
            x[mask],
            t_ns[mask],
            "--",
            color=colors[i % len(colors)],
            linewidth=1.2,
            label=f"atom {i}: x={atom.x0:.2f}m d={atom.depth:.2f}m R={atom.R:.2f}m",
        )
    ax.legend(fontsize=7, loc="lower right")


def plot_scene(scene: Scene, *, observed: bool = True, **kwargs) -> plt.Axes:
    """Plot scene B-scan with ground-truth hyperbolas overlaid.

    Parameters
    ----------
    observed : if True, plot image+clutter+noise; if False, plot clean image only
    """
    image = scene.observed() if observed else scene.image
    if image is None:
        raise RuntimeError("Call scene.render() before plotting.")
    title = "Observed B-scan (image + clutter + noise)" if observed else "Clean B-scan"
    ax = plot_bscan(image, scene.grid, title=title, **kwargs)
    overlay_truth_hyperbolas(ax, scene.atoms, scene.grid, scene.soil, scene.wavelet)
    return ax

"""Visualisation utilities for GPR B-scans and scene ground truth."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

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
        aspect = grid.x_extent / (grid.t_max * 1e9)   # metres per nanosecond
        w = min(max(8.0, aspect * 3.0), 24.0)
        _, ax = plt.subplots(figsize=(w, 5))
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
    """Overlay ground-truth hyperbola loci as dashed coloured lines.

    Atoms are grouped by (depth, R, amplitude) — rebar grids all share the
    same values so they collapse to one colour and one legend entry.
    """
    colors = cm.tab20.colors
    x = grid.x_axis

    # Assign a color index to each unique physical type (depth, R, amplitude)
    type_color: dict[tuple, int] = {}
    type_label_done: set[tuple] = set()

    for atom in atoms:
        key = (round(atom.depth, 3), round(atom.R, 4), round(atom.amplitude, 3))
        if key not in type_color:
            type_color[key] = len(type_color) % len(colors)

    for atom in atoms:
        key = (round(atom.depth, 3), round(atom.R, 4), round(atom.amplitude, 3))
        color = colors[type_color[key]]

        p = atom.depth + atom.R
        t_apex = 2.0 * atom.depth / soil.v
        r_slant = np.sqrt(p**2 + (x - atom.x0) ** 2)
        t_ns = ((2.0 / soil.v) * (r_slant - p) + t_apex) * 1e9

        # Clip to where the signal is physically meaningful:
        # combined factor = geometric spreading × material absorption, relative to apex.
        spread_rel = np.sqrt(p / r_slant)                            # 1 at apex
        absorb_rel = np.exp(-2.0 * soil.alpha * (r_slant - p))      # 1 at apex
        visible = (spread_rel * absorb_rel) > 0.15                   # <15% of apex → don't draw

        mask = (t_ns >= 0) & (t_ns <= grid.t_max * 1e9) & visible

        label: str | None = None
        if key not in type_label_done:
            amp_str = f"{atom.amplitude:+.2f}"
            label = f"d={atom.depth:.2f}m  R={atom.R:.3f}m  amp={amp_str}"
            type_label_done.add(key)

        ax.plot(x[mask], t_ns[mask], "--", color=color,
                linewidth=0.8, alpha=0.6, label=label)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="lower right",
                  framealpha=0.7, ncol=max(1, len(handles) // 10))


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

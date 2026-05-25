"""Visualisation utilities for GPR B-scans and scene ground truth."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

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


def plot_scene_layout(scene: Scene, ax: plt.Axes | None = None) -> plt.Axes:
    """Cross-section view of the physical scene at true scale.

    Draws the ground surface, the GPR antenna sweep direction, and every buried
    object as a circle (or dot for rebar) at its true position and radius.
    Objects are colour-coded by fill material; voids are hatched.

    Parameters
    ----------
    scene : rendered or unrendered Scene — only atoms, grid, and soil are used
    ax    : existing Axes to draw into; a new figure is created if None
    """
    x_max = scene.grid.x_extent
    depth_max = scene.soil.v * scene.grid.t_max / 2.0   # max survey depth (m)
    margin = depth_max * 0.18                             # space above surface for antenna

    if ax is None:
        w = min(max(12.0, x_max * 0.35), 28.0)
        _, ax = plt.subplots(figsize=(w, 6))

    # ── Background ────────────────────────────────────────────────────────────
    ax.fill_between(
        [0, x_max], [0, 0], [depth_max, depth_max],
        color="#C4A265", alpha=0.35, zorder=0, label="_nolegend_",
    )
    ax.axhline(0, color="#5C3D1E", linewidth=2, zorder=2)

    # ── GPR antenna symbol ────────────────────────────────────────────────────
    y_ant = -margin * 0.55
    ax.annotate(
        "", xy=(x_max * 0.92, y_ant), xytext=(x_max * 0.08, y_ant),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5),
        zorder=4,
    )
    antenna_w, antenna_h = x_max * 0.04, margin * 0.45
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_max * 0.06, y_ant - antenna_h / 2), antenna_w, antenna_h,
        boxstyle="round,pad=0.01", facecolor="#EEEEEE", edgecolor="#333333",
        linewidth=1.0, zorder=5,
    ))
    ax.text(x_max * 0.08 + antenna_w / 2, y_ant - antenna_h / 2 - margin * 0.12,
            "GPR", ha="center", va="top", fontsize=7, color="#333333")

    # ── Buried objects ────────────────────────────────────────────────────────
    # Axes are not equal-aspect: use Ellipse with x-radius scaled to appear
    # circular on screen while keeping positions and relative sizes accurate.
    from matplotlib.patches import Ellipse

    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    shown_depth = depth_max + 1.3 * margin
    aspect_correction = (x_max / fig_w) / (shown_depth / fig_h)

    REBAR_THRESHOLD = 0.015

    def _atom_style(atom: Atom) -> dict:
        if atom.amplitude < 0:
            return dict(fc="white", ec="#CC0000", lw=1.2, hatch="///", zorder=3)
        if atom.eps_r >= 60:
            return dict(fc="#1E78C8", ec="#0A3A6E", lw=1.0, hatch=None, zorder=3)
        if atom.eps_r >= 2.5:
            return dict(fc="#909090", ec="#303030", lw=1.0, hatch=None, zorder=3)
        return dict(fc="#B8D8F0", ec="#1A5A9A", lw=1.0, hatch=None, zorder=3)

    for atom in scene.atoms:
        style = _atom_style(atom)
        if atom.R < REBAR_THRESHOLD:
            ax.plot(atom.x0, atom.depth, "o",
                    color=style["ec"], markersize=4, zorder=style["zorder"])
        else:
            ellipse = Ellipse(
                (atom.x0, atom.depth),
                width=2 * atom.R * aspect_correction,
                height=2 * atom.R,
                facecolor=style["fc"], edgecolor=style["ec"],
                linewidth=style["lw"], hatch=style["hatch"], zorder=style["zorder"],
            )
            ax.add_patch(ellipse)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(0, x_max)
    ax.set_ylim(depth_max + margin * 0.3, -margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title("Scene layout (cross-section)")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(fc="#B8D8F0", ec="#1A5A9A", label="air-filled pipe"),
        mpatches.Patch(fc="#909090", ec="#303030", label="PVC / concrete"),
        mpatches.Patch(fc="#1E78C8", ec="#0A3A6E", label="water-filled pipe"),
        mpatches.Patch(fc="white",   ec="#CC0000", hatch="///", label="void (−polarity)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#303030", markersize=5, label="rebar"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right", framealpha=0.8)

    return ax


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

"""Physical object presets that compile to lists of Atom.

Each function takes physical parameters and returns list[Atom].
Pass the result to Scene.add().

Typical eps_r values:
  Air (inside pipe/void): 1.0
  PVC / dry concrete:     3–6
  Moist concrete:         6–12
  Water-filled pipe:      ~80
  Steel / metal:          use high amplitude, eps_r doesn't affect curvature in this model
"""
from __future__ import annotations

import numpy as np

from .scene import Atom


def Pipe(
    x: float,
    depth: float,
    radius: float = 0.05,
    eps_r: float = 1.0,
    amplitude: float = 1.0,
) -> list[Atom]:
    """Single buried pipe or cylinder.

    Parameters
    ----------
    x        : horizontal position of pipe centre, m
    depth    : depth to pipe centre, m  (range: 0.05–3.0 m)
    radius   : pipe outer radius, m     (range: 0.01–0.20 m)
    eps_r    : pipe fill permittivity   (1=air, 3=PVC, 80=water)
    amplitude: reflection amplitude     (1.0 = strong reflector)
    """
    return [Atom(x0=x, depth=depth, R=radius, eps_r=eps_r, amplitude=amplitude)]


def Void(
    x: float,
    depth: float,
    radius: float = 0.10,
    amplitude: float = -0.5,
) -> list[Atom]:
    """Void or air-filled cavity.

    Negative default amplitude reflects the polarity flip at a soil-to-air interface
    (lower permittivity → negative reflection coefficient).

    Parameters
    ----------
    x        : horizontal position, m
    depth    : depth to cavity centre, m   (range: 0.05–3.0 m)
    radius   : cavity radius, m            (range: 0.05–0.50 m)
    amplitude: reflection amplitude; negative = polarity flip
    """
    return [Atom(x0=x, depth=depth, R=radius, eps_r=1.0, amplitude=amplitude)]


def RebarGrid(
    x_start: float,
    x_end: float,
    depth: float = 0.10,
    spacing: float = 0.15,
    radius: float = 0.01,
    amplitude: float = 0.5,
) -> list[Atom]:
    """Row of rebar bars at uniform horizontal spacing.

    Parameters
    ----------
    x_start  : leftmost rebar position, m
    x_end    : rightmost rebar position, m (inclusive)
    depth    : depth to rebar centre, m    (range: 0.05–0.40 m)
    spacing  : centre-to-centre spacing, m (range: 0.10–0.30 m)
    radius   : rebar radius, m             (range: 0.006–0.025 m)
    amplitude: reflection amplitude
    """
    xs = np.arange(x_start, x_end + spacing / 2.0, spacing)
    return [
        Atom(x0=float(x), depth=depth, R=radius, eps_r=1.0, amplitude=amplitude)
        for x in xs
    ]


def PointTarget(
    x: float,
    depth: float,
    R: float = 0.0,
    eps_r: float = 1.0,
    amplitude: float = 1.0,
) -> list[Atom]:
    """Escape hatch: raw Atom with explicit parameters.

    Use when the physical presets above don't fit. R=0 gives a pure point
    scatterer (no radius correction on the apex time).
    """
    return [Atom(x0=x, depth=depth, R=R, eps_r=eps_r, amplitude=amplitude)]

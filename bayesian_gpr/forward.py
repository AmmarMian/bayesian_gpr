"""Forward simulator: Ricker wavelet, per-atom hyperbola rendering, scene rendering."""

from __future__ import annotations

import numpy as np

from .scene import Atom, Scene, SceneGrid, SoilParams, WaveletParams


def ricker(tau: np.ndarray, f0: float) -> np.ndarray:
    """Ricker wavelet

    Parameters
    ----------
    tau : time offset, seconds (any shape)
    f0  : centre frequency, Hz

    Returns
    -------
    Wavelet values, same shape as tau.
    """
    u = (np.pi * f0 * tau) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)


def _hyperbola_locus(atom: Atom, x_axis: np.ndarray, v: float) -> np.ndarray:
    """Two-way travel-time locus for one finite-radius cylinder, shape (Nx,), seconds.

    For radius R=0 and x=x0, returns t_apex = 2*depth/v exactly.
    """
    p = atom.depth + atom.R  # effective depth including scatterer radius
    t_apex = 2.0 * atom.depth / v  # apex two-way travel time
    return (2.0 / v) * (np.sqrt(p**2 + (x_axis - atom.x0) ** 2) - p) + t_apex


def render_atom(
    atom: Atom,
    grid: SceneGrid,
    wavelet: WaveletParams,
    soil: SoilParams,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Render one atom's hyperbola contribution onto a (Nt, Nx) array.

    Attenuation per column x:
      - Cylindrical geometric spreading: sqrt(p / slant_range), normalised at apex.
      - Material absorption: exp(-2 * soil.alpha * slant_range) (two-way path).
    atom.amplitude sets the reflection coefficient at the apex.

    Parameters
    ----------
    out : if given, add in-place (useful for MCMC incremental updates)
    """
    if out is None:
        out = np.zeros((grid.Nt, grid.Nx))
    p = atom.depth + atom.R
    r_slant = np.sqrt(p**2 + (grid.x_axis - atom.x0) ** 2)         # (Nx,) one-way slant range
    spread = np.sqrt(p / r_slant)                                     # cylindrical spreading (1 at apex)
    absorb = np.exp(-2.0 * soil.alpha * r_slant)                      # two-way absorption (depth-dependent)
    t_locus = _hyperbola_locus(atom, grid.x_axis, soil.v)            # (Nx,)
    tau = grid.t_axis[:, None] - t_locus[None, :]                    # (Nt, Nx)
    out += atom.amplitude * (spread * absorb)[None, :] * ricker(tau, wavelet.f0)
    return out


def render_atoms(scene: Scene, method: str = "sum") -> np.ndarray:
    """Render all atoms into a (Nt, Nx) B-scan array.

    Parameters
    ----------
    method : 'sum'  — exact per-atom summation (default; use in MCMC)
             'conv' — FFT convolution approximation (faster for large scenes;
                      approximate because hyperbola curvature varies with depth)

    The 'sum' method is exact and supports incremental MCMC updates via
    render_atom(..., out=cached_image). The 'conv' method is an approximation
    that becomes less accurate when atoms span a wide range of depths.
    """
    if method == "conv":
        return render_via_conv(scene)
    out = np.zeros((scene.grid.Nt, scene.grid.Nx))
    for atom in scene.atoms:
        render_atom(atom, scene.grid, scene.wavelet, scene.soil, out=out)
    return out


def render_via_conv(scene: Scene) -> np.ndarray:
    """FFT-convolution approximation of the forward model.

    The idea: 2D convolution (C ⊛ H)[i,j] = Σ C[k,l]·H[i-k, j-l] pastes a
    copy of template H at every nonzero position in the coefficient map C.
    Build H with its apex at pixel (0, 0), put deltas in C at each atom's
    apex pixel, FFT-convolve.

    Known limitations (use render_atoms(method='sum') when these matter):
    - The template is built with x0=0, so only the right wing (x ≥ 0) of the
      hyperbola is captured. Atoms far from x=0 will miss their left wing.
    - Curvature approximation: H uses depth=0 (p=R); real atoms at depth d
      have p=d+R, giving flatter hyperbolas. Error grows with depth spread.
    - Zero-padding avoids circular wrap-around but doubles memory usage.

    Grouping: one template per unique R value (eps_r does not affect shape).
    """
    if not scene.atoms:
        return np.zeros((scene.grid.Nt, scene.grid.Nx))

    grid = scene.grid
    wavelet = scene.wavelet
    soil = scene.soil
    Nt, Nx = grid.Nt, grid.Nx
    out = np.zeros((Nt, Nx))

    # Group atoms by R so they share the same template shape
    groups: dict[float, list[Atom]] = {}
    for atom in scene.atoms:
        groups.setdefault(atom.R, []).append(atom)

    for R, atoms_grp in groups.items():
        # Build template on a SYMMETRIC x-axis centred at 0:
        #   x_sym = [-Nx*dx, ..., -dx, 0, dx, ..., (Nx-1)*dx]  (2*Nx points)
        # so both wings of the hyperbola are captured.
        # Apex sits at column Nx (where x_sym = 0); roll left by Nx → apex at col 0.
        x_sym = (np.arange(2 * Nx) - Nx) * grid.dx  # (2*Nx,)
        locus_sym = (2.0 / soil.v) * (np.sqrt(R**2 + x_sym**2) - R)  # (2*Nx,)
        tau_sym = grid.t_axis[:, None] - locus_sym[None, :]  # (Nt, 2*Nx)
        H_full = ricker(tau_sym, wavelet.f0)  # apex at col Nx
        H_rolled = np.roll(H_full, -Nx, axis=1)  # apex at col 0

        # Zero-pad in time to 2*Nt to avoid time-axis wrap-around
        H_pad = np.zeros((2 * Nt, 2 * Nx))
        H_pad[:Nt, :] = H_rolled

        # Coefficient map: delta at each atom's (it_apex, ix0)
        C_pad = np.zeros((2 * Nt, 2 * Nx))
        for atom in atoms_grp:
            t_apex = 2.0 * atom.depth / soil.v
            it = int(round((t_apex / grid.t_max) * (Nt - 1)))
            ix = int(round((atom.x0 / grid.x_extent) * (Nx - 1)))
            C_pad[max(0, min(2 * Nt - 1, it)), max(0, min(2 * Nx - 1, ix))] += atom.amplitude

        conv = np.real(np.fft.ifft2(np.fft.fft2(H_pad) * np.fft.fft2(C_pad)))
        out += conv[:Nt, :Nx]

    return out

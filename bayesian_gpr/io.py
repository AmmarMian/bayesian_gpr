"""Scene serialisation to/from .npz files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .scene import Atom, Scene, SceneGrid, SoilParams, WaveletParams


def save_scene(scene: Scene, path: str | Path) -> None:
    """Save scene to a .npz file.

    Saves grid parameters, atom arrays, and any rendered arrays (image,
    clutter, noise) present on the scene object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "x_extent": np.float64(scene.grid.x_extent),
        "t_max": np.float64(scene.grid.t_max),
        "dx": np.float64(scene.grid.dx),
        "dt": np.float64(scene.grid.dt),
        "f0": np.float64(scene.wavelet.f0),
        "eps_soil": np.float64(scene.soil.eps_r),
        "alpha_soil": np.float64(scene.soil.alpha),
        "rng_seed": np.int64(-1 if scene.rng_seed is None else scene.rng_seed),
        "atoms_x0": np.array([a.x0 for a in scene.atoms], dtype=np.float64),
        "atoms_depth": np.array([a.depth for a in scene.atoms], dtype=np.float64),
        "atoms_R": np.array([a.R for a in scene.atoms], dtype=np.float64),
        "atoms_eps_r": np.array([a.eps_r for a in scene.atoms], dtype=np.float64),
        "atoms_amplitude": np.array([a.amplitude for a in scene.atoms], dtype=np.float64),
    }
    for key in ("image", "clutter", "noise"):
        arr = getattr(scene, key)
        if arr is not None:
            arrays[key] = arr

    np.savez(path, **arrays)


def load_scene(path: str | Path) -> Scene:
    """Load a scene from a .npz file."""
    data = np.load(path)
    grid = SceneGrid(
        x_extent=float(data["x_extent"]),
        t_max=float(data["t_max"]),
        dx=float(data["dx"]),
        dt=float(data["dt"]),
    )
    wavelet = WaveletParams(f0=float(data["f0"]))
    soil = SoilParams(
        eps_r=float(data["eps_soil"]),
        alpha=float(data["alpha_soil"]) if "alpha_soil" in data else 0.0,
    )
    rng_seed_raw = int(data["rng_seed"])
    rng_seed = None if rng_seed_raw == -1 else rng_seed_raw

    atoms = [
        Atom(x0=float(x), depth=float(d), R=float(r), eps_r=float(e), amplitude=float(a))
        for x, d, r, e, a in zip(
            data["atoms_x0"],
            data["atoms_depth"],
            data["atoms_R"],
            data["atoms_eps_r"],
            data["atoms_amplitude"],
        )
    ]

    scene = Scene(grid=grid, wavelet=wavelet, soil=soil, atoms=atoms, rng_seed=rng_seed)
    for key in ("image", "clutter", "noise"):
        if key in data:
            setattr(scene, key, data[key])
    return scene


def load_geolithe(path: str | Path) -> np.ndarray:
    """Load the Geolithe IRADAR dataset as a (Nt, Nx) ndarray.

    Not yet implemented. The tarball is at:
        GPR-robust-inversion/data/IRADAR__00H.tar.gz
    Untar it and implement this function to parse the resulting files.
    """
    raise NotImplementedError(
        "Geolithe data loader not yet implemented. "
        "Untar GPR-robust-inversion/data/IRADAR__00H.tar.gz and implement this function."
    )

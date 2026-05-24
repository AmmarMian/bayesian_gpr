"""Core dataclasses: SceneGrid, WaveletParams, SoilParams, Atom, Scene."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

_C = 299_792_458.0  # speed of light, m/s


@dataclass(frozen=True)
class SceneGrid:
    """Spatial and temporal discretisation of the B-scan domain."""

    x_extent: float  # m
    t_max: float     # s
    dx: float        # m
    dt: float        # s

    @property
    def Nx(self) -> int:
        return round(self.x_extent / self.dx)

    @property
    def Nt(self) -> int:
        return round(self.t_max / self.dt)

    @property
    def x_axis(self) -> np.ndarray:
        return np.linspace(0.0, self.x_extent, self.Nx)

    @property
    def t_axis(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.Nt)


@dataclass(frozen=True)
class WaveletParams:
    """Source wavelet parameters."""

    f0: float  # centre frequency, Hz


@dataclass(frozen=True)
class SoilParams:
    """Soil electromagnetic properties."""

    eps_r: float  # relative permittivity

    @property
    def v(self) -> float:
        """EM wave speed in soil, m/s."""
        return _C / math.sqrt(self.eps_r)


@dataclass
class Atom:
    """A single buried scatterer contributing one hyperbola to the B-scan."""

    x0: float        # horizontal position, m
    depth: float     # depth to centre of scatterer, m
    R: float         # scatterer radius, m (0 for point target)
    eps_r: float     # scatterer permittivity (used as metadata; curvature set by soil)
    amplitude: float  # reflection amplitude; negative = polarity flip


@dataclass
class Scene:
    """A complete GPR scene: grid, soil, atoms, and optional rendered arrays."""

    grid: SceneGrid
    wavelet: WaveletParams
    soil: SoilParams
    atoms: list[Atom] = field(default_factory=list)
    image: np.ndarray | None = None    # noise-free B-scan, set by render()
    clutter: np.ndarray | None = None  # low-rank background, set by add_clutter()
    noise: np.ndarray | None = None    # additive noise, set by add_noise()
    rng_seed: int | None = None

    def add(self, atoms: Atom | list[Atom]) -> Scene:
        """Add one atom or a list of atoms to the scene."""
        if isinstance(atoms, Atom):
            self.atoms.append(atoms)
        else:
            self.atoms.extend(atoms)
        return self

    def render(self) -> np.ndarray:
        """Render noise-free B-scan from current atom list. Stores in self.image."""
        from . import forward
        self.image = forward.render_atoms(self)
        return self.image

    def add_clutter(self, kind: str = "low_rank", **kwargs) -> Scene:
        """Add background clutter. kind='low_rank' supported."""
        from . import clutter as _clutter
        rng = np.random.default_rng(self.rng_seed)
        if kind == "low_rank":
            self.clutter = _clutter.low_rank_clutter(
                self.grid,
                rank=kwargs.get("rank", 3),
                snr_db=kwargs.get("snr_db"),
                target_image=self.image,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown clutter kind: {kind!r}. Supported: 'low_rank'")
        return self

    def add_noise(self, sigma: float) -> Scene:
        """Add i.i.d. Gaussian noise with standard deviation sigma."""
        rng = np.random.default_rng(self.rng_seed)
        self.noise = rng.standard_normal((self.grid.Nt, self.grid.Nx)) * sigma
        return self

    def observed(self) -> np.ndarray:
        """Return image + clutter + noise (whatever has been added)."""
        base = self.image if self.image is not None else np.zeros((self.grid.Nt, self.grid.Nx))
        out = base.copy()
        if self.clutter is not None:
            out = out + self.clutter
        if self.noise is not None:
            out = out + self.noise
        return out

    def save(self, path: str | Path) -> None:
        """Save scene to a .npz file."""
        from . import io
        io.save_scene(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Scene:
        """Load scene from a .npz file."""
        from . import io
        return io.load_scene(path)

    def plot(self, *, observed: bool = True, **kwargs):
        """Plot B-scan with ground-truth hyperbolas overlaid."""
        from . import viz
        viz.plot_scene(self, observed=observed, **kwargs)

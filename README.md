# bayesian-gpr

GPR physics building blocks for Bayesian MCMC inversion.

**Scope**: forward simulator + scene builder + clutter generator + Huber norm.
The MCMC sampler, priors, and likelihood are the caller's responsibility.

## Install

```bash
uv sync --extra dev      # recommended
# or: pip install -e ".[dev]"
```

Requires Python ≥ 3.11.

## Quickstart

```python
from bayesian_gpr import Scene, SceneGrid, WaveletParams, SoilParams, presets

scene = Scene(
    grid    = SceneGrid(x_extent=50.0, t_max=40e-9, dx=0.01, dt=0.02e-9),
    wavelet = WaveletParams(f0=400e6),
    soil    = SoilParams(eps_r=9.0, alpha=0.35),   # moist soil at 400 MHz
    rng_seed = 42,
)
scene.add(presets.Pipe(x=4.0,  depth=0.40, radius=0.05, eps_r=1.0))   # air-filled
scene.add(presets.Pipe(x=9.5,  depth=0.65, radius=0.10, eps_r=80.0))  # water-filled
scene.add(presets.Void(x=26.0, depth=0.80, radius=0.15))
scene.add(presets.RebarGrid(x_start=38.0, x_end=38.4, depth=0.10, spacing=0.20))

scene.render()                                      # → scene.image  (Nt × Nx)
scene.add_clutter(kind="low_rank", rank=5, snr_db=6.0)
scene.add_noise(sigma=0.02)
scene.save("my_scene.npz")
scene.plot(observed=True)                           # B-scan + truth hyperbola overlays
```

Run the full demo (saves four PNGs and an `.npz` bundle under `data/`):

```bash
python scripts/make_demo_scene.py
```

## Using from MCMC

```python
from bayesian_gpr import Scene, render_atoms, huber_total
from bayesian_gpr.scene import Atom

# Load once before the loop
scene = Scene.load("data/demo_pipes.npz")   # soil.alpha is preserved on load
D_obs = scene.observed()

# Inside MH loop — given a proposed atom list X (list[Atom]):
D_sim = render_atoms(Scene(grid=scene.grid, wavelet=scene.wavelet,
                           soil=scene.soil, atoms=X))
cost = huber_total(D_obs, D_sim, delta=0.05)
# cost is a scalar — embed it in log p(y|X) however the model requires
```

For single-atom update moves (birth/death/shift), re-render only the changed atom
in-place to avoid recomputing everything:

```python
from bayesian_gpr.forward import render_atom
from bayesian_gpr.scene import Atom
import numpy as np

# Remove old atom, add new atom — O(Nt × Nx) instead of O(k × Nt × Nx)
D_cached = D_current.copy()
old_atom_neg = Atom(old_atom.x0, old_atom.depth, old_atom.R,
                    old_atom.eps_r, -old_atom.amplitude)
render_atom(old_atom_neg, grid, wavelet, soil, out=D_cached)  # subtract
render_atom(new_atom,     grid, wavelet, soil, out=D_cached)  # add
```

## Physical parameters

### Soil (`SoilParams`)

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `eps_r` | 9.0 | 4–30 | dry sand → wet clay; sets v = c / √eps_r |
| `alpha` | 0.0 | 0–1.5 Np/m | one-way EM attenuation coefficient |

`alpha = 0` is lossless (backward-compatible default). Typical values at 400 MHz:

| Soil type | alpha (Np/m) | alpha (dB/m) |
|---|---|---|
| Dry sand | 0.01 | 0.09 |
| Moist soil | 0.1–0.3 | 0.9–2.6 |
| Wet clay | 0.5–1.5 | 4–13 |

The render applies per-column attenuation `√(p / r) · exp(−2·alpha·r)` where
`r = √(p² + (x − x₀)²)` is the one-way slant range and `p = depth + R`.
Both factors are evaluated at each antenna position, so deeper and off-axis
objects are naturally fainter. `atom.amplitude` sets the reflection strength
at the apex before propagation losses.

`alpha` is saved and restored by `scene.save()` / `Scene.load()`.

### Antenna and grid

| Parameter | Default | Range |
|---|---|---|
| `f0` (antenna) | 400 MHz | 100 MHz – 2.6 GHz |
| `t_max` | 40 ns | covers 2 m depth at v = 0.1 m/ns |
| `dx` | 0.01 m | Nyquist-safe for 400 MHz in eps_r = 9 soil |
| `dt` | 0.02 ns | |

### Presets

| Preset | Key params | Default radius | `eps_r` guide |
|---|---|---|---|
| `Pipe` | x, depth, radius, eps_r | 0.05 m | 1 = air · 3 = PVC · 80 = water |
| `Void` | x, depth, radius | 0.10 m | always 1; amplitude is negative |
| `RebarGrid` | x_start, x_end, depth, spacing | 0.01 m | one atom per bar |
| `PointTarget` | x, depth, R, eps_r | 0.0 | raw Atom, no preset logic |

## Clutter (`add_clutter`)

`scene.add_clutter(kind="low_rank", rank=k, snr_db=s)` generates `L = U @ Vᵀ`
with Gaussian U, V, scaled so that:

```
SNR = 20·log10( max|signal| / rms(clutter) ) = snr_db
```

Positive SNR → hyperbola apices stand above the clutter noise floor.
Negative SNR → clutter RMS exceeds the signal peak (targets buried).
Typical values: 3–10 dB for a clean survey, −5 to 5 dB in cluttered ground.

## Rendering methods

`render_atoms(scene, method='sum')` — exact per-atom summation; default and
recommended for MCMC (supports incremental updates via `render_atom`).

`render_atoms(scene, method='conv')` — FFT convolution approximation, faster
for large scenes but approximate: one depth-zero template cannot match atoms
at varying depths.

## Visualisation

```python
from bayesian_gpr import plot_scene_layout

plot_scene_layout(scene)   # cross-section: ground surface, GPR sweep, objects at true radii
scene.plot(observed=True)  # B-scan with truth hyperbola overlays clipped to visible signal
```

`plot_scene_layout` colour-codes objects by fill: light blue = air, dark blue = water,
grey = PVC/concrete, red hatch = void (negative polarity), dots = rebar.

# bayesian-gpr

GPR physics building blocks for Bayesian MCMC inversion.


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
    grid    = SceneGrid(x_extent=2.0, t_max=40e-9, dx=0.02, dt=0.1e-9),
    wavelet = WaveletParams(f0=400e6),
    soil    = SoilParams(eps_r=9.0),
    rng_seed = 42,
)
scene.add(presets.Pipe(x=0.5,  depth=0.3, radius=0.05, eps_r=1.0))   # air-filled
scene.add(presets.Pipe(x=1.2,  depth=0.6, radius=0.08, eps_r=80.0))  # water-filled
scene.add(presets.Void(x=0.8,  depth=1.0, radius=0.15))
scene.add(presets.RebarGrid(x_start=1.5, x_end=1.9, depth=0.1, spacing=0.10))

scene.render()                                     # → scene.image  (Nt × Nx)
scene.add_clutter(kind="low_rank", rank=3, snr_db=8.0)
scene.add_noise(sigma=0.03)
scene.save("my_scene.npz")
scene.plot(observed=True)                          # B-scan + truth hyperbolas
```

Run the full demo (saves `data/demo_pipes.{npz,png}`):

```bash
python scripts/make_demo_scene.py
```

## Using from MCMC

```python
from bayesian_gpr import Scene, render_atoms, huber_total

scene = Scene.load("data/demo_pipes.npz")
D_obs = scene.observed()

# Inside MH loop — given a proposed atom list X (list[Atom]):
D_sim = render_atoms(Scene(grid=scene.grid, wavelet=scene.wavelet,
                           soil=scene.soil, atoms=X))
cost = huber_total(D_obs, D_sim, delta=0.05)
# cost is a scalar cost value — you decide how to embed it in log p(y|X)
```

## Physical parameter defaults (400 MHz infrastructure scenario)

| Parameter | Default | Plausible range |
|---|---|---|
| f0 (antenna) | 400 MHz | 100 MHz – 2.6 GHz |
| eps_soil | 9.0 | 4–30 (dry → wet soil) |
| v (wave speed) | 0.100 m/ns | derived: c/√eps_soil |
| t_max | 40 ns | covers 2 m depth at v=0.1 m/ns |
| Pipe radius | 0.05 m | 0.01–0.20 m |
| Rebar radius | 0.01 m | 0.006–0.025 m |
| Void radius | 0.10 m | 0.05–0.50 m |
| Material absorbtion (alpha) | 0 | ~|

Typical `eps_r` values for object fill: air 1.0 · PVC/dry concrete 3–6 · water ~80.

## Rendering methods

`render_atoms(scene, method='sum')` — exact, per-atom summation. Use in MCMC.

`render_atoms(scene, method='conv')` or `render_via_conv(scene)` — FFT convolution approximation, faster for large scenes. Approximate because hyperbola curvature depends on depth (p = depth + R), so a single centred template cannot represent atoms at different depths exactly.


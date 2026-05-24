"""End-to-end demo: build a GPR scene, render, add clutter and noise, save, plot.

Run from the repo root:
    python scripts/make_demo_scene.py

Outputs:
    data/demo_pipes.npz  — scene bundle (grid, atoms, image, clutter, noise)
    data/demo_pipes.png  — B-scan plot with ground-truth hyperbolas overlaid

Survey: 50 m scan, 40 ns time window, 400 MHz antenna, moist soil (eps_r=9).
Grid: 5000 antenna positions (dx=1 cm) × 2000 time samples (dt=20 ps).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_gpr import Scene, SceneGrid, WaveletParams, SoilParams, presets
from bayesian_gpr.viz import plot_bscan

# ── Scene definition ─────────────────────────────────────────────────────────
scene = Scene(
    grid=SceneGrid(x_extent=50.0, t_max=40e-9, dx=0.01, dt=0.02e-9),
    wavelet=WaveletParams(f0=400e6),
    soil=SoilParams(eps_r=9.0, alpha=0.35),  # 0.35 Np/m ≈ 3 dB/m, moist/clay soil at 400 MHz
    rng_seed=42,
)

# ── Objects  (15 presets → 18 atoms total) ────────────────────────────────────

# Shallow air-filled utility pipe
scene.add(presets.Pipe(x=4.0, depth=0.40, radius=0.05, eps_r=1.0))

# Water mains — large, strong reflector
scene.add(presets.Pipe(x=9.5, depth=0.65, radius=0.10, eps_r=80.0))

# PVC conduit (low contrast)
scene.add(presets.Pipe(x=14.0, depth=0.30, radius=0.03, eps_r=3.0))

# Three closely-spaced pipes (overlapping hyperbolas)
scene.add(presets.Pipe(x=20.0, depth=0.35, radius=0.05, eps_r=1.0))  # air
scene.add(presets.Pipe(x=20.6, depth=0.40, radius=0.04, eps_r=3.0))  # PVC
scene.add(presets.Pipe(x=21.3, depth=0.38, radius=0.05, eps_r=80.0))  # water-filled

# Void / subsurface cavity (negative polarity)
scene.add(presets.Void(x=26.0, depth=0.80, radius=0.15))

# Deep service tunnel — shallower curvature, strongly attenuated
scene.add(presets.Pipe(x=30.0, depth=1.50, radius=0.12, eps_r=1.0))

# Small gas pipe, very shallow
scene.add(presets.Pipe(x=34.5, depth=0.22, radius=0.025, eps_r=1.0))

# Three isolated rebar bars (single concrete tie, not a full slab)
scene.add(presets.RebarGrid(x_start=38.0, x_end=38.4, depth=0.10, spacing=0.20))

# Large-diameter sewer pipe
scene.add(presets.Pipe(x=42.0, depth=0.70, radius=0.15, eps_r=1.0))

# Large void — subsidence feature
scene.add(presets.Void(x=46.0, depth=1.00, radius=0.25))

# Deep water pipe at the far end
scene.add(presets.Pipe(x=49.0, depth=0.55, radius=0.09, eps_r=80.0))

# ── Render ────────────────────────────────────────────────────────────────────
scene.render()
scene.add_clutter(kind="low_rank", rank=3, snr_db=20)
scene.add_noise(sigma=0.02)

# ── Save ──────────────────────────────────────────────────────────────────────
Path("data").mkdir(exist_ok=True)
scene.save("data/demo_pipes.npz")

scene.plot(observed=True)
plt.savefig("data/demo_pipes.png", dpi=150, bbox_inches="tight")
plt.close("all")

# Clean view — observed data only, no overlay
plot_bscan(scene.observed(), scene.grid, title="Observed B-scan (image + clutter + noise)")
plt.savefig("data/demo_pipes_data.png", dpi=150, bbox_inches="tight")
plt.close("all")

print("Saved data/demo_pipes.npz and data/demo_pipes.png")
print(f"  Grid: {scene.grid.Nt} time samples × {scene.grid.Nx} antenna positions")
print(f"  Atoms: {len(scene.atoms)}")
print("  Soil wave speed: {:.4f} m/ns".format(scene.soil.v * 1e-9))
print(
    "  Soil attenuation: {:.2f} Np/m ({:.2f} dB/m)".format(
        scene.soil.alpha, scene.soil.alpha * 8.686
    )
)

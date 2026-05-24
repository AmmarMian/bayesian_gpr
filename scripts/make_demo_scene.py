"""End-to-end demo: build a GPR scene, render, add clutter and noise, save, plot.

Run from the repo root:
    python scripts/make_demo_scene.py

Outputs:
    data/demo_pipes.npz  — scene bundle (grid, atoms, image, clutter, noise)
    data/demo_pipes.png  — B-scan plot with ground-truth hyperbolas overlaid
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no display required
import matplotlib.pyplot as plt

from bayesian_gpr import Scene, SceneGrid, WaveletParams, SoilParams, presets

# ── Scene definition ─────────────────────────────────────────────────────────
# 400 MHz survey, typical moist soil (eps_r=9 → v ≈ 0.10 m/ns)
# Grid: 100 antenna positions × 400 time samples (2 m scan, 40 ns window)

scene = Scene(
    grid=SceneGrid(x_extent=2.0, t_max=40e-9, dx=0.02, dt=0.1e-9),
    wavelet=WaveletParams(f0=400e6),
    soil=SoilParams(eps_r=9.0),
    rng_seed=42,
)

# Air-filled PVC pipe, 5 cm radius, 0.3 m depth
scene.add(presets.Pipe(x=0.4, depth=0.3, radius=0.05, eps_r=1.0))

# Water-filled pipe, 8 cm radius, 0.6 m depth — shallower curvature (same soil speed)
scene.add(presets.Pipe(x=1.2, depth=0.6, radius=0.08, eps_r=80.0))

# Rebar grid at 0.1 m depth, 10 cm spacing
scene.add(presets.RebarGrid(x_start=1.6, x_end=1.9, depth=0.1, spacing=0.10))

# Void / cavity at 1.0 m depth, negative amplitude (polarity flip)
scene.add(presets.Void(x=0.8, depth=1.0, radius=0.15))

# ── Render ────────────────────────────────────────────────────────────────────
scene.render()                                    # clean B-scan
scene.add_clutter(kind="low_rank", rank=3, snr_db=8.0)  # add background clutter
scene.add_noise(sigma=0.03)                       # add measurement noise

# ── Save ──────────────────────────────────────────────────────────────────────
Path("data").mkdir(exist_ok=True)
scene.save("data/demo_pipes.npz")
scene.plot(observed=True)
plt.savefig("data/demo_pipes.png", dpi=150, bbox_inches="tight")
plt.close("all")

print("Saved data/demo_pipes.npz and data/demo_pipes.png")
print(f"  Grid: {scene.grid.Nt} time samples × {scene.grid.Nx} antenna positions")
print(f"  Atoms: {len(scene.atoms)}")
print("  Soil wave speed: {:.4f} m/ns".format(scene.soil.v * 1e-9))

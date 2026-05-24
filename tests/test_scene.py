import tempfile
from pathlib import Path
import numpy as np
from bayesian_gpr import Scene, SceneGrid, WaveletParams, SoilParams, presets
from bayesian_gpr.io import save_scene, load_scene

GRID = SceneGrid(x_extent=2.0, t_max=40e-9, dx=0.02, dt=0.1e-9)
WAVELET = WaveletParams(f0=400e6)
SOIL = SoilParams(eps_r=9.0)


def make_scene() -> Scene:
    s = Scene(grid=GRID, wavelet=WAVELET, soil=SOIL, rng_seed=0)
    s.add(presets.Pipe(x=0.5, depth=0.3))
    s.add(presets.Void(x=1.0, depth=0.5))
    s.add(presets.RebarGrid(x_start=1.4, x_end=1.8, depth=0.1, spacing=0.2))
    return s


def test_pipe_returns_one_atom():
    assert len(presets.Pipe(x=0.5, depth=0.3)) == 1


def test_pipe_position():
    atoms = presets.Pipe(x=0.7, depth=0.4)
    assert atoms[0].x0 == 0.7
    assert atoms[0].depth == 0.4


def test_void_negative_amplitude():
    atoms = presets.Void(x=1.0, depth=0.5)
    assert atoms[0].amplitude < 0


def test_rebar_count():
    # x from 0.0 to 1.0 with spacing 0.25 → 5 bars: 0, 0.25, 0.5, 0.75, 1.0
    atoms = presets.RebarGrid(x_start=0.0, x_end=1.0, spacing=0.25)
    assert len(atoms) == 5


def test_scene_add_atoms():
    s = make_scene()
    assert len(s.atoms) > 0


def test_scene_render_shape():
    s = make_scene()
    img = s.render()
    assert img.shape == (GRID.Nt, GRID.Nx)
    assert s.image is img


def test_scene_observed_differs_from_clean():
    s = make_scene()
    s.render()
    s.add_clutter(kind="low_rank", rank=2, snr_db=10.0)
    s.add_noise(sigma=0.02)
    obs = s.observed()
    assert obs.shape == (GRID.Nt, GRID.Nx)
    assert not np.allclose(obs, s.image)


def test_save_load_roundtrip():
    s = make_scene()
    s.render()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test_scene.npz"
        save_scene(s, p)
        s2 = load_scene(p)
    assert len(s2.atoms) == len(s.atoms)
    assert s2.grid.x_extent == s.grid.x_extent
    assert s2.wavelet.f0 == s.wavelet.f0
    np.testing.assert_allclose(s2.image, s.image)


def test_save_load_preserves_clutter_noise():
    s = make_scene()
    s.render()
    s.add_clutter(kind="low_rank", rank=2)
    s.add_noise(sigma=0.01)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test_full.npz"
        save_scene(s, p)
        s2 = load_scene(p)
    np.testing.assert_allclose(s2.clutter, s.clutter)
    np.testing.assert_allclose(s2.noise, s.noise)

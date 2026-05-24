import numpy as np
from bayesian_gpr.forward import ricker, render_atoms, render_via_conv
from bayesian_gpr.scene import Atom, Scene, SceneGrid, SoilParams, WaveletParams

GRID = SceneGrid(x_extent=2.0, t_max=40e-9, dx=0.02, dt=0.1e-9)
WAVELET = WaveletParams(f0=400e6)
SOIL = SoilParams(eps_r=9.0)


def test_ricker_peak_at_zero():
    tau = np.linspace(-5e-9, 5e-9, 10000)
    r = ricker(tau, 400e6)
    assert abs(tau[np.argmax(r)]) < 0.2e-9


def test_ricker_analytic_zeros():
    f0 = 400e6
    # Zero when 1-2u=0 → u=0.5 → (pi*f0*tau)^2=0.5 → tau=1/(pi*f0*sqrt(2))
    tau_zero = 1.0 / (np.pi * f0 * np.sqrt(2.0))
    val = ricker(np.array([tau_zero]), f0)[0]
    assert abs(val) < 1e-10


def test_atom_apex_timing_no_radius():
    depth = 0.5
    atom = Atom(x0=1.0, depth=depth, R=0.0, eps_r=1.0, amplitude=1.0)
    scene = Scene(grid=GRID, wavelet=WAVELET, soil=SOIL, atoms=[atom])
    image = render_atoms(scene)
    col = np.argmin(np.abs(GRID.x_axis - 1.0))
    t_peak = GRID.t_axis[np.argmax(image[:, col])]
    t_apex_expected = 2.0 * depth / SOIL.v
    assert abs(t_peak - t_apex_expected) < 2 * GRID.dt


def test_atom_apex_timing_with_radius():
    depth, R = 0.3, 0.05
    atom = Atom(x0=1.0, depth=depth, R=R, eps_r=1.0, amplitude=1.0)
    scene = Scene(grid=GRID, wavelet=WAVELET, soil=SOIL, atoms=[atom])
    image = render_atoms(scene)
    col = np.argmin(np.abs(GRID.x_axis - 1.0))
    t_peak = GRID.t_axis[np.argmax(image[:, col])]
    t_apex_expected = 2.0 * depth / SOIL.v
    assert abs(t_peak - t_apex_expected) < 2 * GRID.dt



def test_render_energy_nonzero():
    atom = Atom(x0=1.0, depth=0.5, R=0.0, eps_r=1.0, amplitude=1.0)
    scene = Scene(grid=GRID, wavelet=WAVELET, soil=SOIL, atoms=[atom])
    assert np.sum(render_atoms(scene) ** 2) > 0


def test_empty_scene_returns_zeros():
    scene = Scene(grid=GRID, wavelet=WAVELET, soil=SOIL, atoms=[])
    img = render_atoms(scene)
    assert img.shape == (GRID.Nt, GRID.Nx)
    np.testing.assert_allclose(img, 0.0)


def test_render_via_conv_shape_and_nonzero():
    atoms = [
        Atom(x0=0.5, depth=0.4, R=0.0, eps_r=1.0, amplitude=1.0),
        Atom(x0=1.5, depth=0.4, R=0.0, eps_r=1.0, amplitude=0.7),
    ]
    scene = Scene(grid=GRID, wavelet=WAVELET, soil=SOIL, atoms=atoms)
    img = render_via_conv(scene)
    assert img.shape == (GRID.Nt, GRID.Nx)
    assert np.sum(img**2) > 0


def test_render_atoms_method_conv():
    # method='conv' is accepted and returns correct shape with non-zero energy.
    # Conv is an approximation with known limitations (only right wing of hyperbola
    # at x=0; works best for atoms near x=0 with small depth spread).
    atoms = [Atom(x0=0.0, depth=0.3, R=0.0, eps_r=1.0, amplitude=1.0)]
    scene = Scene(grid=GRID, wavelet=WAVELET, soil=SOIL, atoms=atoms)
    img_conv = render_atoms(scene, method="conv")
    assert img_conv.shape == (GRID.Nt, GRID.Nx)
    assert np.sum(img_conv**2) > 0

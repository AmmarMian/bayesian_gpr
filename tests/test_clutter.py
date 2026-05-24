import numpy as np
from bayesian_gpr.clutter import low_rank_clutter
from bayesian_gpr.scene import SceneGrid

GRID = SceneGrid(x_extent=2.0, t_max=40e-9, dx=0.02, dt=0.1e-9)


def test_clutter_shape():
    L = low_rank_clutter(GRID, rank=3, rng=np.random.default_rng(0))
    assert L.shape == (GRID.Nt, GRID.Nx)


def test_clutter_rank():
    L = low_rank_clutter(GRID, rank=3, rng=np.random.default_rng(0))
    assert np.linalg.matrix_rank(L, tol=1e-10) == 3


def test_clutter_snr_scaling():
    rng_target = np.random.default_rng(99)
    target = rng_target.standard_normal((GRID.Nt, GRID.Nx))
    L = low_rank_clutter(
        GRID, rank=3, snr_db=10.0, target_image=target, rng=np.random.default_rng(1)
    )
    actual_ratio = np.linalg.norm(L, "fro") / np.linalg.norm(target, "fro")
    expected_ratio = 10 ** (-10.0 / 20.0)
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01


def test_clutter_no_scaling_without_target():
    # Without target_image, snr_db is ignored; result should still be rank-k
    L = low_rank_clutter(GRID, rank=2, snr_db=10.0, rng=np.random.default_rng(5))
    assert L.shape == (GRID.Nt, GRID.Nx)
    assert np.linalg.matrix_rank(L, tol=1e-10) == 2

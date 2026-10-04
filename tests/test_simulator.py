import math

import numpy as np
import pytest

from abp_ldp.simulator import (ABPParams, ABPPopulation, Geometry, SimulationError, WCA_CUTOFF,
                               _wca_forces, count_in_square, count_in_squares, n_cells)


def brute_force_wca(x, y, L, eps=1.0):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    r2 = dx ** 2 + dy ** 2
    np.fill_diagonal(r2, np.inf)
    inside = r2 < WCA_CUTOFF ** 2
    ir2 = np.where(inside, 1.0 / r2, 0.0)
    ir6 = ir2 ** 3
    f = 48.0 * eps * ir2 * ir6 * (ir6 - 0.5)
    return (f * dx).sum(1), (f * dy).sum(1)


def wca_energy(x, y, L, eps=1.0):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    r2 = dx ** 2 + dy ** 2
    iu = np.triu_indices(x.size, 1)
    r2 = r2[iu]
    r2 = r2[r2 < WCA_CUTOFF ** 2]
    ir6 = 1.0 / r2 ** 3
    return np.sum(4 * eps * (ir6 ** 2 - ir6) + eps)


@pytest.mark.parametrize("L", [3.5, 6.0, 12.7])
def test_cell_list_forces_match_brute_force(L):
    rng = np.random.default_rng(0)
    n = int(0.6 * L * L)
    x = rng.uniform(0, L, n)
    y = rng.uniform(0, L, n)
    # avoid extreme overlaps
    keep = [0]
    for i in range(1, n):
        dx = x[i] - x[keep]
        dy = y[i] - y[keep]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        if np.min(dx ** 2 + dy ** 2) > 0.85 ** 2:
            keep.append(i)
    x, y = x[keep].copy(), y[keep].copy()
    nc = n_cells(L)
    fx = np.zeros(x.size)
    fy = np.zeros(x.size)
    _wca_forces(x, y, L, 1.0, nc, np.empty(max(nc * nc, 1), np.int64), np.empty(x.size, np.int64), fx, fy)
    bx, by = brute_force_wca(x, y, L)
    assert np.allclose(fx, bx, atol=1e-9, rtol=1e-9)
    assert np.allclose(fy, by, atol=1e-9, rtol=1e-9)
    assert abs(fx.sum()) < 1e-8 and abs(fy.sum()) < 1e-8


def test_forces_are_minus_energy_gradient():
    rng = np.random.default_rng(1)
    L = 5.0
    x = np.array([1.0, 1.95, 4.6, 2.5])
    y = np.array([1.0, 1.3, 1.0, 4.2])
    nc = n_cells(L)
    fx = np.zeros(4)
    fy = np.zeros(4)
    _wca_forces(x, y, L, 1.0, nc, np.empty(nc * nc, np.int64), np.empty(4, np.int64), fx, fy)
    h = 1e-6
    for i in range(4):
        for arr, f in ((x, fx), (y, fy)):
            a0 = arr[i]
            arr[i] = a0 + h
            ep = wca_energy(x, y, L)
            arr[i] = a0 - h
            em = wca_energy(x, y, L)
            arr[i] = a0
            assert f[i] == pytest.approx(-(ep - em) / (2 * h), rel=1e-5, abs=1e-6)


def test_count_in_square():
    pos = np.array([[[0.5, 1.0, 1.99, 2.0, 3.0], [1.5, 1.0, 1.0, 1.5, 1.5]]])
    assert count_in_square(pos, 1.0, 2.0)[0] == 2  # [lo, hi) in both directions
    out = count_in_squares(pos, np.array([1.0, 0.0]), np.array([2.0, 4.0]))
    assert out[0, 0] == 2 and out[0, 1] == 5


def test_free_abp_msd_and_orientation():
    """Ideal ABPs: MSD = 4 Dt t + 2 v0^2/Dr^2 (Dr t - 1 + e^{-Dr t}), <cos dtheta> = e^{-Dr t}."""
    p = ABPParams(interacting=False, v0=10.0, Dr=3.0, dt=1e-3)
    g = Geometry(ell=10.0, kappa=1.0)
    pop = ABPPopulation(p, g, M=20, seed=5)
    th0 = pop.theta.copy()
    # unwrap positions by integrating in small chunks
    x0 = pop.pos.copy()
    disp = np.zeros_like(x0)
    prev = pop.pos.copy()
    n_chunks, chunk = 50, 10
    for _ in range(n_chunks):
        pop.advance(chunk)
        d = pop.pos - prev
        d -= g.L * np.round(d / g.L)
        disp += d
        prev = pop.pos.copy()
    t = n_chunks * chunk * p.dt
    msd = np.mean(np.sum(disp ** 2, axis=1))
    exact = 4 * p.Dt * t + 2 * p.v0 ** 2 / p.Dr ** 2 * (p.Dr * t - 1 + math.exp(-p.Dr * t))
    n_samples = pop.M * pop.N
    assert msd == pytest.approx(exact, rel=8.0 / math.sqrt(n_samples) + 0.02)
    # orientation decorrelation (theta wrapped, so use the cosine of the difference)
    c = np.mean(np.cos(pop.theta - th0))
    assert c == pytest.approx(math.exp(-p.Dr * t), abs=6.0 / math.sqrt(n_samples))


def test_clones_decorrelate_and_runs_reproducible():
    p = ABPParams(dt=1e-4)
    g = Geometry(ell=4.0, kappa=2.0)
    a = ABPPopulation(p, g, M=4, seed=11)
    b = ABPPopulation(p, g, M=4, seed=11)
    a.soft_start()
    b.soft_start()
    a.advance(50)
    b.advance(50)
    assert np.array_equal(a.pos, b.pos)
    a.resample(np.array([0, 0, 0]))
    assert a.M == 3 and np.array_equal(a.pos[0], a.pos[1])
    a.advance(10)
    assert not np.array_equal(a.pos[0], a.pos[1])


def test_results_independent_of_thread_count():
    import numba
    p = ABPParams(dt=1e-4)
    g = Geometry(ell=4.0, kappa=2.0)
    n0 = numba.get_num_threads()
    out = []
    for nt in (1, n0):
        numba.set_num_threads(nt)
        a = ABPPopulation(p, g, M=6, seed=3)
        a.soft_start()
        a.advance(100)
        out.append(a.pos.copy())
    numba.set_num_threads(n0)
    assert np.array_equal(out[0], out[1])


def test_blow_up_is_detected_not_swallowed():
    p = ABPParams(dt=1e-4)
    g = Geometry(ell=4.0, kappa=2.0)
    a = ABPPopulation(p, g, M=3, seed=1)
    a.pos[1, :, 1] = a.pos[1, :, 0] + 0.05   # two particles almost on top of each other
    with pytest.raises(SimulationError):
        a.advance(5)

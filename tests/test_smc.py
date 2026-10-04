import math

import numpy as np
import pytest

from abp_ldp import analysis as an
from abp_ldp.simulator import ABPParams, Geometry, count_in_square
from abp_ldp.smc import SMCConfig, combine_replicates, run_lambda_grid
from abp_ldp.twist import _interval_prob_deriv, _phi_and_grad, free_abp_moments, twist_energy


def test_twist_at_zero_remaining_time_is_exact_tilt():
    rng = np.random.default_rng(0)
    g = Geometry(ell=4.0, kappa=3.0)
    pos = rng.uniform(0, g.L, size=(5, 2, 40))
    theta = rng.uniform(0, 2 * np.pi, size=(5, 40))
    for lam in (-1.3, 0.7):
        U = twist_energy(pos, theta, 0.0, lam, 24.0, 1.0, 3.0, g.lo, g.hi, g.L)
        assert np.allclose(U, -lam * count_in_square(pos, g.lo, g.hi))


def test_twist_long_time_limit_is_uniform():
    """For s -> infinity p_s -> v/V for every particle."""
    rng = np.random.default_rng(1)
    g = Geometry(ell=4.0, kappa=3.0)
    pos = rng.uniform(0, g.L, size=(3, 2, 30))
    theta = rng.uniform(0, 2 * np.pi, size=(3, 30))
    lam = 0.8
    U = twist_energy(pos, theta, 50.0, lam, 24.0, 1.0, 3.0, g.lo, g.hi, g.L)
    f = g.v / g.V
    assert np.allclose(U, -30 * np.log1p(np.expm1(lam) * f), atol=1e-8)


@pytest.mark.parametrize("sig", [0.05, 0.7, 3.0, 40.0])
def test_interval_probability_derivative(sig):
    L, lo, hi = 12.0, 4.0, 8.0
    for mu in (-3.0, 3.9, 6.0, 8.05, 11.5):
        P, dP = _interval_prob_deriv(mu, sig, lo, hi, L)
        h = 1e-6
        Pp, _ = _interval_prob_deriv(mu + h, sig, lo, hi, L)
        Pm, _ = _interval_prob_deriv(mu - h, sig, lo, hi, L)
        assert 0.0 <= P <= 1.0 + 1e-12
        assert dP == pytest.approx((Pp - Pm) / (2 * h), rel=1e-5, abs=1e-7)


def test_phi_gradient_matches_finite_differences():
    L, lo, hi = 12.0, 4.0, 8.0
    a, sig = free_abp_moments(0.15, 24.0, 1.0, 3.0)
    em1 = math.expm1(0.9)
    x, y, th = 3.1, 6.2, 0.4
    phi, gx, gy, gth = _phi_and_grad(x, y, th, a, sig, em1, lo, hi, L)
    h = 1e-6
    fd = []
    for d in ((h, 0, 0), (0, h, 0), (0, 0, h)):
        pp = _phi_and_grad(x + d[0], y + d[1], th + d[2], a, sig, em1, lo, hi, L)[0]
        pm = _phi_and_grad(x - d[0], y - d[1], th - d[2], a, sig, em1, lo, hi, L)[0]
        fd.append((pp - pm) / (2 * h))
    assert gx == pytest.approx(fd[0], rel=1e-5, abs=1e-8)
    assert gy == pytest.approx(fd[1], rel=1e-5, abs=1e-8)
    assert gth == pytest.approx(fd[2], rel=1e-5, abs=1e-8)


class _ExactIdealReservoir:
    """Uniform positions/orientations: the exact steady state of ideal ABPs."""

    def __init__(self, n_samples, N, L, seed):
        gen = np.random.default_rng(seed)
        self.N_tot = N
        self.n_samples = n_samples
        self.pos = gen.uniform(0, L, size=(n_samples, 2, N))
        self.theta = gen.uniform(0, 2 * np.pi, size=(n_samples, N))


@pytest.mark.parametrize("guided", [True, False])
def test_smc_reproduces_exact_binomial_for_ideal_abps(guided):
    p = ABPParams(interacting=False, v0=24.0)
    g = Geometry(ell=4.0, kappa=3.0)
    N = g.n_particles(p.rho0)
    M, R = 64, 6
    res = _ExactIdealReservoir(M * R, N, g.L, seed=3)
    lams = [-0.6, 0.6]
    out = run_lambda_grid(p, g, SMCConfig(M=M, n_weight=20, guided=guided, seed=11), res, lams, R,
                          0.3, p.v0, p.Dt, verbose=False)
    lnZ, err = combine_replicates(out["lnZ_reps"])
    exact = an.binomial_lnZ(N, g.v / g.V, lams)
    sd = np.std(out["lnZ_reps"], axis=1, ddof=1) / np.sqrt(R)
    for z, e, s in zip(lnZ, exact, sd):
        assert abs(z - e) < 4 * s + 0.05
    # final weighted populations sample the tilted ensemble: <N>_lambda = d lnZ / d lambda
    meanN = (out["hist_reps"].mean(1) * np.arange(N + 1)).sum(1)
    f = g.v / g.V
    exact_mean = [N * f * np.exp(l) / (1 - f + f * np.exp(l)) for l in lams]
    assert np.allclose(meanN, exact_mean, rtol=0.1)


def test_deconvolution_recovers_poisson_rate_function_for_ideal_gas():
    rb, f = 0.4, 1 / 9
    for grid in (np.arange(0.01, 1.2, 0.01), np.arange(0.005, 1.2, 0.013)):
        I_f = an.ideal_rate_function(grid, rb, f)
        I_0 = an.deconvolve_finite_reservoir(grid, I_f + 1e-4, rb, f)
        exact = an.ideal_rate_function(grid, rb, 0.0)
        m = (grid > 0.05) & (grid < 1.1)
        assert np.nanmax(np.abs(I_0[m] - exact[m])) < 0.01

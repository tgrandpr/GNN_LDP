"""Analysis: WHAM reconstruction of ln P_v(N), rate functions, SCGFs, finite-size scaling.

Conventions (large parameter = subvolume area v):
    psi_v(lambda) = (1/v) ln E[exp(lambda N_v)]
    I_v(rho)      = -(1/v) [ln P_v(N_v = rho v) - max_N ln P_v(N)]
"""

import numpy as np
from scipy.special import gammaln, logsumexp


# --------------------------------------------------------------------------
# WHAM with known bias energies
# --------------------------------------------------------------------------

def wham(hist, U, n_eff=None, f_init=None, tol=1e-10, max_iter=100000):
    """Self-consistent WHAM / MBAR-on-a-histogram for discrete N.

    hist : (J, K) weighted histograms of N = 0..K-1 in window j, each row summing to
           the number of recorded snapshots of that window.
    U    : (J, K) bias energies, window j sampled P(N) exp(-U_j(N)) / Z_j.
    n_eff: (J,) sample sizes used to weight windows (defaults to row sums of hist).
    f_init: optional initial guess for f_j = -ln Z_j.

    Returns (lnP, f) with lnP normalised over the support (N with any counts) and
    lnP = -inf elsewhere; f_j = -ln sum_N P(N) exp(-U_j(N)) consistent with lnP.
    """
    hist = np.asarray(hist, float)
    U = np.asarray(U, float)
    J, K = hist.shape
    n = hist.sum(1) if n_eff is None else np.asarray(n_eff, float)
    tot = hist.sum(0)
    support = tot > 0
    ln_tot = np.full(K, -np.inf)
    ln_tot[support] = np.log(tot[support])
    ln_n = np.log(np.maximum(n, 1e-300))
    f = np.zeros(J) if f_init is None else np.asarray(f_init, float).copy()
    f -= f[0]
    lnP = None
    for _ in range(max_iter):
        # ln P(N) = ln sum_j h_j(N) - ln sum_j n_j exp(f_j - U_j(N))
        denom = logsumexp(ln_n[:, None] + f[:, None] - U[:, support], axis=0)
        lnP = np.full(K, -np.inf)
        lnP[support] = ln_tot[support] - denom
        lnP -= logsumexp(lnP[support])
        f_new = -logsumexp(lnP[None, support] - U[:, support], axis=1)
        if np.max(np.abs((f_new - f_new[0]) - (f - f[0]))) < tol:
            f = f_new
            break
        f = f_new
    return lnP, f


def window_estimates(hist, U, lnZ):
    """Per-window estimate ln P(N) = ln p_j(N) + U_j(N) + ln Z_j (SMC normalisation)."""
    hist = np.asarray(hist, float)
    p = hist / hist.sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(p) + U + np.asarray(lnZ)[:, None]


# --------------------------------------------------------------------------
# Large-deviation quantities
# --------------------------------------------------------------------------

def rate_function(lnP, v):
    """rho = N / v and I_v(rho) = -(lnP - max lnP)/v on the support of lnP."""
    N = np.arange(lnP.size)
    ok = np.isfinite(lnP)
    return N[ok] / v, -(lnP[ok] - lnP[ok].max()) / v


def scgf_from_lnP(lnP, v, lambdas):
    """psi_v(lambda) = (1/v) ln sum_N P(N) exp(lambda N)."""
    N = np.arange(lnP.size)
    ok = np.isfinite(lnP)
    lam = np.atleast_1d(np.asarray(lambdas, float))
    return logsumexp(lnP[None, ok] + lam[:, None] * N[None, ok], axis=1) / v


def tilted_mean_from_lnP(lnP, v, lambdas):
    """<rho_v>_lambda = d psi_v / d lambda from the reconstructed distribution."""
    N = np.arange(lnP.size)
    ok = np.isfinite(lnP)
    lam = np.atleast_1d(np.asarray(lambdas, float))
    a = lnP[None, ok] + lam[:, None] * N[None, ok]
    a -= a.max(1, keepdims=True)
    w = np.exp(a)
    w /= w.sum(1, keepdims=True)
    return (w * N[None, ok]).sum(1) / v


def thermodynamic_integration(lambdas, meanN):
    """ln Z(lambda) = int_0^lambda <N>_l dl by the trapezoid rule, for a grid containing 0."""
    lam = np.asarray(lambdas, float)
    m = np.asarray(meanN, float)
    order = np.argsort(lam)
    lam, m = lam[order], m[order]
    i0 = int(np.argmin(np.abs(lam)))
    if lam[i0] != 0.0:
        raise ValueError("the lambda grid must contain lambda = 0")
    out = np.zeros_like(lam)
    for i in range(i0 + 1, lam.size):
        out[i] = out[i - 1] + 0.5 * (m[i] + m[i - 1]) * (lam[i] - lam[i - 1])
    for i in range(i0 - 1, -1, -1):
        out[i] = out[i + 1] - 0.5 * (m[i] + m[i + 1]) * (lam[i + 1] - lam[i])
    res = np.empty_like(out)
    res[order] = out
    return res


def legendre_fenchel(lambdas, psi, rhos):
    """I(rho) = sup_lambda [lambda rho - psi(lambda)] over the sampled lambda grid."""
    lam = np.asarray(lambdas, float)
    psi = np.asarray(psi, float)
    r = np.atleast_1d(np.asarray(rhos, float))
    return np.max(r[:, None] * lam[None, :] - psi[None, :], axis=1)


# --------------------------------------------------------------------------
# Exact references
# --------------------------------------------------------------------------

def binomial_lnP(N_tot, f):
    """ln P(N) for N ~ Binomial(N_tot, f): exact for ideal particles (uniform steady state)."""
    N = np.arange(N_tot + 1)
    return (gammaln(N_tot + 1) - gammaln(N + 1) - gammaln(N_tot - N + 1)
            + N * np.log(f) + (N_tot - N) * np.log1p(-f))


def binomial_lnZ(N_tot, f, lambdas):
    """ln E[exp(lambda N)] = N_tot ln(1 - f + f e^lambda)."""
    lam = np.asarray(lambdas, float)
    return N_tot * np.log1p(f * np.expm1(lam))


def ideal_rate_function(rho, rho0, f):
    """v -> infinity rate function for ideal particles at fixed f = v/V and mean density rho0.

    Binomial(N_tot = rho0 V, f) with v = f V gives
      I(rho) = rho ln(rho/rho0) + ((1-f)/f) rho_o ln(rho_o/rho0) where rho_o = (rho0 - f rho)/(1-f),
    which -> rho ln(rho/rho0) - rho + rho0 (Poisson) as f -> 0.
    """
    rho = np.asarray(rho, float)
    if f == 0:
        return rho * np.log(rho / rho0) - rho + rho0
    rho_o = (rho0 - f * rho) / (1.0 - f)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(rho > 0, rho * np.log(rho / rho0), 0.0)
        b = np.where(rho_o > 0, rho_o * np.log(rho_o / rho0), 0.0)
    return a + (1.0 - f) / f * b


# --------------------------------------------------------------------------
# Finite-size scaling
# --------------------------------------------------------------------------

def interp_rate(rho_grid, rho, I):
    """Linear interpolation of I_v(rho) onto rho_grid; NaN outside the sampled range."""
    out = np.interp(rho_grid, rho, I, left=np.nan, right=np.nan)
    return out


def extrapolate_in_inverse_ell(ells, values, order=1):
    """Fit values(ell) = a0 + a1/ell (+ a2/ell^2) and return a0 and its fit stderr."""
    ells = np.asarray(ells, float)
    y = np.asarray(values, float)
    ok = np.isfinite(y)
    x = 1.0 / ells[ok]
    y = y[ok]
    if y.size < order + 1:
        return np.nan, np.nan
    A = np.vander(x, order + 1, increasing=True)
    coef, res, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    dof = y.size - (order + 1)
    if dof > 0:
        s2 = np.sum((A @ coef - y) ** 2) / dof
        cov = s2 * np.linalg.inv(A.T @ A)
        err = float(np.sqrt(cov[0, 0]))
    else:
        err = np.nan
    return float(coef[0]), err


def deconvolve_finite_reservoir(rho, I_f, rho_bar, f, n_iter=200, tol=1e-10):
    """Estimate the f -> 0 (infinite reservoir) rate function I_0 from I_f at fixed f = v/V.

    Assumes additivity of the static fluctuations (exact for ideal and short-ranged
    equilibrium systems, a hypothesis for ABPs):
        I_f(rho) = I_0(rho) + ((1 - f)/f) I_0(rho_out),  rho_out = (rho_bar - f rho)/(1 - f).
    The equation fixes I_0 only up to a + b (rho - rho_bar) (the canonical gauge
    freedom exp(lambda N_v) exp(lambda (N_tot - N_v)) = const), and the plain fixed-point
    map amplifies constant errors by (1 - f)/f.  Each iterate is therefore projected onto
    the gauge I_0(rho_bar) = I_0'(rho_bar) = 0 and the update is damped; the remaining
    (smooth, quadratic and higher) components then contract.  Returns NaN if it does not
    converge.
    """
    rho = np.asarray(rho, float)
    I_f = np.asarray(I_f, float)
    ok = np.isfinite(I_f)
    r, If = rho[ok], I_f[ok]
    order = np.argsort(r)
    r, If = r[order], If[order]
    out = np.full(rho.shape, np.nan)
    if r.size < 3 or not (r[0] <= rho_bar <= r[-1]):
        return out
    rho_out = (rho_bar - f * r) / (1.0 - f)
    c = (1.0 - f) / f

    def gauge(I):
        a0 = np.interp(rho_bar, r, I)
        s0 = np.interp(rho_bar, r, np.gradient(I, r))
        return I - a0 - s0 * (r - rho_bar)

    I0 = gauge(If.copy())
    converged = False
    for _ in range(n_iter):
        # damped (Mann) update: kills the non-analytic |rho - rho_bar| mode created by
        # interpolation near rho_bar, which the undamped map flips with factor -1
        new = gauge(0.5 * I0 + 0.5 * (If - c * np.interp(rho_out, r, I0)))
        if np.max(np.abs(new - I0)) < tol:
            I0 = new
            converged = True
            break
        I0 = new
    if converged:
        out[np.flatnonzero(ok)[order]] = I0
    return out


def tilted_rate_points(lambdas, lnZ, mean_rho, v):
    """Legendre points of the tilted ensembles: (rho_lambda, lambda rho_lambda - psi_v(lambda))."""
    lam = np.asarray(lambdas, float)
    psi = np.asarray(lnZ, float) / v
    r = np.asarray(mean_rho, float)
    return r, lam * r - psi

"""One-body "twist" (approximate value function) for terminal tilting of N_v.

For a finite horizon T and a terminal tilt exp(lambda N_v(x_T)), the optimal
intermediate bias at time t (remaining horizon s = T - t) is the value function

    h_s(x) = ln E[ exp(lambda N_v(x_T)) | x_t = x ].

For non-interacting ABPs it is exactly one-body,

    h_s(x) = sum_i ln[ 1 + (e^lambda - 1) p_s(r_i, theta_i) ],

with p_s(r, theta) the probability that a free ABP starting at (r, theta) is
inside the subvolume after a time s.  We use it (with effective parameters for
interacting particles) as the twist U_k(x) = -h_{s_k}(x); at s = 0 it reduces
exactly to U = -lambda N_v.  The Feynman-Kac estimator is exact for ANY choice of
twist - the twist only controls the variance.

p_s is approximated by an isotropic Gaussian for the displacement of a free ABP
with initial orientation theta:

    mean     m_s = (v/Dr) (1 - e^{-Dr s}) e(theta)
    MSD(s)   = 4 Dt s + 2 (v/Dr)^2 (Dr s - 1 + e^{-Dr s})
    variance per axis  sigma_s^2 = (MSD(s) - |m_s|^2) / 2

so p_s factorises into erf differences along x and y (with periodic images).
"""

import math

import numpy as np
from numba import njit, prange

from .simulator import R2_FAIL, _normal_pair, _wca_forces


@njit(inline="always")
def _interval_prob(mu, sig, lo, hi, L):
    """P(X mod L in [lo, hi)) for X ~ N(mu, sig^2)."""
    if sig <= 0.0:
        xm = mu - L * math.floor(mu / L)
        return 1.0 if (xm >= lo and xm < hi) else 0.0
    if sig > 2.0 * L:
        # Gaussian much wider than the box: uniform up to exponentially small terms
        k = 2.0 * math.pi / L
        return (hi - lo) / L + (2.0 / math.pi) * math.exp(-0.5 * (k * sig) ** 2) * (
            math.sin(k * (hi - mu)) - math.sin(k * (lo - mu))) * 0.5
    # shift mu into [0, L) and sum over enough images
    mu = mu - L * math.floor(mu / L)
    n_img = int(math.ceil(6.0 * sig / L)) + 1
    s2 = sig * math.sqrt(2.0)
    tot = 0.0
    for n in range(-n_img, n_img + 1):
        a = (lo + n * L - mu) / s2
        b = (hi + n * L - mu) / s2
        tot += 0.5 * (math.erf(b) - math.erf(a))
    return tot


@njit(inline="always")
def free_abp_moments(s, v, Dt, Dr):
    """(persistent shift length, per-axis std) of a free ABP displacement after time s."""
    if s <= 0.0:
        return 0.0, 0.0
    a = (v / Dr) * (1.0 - math.exp(-Dr * s))
    msd = 4.0 * Dt * s + 2.0 * (v / Dr) ** 2 * (Dr * s - 1.0 + math.exp(-Dr * s))
    var = 0.5 * (msd - a * a)
    if var < 0.0:
        var = 0.0
    return a, math.sqrt(var)


@njit(parallel=True, cache=True)
def twist_energy(pos, theta, s, lam, v, Dt, Dr, lo, hi, L):
    """U(x) = -sum_i ln[1 + (e^lam - 1) p_s(r_i, theta_i)] for every replica; s = remaining time.

    At s = 0 this is exactly -lam * N_v(x).
    """
    M = pos.shape[0]
    n = pos.shape[2]
    out = np.zeros(M)
    em1 = math.expm1(lam)
    a, sig = free_abp_moments(s, v, Dt, Dr)
    for m in prange(M):
        acc = 0.0
        if s <= 0.0:
            c = 0
            for i in range(n):
                xi = pos[m, 0, i]
                yi = pos[m, 1, i]
                if xi >= lo and xi < hi and yi >= lo and yi < hi:
                    c += 1
            out[m] = -lam * c
            continue
        for i in range(n):
            th = theta[m, i]
            px = _interval_prob(pos[m, 0, i] + a * math.cos(th), sig, lo, hi, L)
            py = _interval_prob(pos[m, 1, i] + a * math.sin(th), sig, lo, hi, L)
            acc += math.log1p(em1 * px * py)
        out[m] = -acc
    return out


def p_inside(x, y, theta, s, v, Dt, Dr, lo, hi, L):
    """Vectorised p_s for single particles (used in tests and diagnostics)."""
    x = np.atleast_1d(np.asarray(x, float))
    y = np.broadcast_to(np.asarray(y, float), x.shape)
    th = np.broadcast_to(np.asarray(theta, float), x.shape)
    pos = np.stack([x, y])[None]
    out = np.empty(x.size)
    # ln(1 + (e - 1) p) with lam -> 0 limit would lose p; use lam = ln 2 so that U = -ln(1 + p)
    U = np.array([twist_energy(pos[:, :, i:i + 1].copy(), th[None, i:i + 1].copy(), s, math.log(2.0),
                               v, Dt, Dr, lo, hi, L)[0] for i in range(x.size)])
    out[:] = np.expm1(-U)
    return out


# --------------------------------------------------------------------------
# Doob-guided dynamics: control force / torque from the same one-body value function
# --------------------------------------------------------------------------

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@njit(inline="always")
def _interval_prob_deriv(mu, sig, lo, hi, L):
    """(P, dP/dmu) for P = P(X mod L in [lo, hi)), X ~ N(mu, sig^2), sig > 0."""
    if sig <= 0.0:
        xm = mu - L * math.floor(mu / L)
        return (1.0 if (xm >= lo and xm < hi) else 0.0), 0.0
    if sig > 2.0 * L:
        k = 2.0 * math.pi / L
        e = math.exp(-0.5 * (k * sig) ** 2) / math.pi
        P = (hi - lo) / L + e * (math.sin(k * (hi - mu)) - math.sin(k * (lo - mu)))
        dP = e * k * (math.cos(k * (lo - mu)) - math.cos(k * (hi - mu)))
        return P, dP
    mu = mu - L * math.floor(mu / L)
    n_img = int(math.ceil(6.0 * sig / L)) + 1
    s2 = sig * math.sqrt(2.0)
    P = 0.0
    dP = 0.0
    for n in range(-n_img, n_img + 1):
        za = (lo + n * L - mu) / sig
        zb = (hi + n * L - mu) / sig
        if za > 9.0 or zb < -9.0:
            continue
        P += 0.5 * (math.erf(zb / math.sqrt(2.0)) - math.erf(za / math.sqrt(2.0)))
        dP += _INV_SQRT_2PI * (math.exp(-0.5 * za * za) - math.exp(-0.5 * zb * zb)) / sig
    return P, dP


@njit(inline="always")
def _phi_and_grad(x, y, th, a, sig, em1, lo, hi, L):
    """phi = ln(1 + em1 p_s) and its derivatives w.r.t. x, y, theta."""
    c = math.cos(th)
    sn = math.sin(th)
    Px, dPx = _interval_prob_deriv(x + a * c, sig, lo, hi, L)
    Py, dPy = _interval_prob_deriv(y + a * sn, sig, lo, hi, L)
    p = Px * Py
    den = 1.0 + em1 * p
    gx = em1 * dPx * Py / den
    gy = em1 * Px * dPy / den
    gth = em1 * (dPx * (-a * sn) * Py + Px * dPy * (a * c)) / den
    return math.log1p(em1 * p), gx, gy, gth


@njit(parallel=True, cache=True)
def advance_controlled(pos, theta, rng, n_steps, s_start, lam, tv, tDt, L, v0, Dt, Dr, eps, dt,
                       interacting, lo, hi, control_every, s_floor, s_fine, logPQ, failed):
    """Advance replicas under the Doob-guided dynamics and accumulate ln(dP/dQ).

    Control force u_i = 2 Dt d phi_s / d r_i and torque w_i = 2 Dr d phi_s / d theta_i with
    phi_s = ln(1 + (e^lam - 1) p_s(r_i, theta_i)) evaluated with the twist parameters
    (tv, tDt) at the remaining time s = max(s_start - k dt, s_floor); the control is
    refreshed every ``control_every`` steps, and at every step once s < s_fine (close to
    the final time the value function sharpens).  For each Euler-Maruyama step with standard
    normal noises xi (translation) and eta (rotation) of the guided chain Q,
        ln dP/dQ = -u.xi sqrt(dt / 2Dt) - dt |u|^2 / 4Dt - w eta sqrt(dt / 2Dr) - dt w^2 / 4Dr,
    which is exact for the discrete-time chains.
    """
    M = theta.shape[0]
    n = theta.shape[1]
    ncell = max(int(L / 1.122462048309373), 1)
    sq_t = math.sqrt(2.0 * Dt * dt)
    sq_r = math.sqrt(2.0 * Dr * dt)
    ct = math.sqrt(dt / (2.0 * Dt))
    cr = math.sqrt(dt / (2.0 * Dr))
    em1 = math.expm1(lam)
    two_pi = 2.0 * math.pi
    for m in prange(M):
        x = pos[m, 0]
        y = pos[m, 1]
        th = theta[m]
        fx = np.zeros(n)
        fy = np.zeros(n)
        ux = np.zeros(n)
        uy = np.zeros(n)
        wt = np.zeros(n)
        head = np.empty(ncell * ncell, np.int64)
        nxt = np.empty(n, np.int64)
        s = rng[m]
        have_spare = False
        spare = 0.0
        acc = 0.0
        for k in range(n_steps):
            if k % control_every == 0 or s_start - k * dt < s_fine:
                srem = s_start - k * dt
                if srem < s_floor:
                    srem = s_floor
                a, sig = free_abp_moments(srem, tv, tDt, Dr)
                for i in range(n):
                    _ph, gx, gy, gth = _phi_and_grad(x[i], y[i], th[i], a, sig, em1, lo, hi, L)
                    ux[i] = 2.0 * Dt * gx
                    uy[i] = 2.0 * Dt * gy
                    wt[i] = 2.0 * Dr * gth
            if interacting:
                r2min = _wca_forces(x, y, L, eps, ncell, head, nxt, fx, fy)
                if r2min < R2_FAIL:
                    failed[m] = True
                    break
            for i in range(n):
                s, g1, g2 = _normal_pair(s)
                if have_spare:
                    g3 = spare
                    have_spare = False
                else:
                    s, g3, spare = _normal_pair(s)
                    have_spare = True
                aa = th[i]
                xi = x[i] + dt * (v0 * math.cos(aa) + fx[i] + ux[i]) + sq_t * g1
                yi = y[i] + dt * (v0 * math.sin(aa) + fy[i] + uy[i]) + sq_t * g2
                xi -= L * math.floor(xi / L)
                yi -= L * math.floor(yi / L)
                if xi >= L:
                    xi -= L
                if yi >= L:
                    yi -= L
                if not (math.isfinite(xi) and math.isfinite(yi)):
                    failed[m] = True
                x[i] = xi
                y[i] = yi
                aa += dt * wt[i] + sq_r * g3
                th[i] = aa - two_pi * math.floor(aa / two_pi)
                acc += (-(ux[i] * g1 + uy[i] * g2) * ct - dt * (ux[i] * ux[i] + uy[i] * uy[i]) / (4.0 * Dt)
                        - wt[i] * g3 * cr - dt * wt[i] * wt[i] / (4.0 * Dr))
            if failed[m]:
                break
        logPQ[m] += acc
        rng[m] = s

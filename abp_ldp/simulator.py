"""Batched simulator for two-dimensional active Brownian particles (ABPs).

Model (overdamped, Ito, Euler-Maruyama), in units sigma = mu = Dt = 1:

    dr_i     = [ v0 e(theta_i) + F_i ] dt + sqrt(2 Dt) dW_i
    dtheta_i = sqrt(2 Dr) dW'_i

with a WCA pair repulsion U(r) = 4 eps [(1/r)^12 - (1/r)^6] + eps for r < 2^(1/6).
The box is an L x L square with periodic boundaries.

Many independent replicas ("clones") are advanced at once.  Replicas are
distributed over threads with numba ``prange``; each replica owns a cell list
and its own SplitMix64 random stream, so a run is reproducible for a given
seed regardless of the number of threads.

Array layout
------------
pos   : float64 (M, 2, N)   x = pos[m, 0], y = pos[m, 1], in [0, L)
theta : float64 (M, N)      orientation angles in [0, 2 pi)
rng   : uint64  (M,)        per-replica SplitMix64 state
"""

import math
from dataclasses import dataclass, asdict

import numpy as np
from numba import njit, prange

WCA_CUTOFF = 2.0 ** (1.0 / 6.0)
R2_FAIL = 0.25   # a WCA pair closer than 0.5 sigma (force ~ 4e5) means the integration blew up


class SimulationError(RuntimeError):
    """Raised when a replica's integration blows up (see R2_FAIL)."""


def check_failed(failed, where):
    if np.any(failed):
        bad = np.flatnonzero(failed)
        raise SimulationError(f"{where}: integration failed in replica(s) {bad[:10].tolist()} "
                              f"(pair closer than {np.sqrt(R2_FAIL):.2f} sigma or non-finite "
                              f"position); reduce dt or call soft_start() first")

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_TWO_M53 = 1.0 / 9007199254740992.0


@dataclass
class ABPParams:
    """Physical and numerical parameters of the ABP model (sigma = Dt = 1)."""

    rho0: float = 0.4          # number density N / L^2
    v0: float = 24.0           # self-propulsion speed
    Dt: float = 1.0            # translational diffusion coefficient
    Dr: float = 3.0            # rotational diffusion coefficient (3 Dt / sigma^2)
    eps: float = 1.0           # WCA energy scale
    dt: float = 1e-4           # Euler-Maruyama time step
    interacting: bool = True   # False -> ideal (non-interacting) ABPs
    force_cap: float = 0.0     # cap on |F_i| (0 = no cap); only used for soft starts

    @property
    def peclet(self):
        """Pe = 3 v0 / (sigma Dr), the usual ABP Peclet number."""
        return 3.0 * self.v0 / self.Dr

    def to_dict(self):
        return asdict(self)


@dataclass
class Geometry:
    """Square box of side L = kappa * ell with a central square subvolume of side ell."""

    ell: float                 # subvolume side; subvolume area v = ell^2
    kappa: float = 3.0         # L / ell, held fixed when ell is varied
    L: float = None            # box side; if given, overrides kappa * ell

    def __post_init__(self):
        if self.L is None:
            self.L = self.kappa * self.ell
        else:
            self.kappa = self.L / self.ell
        if self.ell > self.L:
            raise ValueError("subvolume larger than the box")

    @property
    def v(self):
        return self.ell * self.ell

    @property
    def V(self):
        return self.L * self.L

    @property
    def lo(self):
        return 0.5 * (self.L - self.ell)

    @property
    def hi(self):
        return 0.5 * (self.L + self.ell)

    def n_particles(self, rho0):
        return int(round(rho0 * self.V))

    def to_dict(self):
        return dict(ell=self.ell, kappa=self.kappa, L=self.L, v=self.v, V=self.V,
                    lo=self.lo, hi=self.hi)


# --------------------------------------------------------------------------
# Random numbers: SplitMix64 + Box-Muller, one independent stream per replica
# --------------------------------------------------------------------------

@njit(inline="always")
def _splitmix64(s):
    s = s + _GOLDEN
    z = s
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    z = z ^ (z >> np.uint64(31))
    return s, z


@njit(inline="always")
def _uniform_open0(s):
    """Uniform deviate in (0, 1]."""
    s, z = _splitmix64(s)
    return s, ((z >> np.uint64(11)) + np.uint64(1)) * _TWO_M53


@njit(inline="always")
def _normal_pair(s):
    """Two independent standard normals (Marsaglia polar method)."""
    while True:
        s, z1 = _splitmix64(s)
        s, z2 = _splitmix64(s)
        u = 2.0 * ((z1 >> np.uint64(11)) * _TWO_M53) - 1.0
        w = 2.0 * ((z2 >> np.uint64(11)) * _TWO_M53) - 1.0
        q = u * u + w * w
        if q < 1.0 and q > 0.0:
            f = math.sqrt(-2.0 * math.log(q) / q)
            return s, u * f, w * f


def new_rng_states(n, gen):
    """Fresh, independent uint64 stream states drawn from a numpy Generator."""
    return gen.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64, endpoint=True)


# --------------------------------------------------------------------------
# Forces
# --------------------------------------------------------------------------

@njit(inline="always")
def _wca_pair(xi, yi, xj, yj, L, eps, rc2):
    """WCA force on i from j (minimum image) and the squared distance."""
    # minimum image; coordinates live in [0, L) so |dx| < L
    dx = xi - xj
    if dx > 0.5 * L:
        dx -= L
    elif dx < -0.5 * L:
        dx += L
    dy = yi - yj
    if dy > 0.5 * L:
        dy -= L
    elif dy < -0.5 * L:
        dy += L
    r2 = dx * dx + dy * dy
    if r2 < rc2:
        r2s = r2 if r2 > 1e-12 else 1e-12   # coincident particles: finite force (flagged by callers)
        ir2 = 1.0 / r2s
        ir6 = ir2 * ir2 * ir2
        f = 48.0 * eps * ir2 * ir6 * (ir6 - 0.5)
        return f * dx, f * dy, r2
    return 0.0, 0.0, r2


@njit(cache=True)
def _wca_forces(x, y, L, eps, ncell, head, nxt, fx, fy):
    """WCA forces on all particles of one replica (cell list, Newton's third law).

    Returns the smallest squared pair distance among interacting pairs (rc^2 if none),
    which callers use to detect a numerical blow-up."""
    n = x.shape[0]
    rc2 = WCA_CUTOFF * WCA_CUTOFF
    r2min = rc2
    for i in range(n):
        fx[i] = 0.0
        fy[i] = 0.0
    if ncell >= 3:
        cs = L / ncell
        for c in range(ncell * ncell):
            head[c] = -1
        for i in range(n):
            cx = int(x[i] / cs)
            cy = int(y[i] / cs)
            if cx >= ncell:
                cx = ncell - 1
            if cy >= ncell:
                cy = ncell - 1
            c = cx * ncell + cy
            nxt[i] = head[c]
            head[c] = i
        for cx in range(ncell):
            for cy in range(ncell):
                i = head[cx * ncell + cy]
                while i >= 0:
                    xi = x[i]
                    yi = y[i]
                    fxi = 0.0
                    fyi = 0.0
                    # remaining particles in the same cell
                    j = nxt[i]
                    while j >= 0:
                        gx, gy, r2 = _wca_pair(xi, yi, x[j], y[j], L, eps, rc2)
                        if r2 < r2min:
                            r2min = r2
                        fxi += gx
                        fyi += gy
                        fx[j] -= gx
                        fy[j] -= gy
                        j = nxt[j]
                    # half shell of neighbouring cells: (1,0), (1,1), (0,1), (-1,1)
                    for k in range(4):
                        if k == 0:
                            ox, oy = 1, 0
                        elif k == 1:
                            ox, oy = 1, 1
                        elif k == 2:
                            ox, oy = 0, 1
                        else:
                            ox, oy = -1, 1
                        nx = (cx + ox + ncell) % ncell
                        ny = (cy + oy + ncell) % ncell
                        j = head[nx * ncell + ny]
                        while j >= 0:
                            gx, gy, r2 = _wca_pair(xi, yi, x[j], y[j], L, eps, rc2)
                            if r2 < r2min:
                                r2min = r2
                            fxi += gx
                            fyi += gy
                            fx[j] -= gx
                            fy[j] -= gy
                            j = nxt[j]
                    fx[i] += fxi
                    fy[i] += fyi
                    i = nxt[i]
    else:
        for i in range(n):
            for j in range(i + 1, n):
                gx, gy, r2 = _wca_pair(x[i], y[i], x[j], y[j], L, eps, rc2)
                if r2 < r2min:
                    r2min = r2
                fx[i] += gx
                fy[i] += gy
                fx[j] -= gx
                fy[j] -= gy

    return r2min

def n_cells(L):
    """Cells per side for the cell list (cell side >= WCA cutoff)."""
    return max(int(L / WCA_CUTOFF), 1)


@njit(inline="always")
def _step(x, y, th, fx, fy, head, nxt, ncell, s, have_spare, spare,
          L, v0, sq_t, sq_r, eps, dt, interacting, force_cap, lo, hi):
    """One Euler-Maruyama step of one replica.  Returns the updated RNG state, the
    number of particles inside the square [lo, hi)^2 after the step, and a failure flag
    (a pair closer than sqrt(R2_FAIL) without force cap, or a non-finite position)."""
    n = x.shape[0]
    two_pi = 2.0 * math.pi
    bad = False
    if interacting:
        r2min = _wca_forces(x, y, L, eps, ncell, head, nxt, fx, fy)
        if force_cap <= 0.0 and r2min < R2_FAIL:
            bad = True
        if force_cap > 0.0:
            for i in range(n):
                f = math.sqrt(fx[i] * fx[i] + fy[i] * fy[i])
                if f > force_cap:
                    fx[i] *= force_cap / f
                    fy[i] *= force_cap / f
    n_in = 0
    for i in range(n):
        s, g1, g2 = _normal_pair(s)
        if have_spare:
            g3 = spare
            have_spare = False
        else:
            s, g3, spare = _normal_pair(s)
            have_spare = True
        a = th[i]
        xi = x[i] + dt * (v0 * math.cos(a) + fx[i]) + sq_t * g1
        yi = y[i] + dt * (v0 * math.sin(a) + fy[i]) + sq_t * g2
        xi -= L * math.floor(xi / L)
        yi -= L * math.floor(yi / L)
        if xi >= L:
            xi -= L
        if yi >= L:
            yi -= L
        if not (math.isfinite(xi) and math.isfinite(yi)):
            bad = True
            xi = 0.0
            yi = 0.0
        x[i] = xi
        y[i] = yi
        if xi >= lo and xi < hi and yi >= lo and yi < hi:
            n_in += 1
        a += sq_r * g3
        th[i] = a - two_pi * math.floor(a / two_pi)
    return s, have_spare, spare, n_in, bad


@njit(parallel=True, cache=True)
def _advance(pos, theta, rng, n_steps, L, v0, Dt, Dr, eps, dt, interacting, force_cap, failed):
    M = theta.shape[0]
    n = theta.shape[1]
    ncell = max(int(L / WCA_CUTOFF), 1)
    sq_t = math.sqrt(2.0 * Dt * dt)
    sq_r = math.sqrt(2.0 * Dr * dt)
    for m in prange(M):
        fx = np.zeros(n)
        fy = np.zeros(n)
        head = np.empty(ncell * ncell, np.int64)
        nxt = np.empty(n, np.int64)
        s = rng[m]
        have_spare = False
        spare = 0.0
        for _ in range(n_steps):
            s, have_spare, spare, _n, bad = _step(pos[m, 0], pos[m, 1], theta[m], fx, fy, head, nxt,
                                                  ncell, s, have_spare, spare, L, v0, sq_t, sq_r,
                                                  eps, dt, interacting, force_cap, 0.0, 0.0)
            if bad:
                failed[m] = True
                break
        rng[m] = s


@njit(parallel=True, cache=True)
def count_in_square(pos, lo, hi):
    """Number of particle centres inside [lo, hi)^2 for every replica."""
    M = pos.shape[0]
    n = pos.shape[2]
    out = np.zeros(M, np.int64)
    for m in prange(M):
        c = 0
        for i in range(n):
            xi = pos[m, 0, i]
            yi = pos[m, 1, i]
            if xi >= lo and xi < hi and yi >= lo and yi < hi:
                c += 1
        out[m] = c
    return out


@njit(parallel=True, cache=True)
def count_in_squares(pos, los, his):
    """Counts inside several (e.g. concentric) squares; returns (M, K)."""
    M = pos.shape[0]
    n = pos.shape[2]
    K = los.shape[0]
    out = np.zeros((M, K), np.int64)
    for m in prange(M):
        for i in range(n):
            xi = pos[m, 0, i]
            yi = pos[m, 1, i]
            for k in range(K):
                if xi >= los[k] and xi < his[k] and yi >= los[k] and yi < his[k]:
                    out[m, k] += 1
    return out


@njit(cache=True)
def min_pair_distance(x, y, L):
    """Smallest minimum-image pair distance in one replica (diagnostic, O(N^2))."""
    n = x.shape[0]
    best = 1e300
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dx -= L * math.floor(dx / L + 0.5)
            dy = y[i] - y[j]
            dy -= L * math.floor(dy / L + 0.5)
            r2 = dx * dx + dy * dy
            if r2 < best:
                best = r2
    return math.sqrt(best)


class ABPPopulation:
    """A population of M independent ABP replicas sharing parameters and geometry."""

    def __init__(self, params, geom, M, seed=0):
        self.params = params
        self.geom = geom
        self.M = int(M)
        self.N = geom.n_particles(params.rho0)
        self.rng_master = np.random.default_rng(seed)
        self.pos, self.theta = random_init(self.M, self.N, geom.L, self.rng_master)
        self.rng = new_rng_states(self.M, self.rng_master)
        self.time = 0.0

    @property
    def rho_actual(self):
        return self.N / self.geom.V

    def advance(self, n_steps, force_cap=None):
        p = self.params
        cap = p.force_cap if force_cap is None else force_cap
        failed = np.zeros(self.M, np.bool_)
        _advance(self.pos, self.theta, self.rng, int(n_steps), float(self.geom.L),
                 float(p.v0), float(p.Dt), float(p.Dr), float(p.eps), float(p.dt),
                 bool(p.interacting), float(cap), failed)
        check_failed(failed, "ABPPopulation.advance")
        self.time += n_steps * p.dt

    def count(self):
        return count_in_square(self.pos, self.geom.lo, self.geom.hi)

    def resample(self, idx):
        """Replace the population by copies of replicas ``idx`` (any length); every replica
        gets a fresh, independent noise stream so that copies decorrelate."""
        idx = np.asarray(idx)
        self.pos = np.ascontiguousarray(self.pos[idx])
        self.theta = np.ascontiguousarray(self.theta[idx])
        self.M = int(idx.size)
        self.rng = new_rng_states(self.M, self.rng_master)

    def soft_start(self, t_soft=0.2, cap=50.0):
        """Force-capped run that removes the overlaps of the random initial condition.

        The cap is raised geometrically so that particles are pushed apart gently.
        Does nothing for ideal particles."""
        if not self.params.interacting:
            return
        n = max(int(round(t_soft / self.params.dt / 4)), 1)
        for c in (cap, 4 * cap, 16 * cap, 64 * cap):
            self.advance(n, force_cap=c)


def random_init(M, N, L, gen):
    """Uniformly random positions and orientations (the exact steady state of ideal ABPs).

    Overlaps of interacting particles are removed by ``ABPPopulation.soft_start``."""
    pos = gen.uniform(0.0, L, size=(M, 2, N))
    theta = gen.uniform(0.0, 2.0 * math.pi, size=(M, N))
    return np.ascontiguousarray(pos), np.ascontiguousarray(theta)

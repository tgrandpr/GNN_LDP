"""Finite-horizon, twisted Feynman-Kac SMC for STATIC large deviations of N_v.

Target: for a biasing field lambda conjugate to the subvolume particle number,

    Z_v(lambda) = E_ss[ exp(lambda N_v) ],        psi_v(lambda) = ln Z_v(lambda) / v,
    pi_lambda(x) = P_ss(x) exp(lambda N_v(x)) / Z_v(lambda),

where P_ss is the unknown, non-Boltzmann steady state of the ABP dynamics and the
large parameter is the subvolume area v (NOT time: N_v is an instantaneous count).

Algorithm (one run = one lambda)
--------------------------------
1. Draw M independent steady-state configurations x_0 ~ P_ss from a reservoir.
2. Evolve them for a finite horizon T with the UNBIASED ABP dynamics.
3. At weighting times t_0 = 0 < t_1 < ... < t_K = T multiply each replica's
   weight by  exp(-U_k(x_{t_k}) + U_{k-1}(x_{t_{k-1}}))  with U_{-1} = 0 and
   U_K(x) = -lambda N_v(x) exactly.  The intermediate twists
       U_k(x) = -sum_i ln[1 + (e^lambda - 1) p_{T - t_k}(r_i, theta_i)]
   approximate the value function ln E[exp(lambda N_v(x_T)) | x_{t_k}] (exact for
   ideal ABPs), see twist.py.
4. Resample (systematic) whenever the effective sample size drops below
   ess_threshold * M; children get independent noise.

The weights telescope to exp(lambda N_v(x_T)); since x_0 ~ P_ss and the dynamics
preserve P_ss,

    Z_hat = prod over resampling epochs of mean_i(w_i)   is unbiased for Z_v(lambda),

and the final weighted population samples pi_lambda.  This holds for ANY twist;
the twist and M only set the variance.  (Strictly, exact unbiasedness holds for a
resampling schedule fixed in advance; with the ESS-triggered schedule used here the
estimator is consistent with an O(1/M) bias, negligible against the statistical error
and checked against the exact binomial result for ideal ABPs.)  Importantly, lambda is never "held":
holding a tilt against dynamics that relax to P_ss makes the variance grow like
exp(lambda^2 Var N_v), i.e. exponentially in v.

Errors are estimated from independent replicate runs (independent reservoir
samples); Z (not ln Z) is averaged over replicates.
"""

import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.special import logsumexp

from .simulator import ABPPopulation, check_failed, count_in_square, new_rng_states
from .twist import advance_controlled, twist_energy


@dataclass
class SMCConfig:
    M: int = 128                 # replicas per run
    n_weight: int = 20           # steps between weightings (cheap; resampling is adaptive)
    ess_threshold: float = 0.5
    guided: bool = True          # Doob-guided dynamics (control force + torque) with Girsanov weights
    control_every: int = 10      # steps between control refreshes
    fine_steps: int = 50         # ... but refresh every step during the last fine_steps steps
    s_floor_steps: int = 5       # control uses remaining time >= s_floor_steps * dt
    seed: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ReservoirConfig:
    M: int = 512                 # independent replicas; each contributes ONE x_0 sample
    t_burn: float = 4.0          # burn-in from random initial conditions (>> tau of the box mode)
    t_prod: float = 2.0          # production run for brute-force histograms and the N_v ACF;
    t_prod_max: float = 16.0     # ... extended (doubling) until the ACF has decayed below 1/e
    acf_dt: float = 0.02         # lag spacing of the N_v autocorrelation
    measure_every: float = 0.01  # N_v measurement spacing during production
    n_extra: int = 32            # extra replicas, never used as x_0: they set tau_v (horizon) and
                                 # the twist parameters, which must not depend on the x_0 samples
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def systematic_resample(w, gen):
    M = w.shape[0]
    u = (gen.random() + np.arange(M)) / M
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, u, side="right")


def _logmeanexp(a):
    m = np.max(a)
    return m + np.log(np.mean(np.exp(a - m)))


class Reservoir:
    """Independent steady-state configurations plus unbiased statistics of N_v.

    Also collects brute-force histograms of N_v over all kappa^2 tiled placements
    of the subvolume (the steady state is translation invariant, so every
    placement has the same distribution) and the N_v autocorrelation function.
    """

    def __init__(self, params, geom, cfg, verbose=True):
        self.params, self.geom, self.cfg, self.verbose = params, geom, cfg, verbose
        self.pop = ABPPopulation(params, geom, cfg.M + cfg.n_extra, seed=[cfg.seed, 0xA, 1])
        self.N_tot = self.pop.N
        g = geom
        k = int(round(g.kappa))
        if abs(g.kappa - k) < 1e-9 and k >= 1:
            offs = g.lo + g.ell * np.arange(-(k // 2), k - k // 2)
            offs = np.mod(offs, g.L)
            # tiled placements (square j covers [o, o + ell)); wrap-around handled by shifting
            self.tile_offsets = [(ox, oy) for ox in offs for oy in offs]
        else:
            self.tile_offsets = [(g.lo, g.lo)]

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _tiled_counts(self):
        """N_v for every tiled placement: returns (M, n_tiles)."""
        g = self.geom
        out = []
        for ox, oy in self.tile_offsets:
            sx = (self.pop.pos[:, 0] - ox) % g.L
            sy = (self.pop.pos[:, 1] - oy) % g.L
            out.append(np.sum((sx < g.ell) & (sy < g.ell), axis=1))
        return np.stack(out, 1)

    def build(self):
        """Burn in, then run a production period collecting unbiased statistics.

        The final configuration of every replica is one reservoir sample.  Samples
        are therefore fully independent (different trajectories), which matters:
        replicate SMC runs drawn from correlated snapshots share a common error.
        """
        cfg, p, g = self.cfg, self.params, self.geom
        t0 = time.time()
        self.pop.soft_start()
        n_meas = max(int(round(cfg.measure_every / p.dt)), 1)
        n_burn = int(round(cfg.t_burn / p.dt))
        done = 0
        while done < n_burn:
            k = min(4000, n_burn - done)
            self.pop.advance(k)
            done += k
        self._log(f"  reservoir: {cfg.M} replicas, burn-in t={cfg.t_burn} ({time.time() - t0:.0f}s)")
        K = self.N_tot + 1
        n_blocks = 8
        self.hist_center = np.zeros(K)
        self.hist_tiles = np.zeros(K)
        chunks = []
        tile_chunks = []
        t_done = 0.0
        t_next = cfg.t_prod
        while True:
            n_series = max(int(round((t_next - t_done) / (n_meas * p.dt))), 2)
            seg = np.zeros((n_series, self.pop.M), np.int64)
            tseg = np.zeros((n_series, K))
            for q in range(n_series):
                self.pop.advance(n_meas)
                Nc = self.pop.count()
                seg[q] = Nc
                self.hist_center += np.bincount(Nc[:cfg.M], minlength=K)
                bc = np.bincount(self._tiled_counts()[:cfg.M].ravel(), minlength=K)
                self.hist_tiles += bc
                tseg[q] = bc
            chunks.append(seg)
            tile_chunks.append(tseg)
            t_done = t_next
            allseries = np.concatenate(chunks)
            series = allseries[:, :cfg.M]
            # the horizon is set from the extra replicas only (independent of the x_0 samples)
            src = allseries[:, cfg.M:] if cfg.n_extra >= 4 else series
            self.acf_t, self.acf, self.tau_v, decayed = self._acf(src, n_meas * p.dt)
            if decayed or t_done >= cfg.t_prod_max:
                break
            self._log(f"  reservoir: N_v ACF not below 1/e within t_prod/2 = {t_done / 2:.2f}; "
                      f"extending production")
            t_next = min(2 * t_done, cfg.t_prod_max)
        if not decayed:
            self._log(f"  WARNING: tau_v not resolved; using an exponential fit of the ACF tail")
        tiles_t = np.concatenate(tile_chunks)
        nb = np.minimum(np.arange(tiles_t.shape[0]) * n_blocks // tiles_t.shape[0], n_blocks - 1)
        self.hist_tiles_blocks = np.stack([tiles_t[nb == b].sum(0) for b in range(n_blocks)])
        self.t_prod_used = t_done
        self.pos = self.pop.pos[:cfg.M].copy()
        self.theta = self.pop.theta[:cfg.M].copy()
        self.extra_pos = self.pop.pos[cfg.M:].copy()
        self.extra_theta = self.pop.theta[cfg.M:].copy()
        self.n_samples = self.pos.shape[0]
        self.var_N = float(np.var(series))
        self.mean_N = float(np.mean(series))
        self._log(f"  reservoir: {self.n_samples} samples, <N_v>={self.mean_N:.2f} "
                  f"(exact {self.N_tot * g.v / g.V:.2f}), Var N_v / v={self.var_N / g.v:.4f}, "
                  f"tau_v={self.tau_v:.3f} ({time.time() - t0:.0f}s)")
        return self

    def _acf(self, series, dt_meas):
        """Normalised ACF of N_v up to half the series length; tau_v = first 1/e crossing
        (linear interpolation), or from an exponential fit of the tail if not reached."""
        cfg = self.cfg
        n = series.shape[0]
        x = series.astype(float) - series.mean()
        var = np.mean(x * x)
        step = max(int(round(cfg.acf_dt / dt_meas)), 1)
        lags = np.arange(0, n // 2 + 1, step)
        acf = np.array([np.mean(x[:n - k] * x[k:]) / var for k in lags])
        t = lags * dt_meas
        target = np.exp(-1.0)
        below = np.flatnonzero(acf < target)
        if below.size:
            i = below[0]
            t0, t1, a0, a1 = t[i - 1], t[i], acf[i - 1], acf[i]
            tau = t0 + (a0 - target) / (a0 - a1) * (t1 - t0)
            return t, acf, float(tau), True
        good = acf > 0.05
        sl = np.polyfit(t[good][len(t[good]) // 2:], np.log(acf[good][len(t[good]) // 2:]), 1)[0] \
            if good.sum() > 4 else -1.0 / max(t[-1], 1e-9)
        return t, acf, float(min(-1.0 / sl if sl < 0 else 10 * t[-1], 10 * t[-1])), False

    def measure_self_propagation(self, s_lag=None, n_sub=None):
        """Effective (v, Dt) of a tagged particle from <dr . e(theta_0)> and <dr^2> at lag s.

        Used to adapt the one-body twist to interacting particles.
        """
        p = self.params
        if s_lag is None:
            s_lag = 1.0 / p.Dr
        # extra replicas only: the twist must not depend on the x_0 used by the estimator
        n_sub = self.extra_pos.shape[0] if n_sub is None else min(n_sub, self.extra_pos.shape[0])
        if n_sub < 1:
            raise ValueError("ReservoirConfig.n_extra must be >= 1 to measure twist parameters")
        sub = ABPPopulation(p, self.geom, n_sub, seed=[self.cfg.seed, 0xA, 2])
        sub.pos = self.extra_pos[:n_sub].copy()
        sub.theta = self.extra_theta[:n_sub].copy()
        sub.rng = new_rng_states(n_sub, np.random.default_rng([self.cfg.seed, 0xA, 3]))
        th0 = sub.theta.copy()
        disp = np.zeros_like(sub.pos)
        prev = sub.pos.copy()
        n = int(round(s_lag / p.dt))
        chunk = 20
        for _ in range(n // chunk):
            sub.advance(chunk)
            d = sub.pos - prev
            d -= self.geom.L * np.round(d / self.geom.L)
            disp += d
            prev = sub.pos.copy()
        s = (n // chunk) * chunk * p.dt
        proj = np.mean(disp[:, 0] * np.cos(th0) + disp[:, 1] * np.sin(th0))
        msd = np.mean(np.sum(disp ** 2, axis=1))
        g = (1.0 - np.exp(-p.Dr * s)) / p.Dr
        v_eff = proj / g
        Dt_eff = (msd - 2.0 * (v_eff / p.Dr) ** 2 * (p.Dr * s - 1.0 + np.exp(-p.Dr * s))) / (4.0 * s)
        return float(v_eff), float(max(Dt_eff, 1e-3)), float(s)

    def brute_force_lnP(self, tiles=True):
        h = self.hist_tiles if tiles else self.hist_center
        with np.errstate(divide="ignore"):
            return np.log(h / h.sum())


def run_twisted_smc(params, geom, cfg, reservoir, sample_idx, lam, horizon, twist_v, twist_Dt,
                    record_traj=False):
    """One finite-horizon twisted FK-SMC run at tilt ``lam``.

    sample_idx: indices of the reservoir configurations used as x_0 (length M).
    Returns a dict with ln Z_hat, the final weighted N_v sample, and diagnostics.
    """
    p, g = params, geom
    if cfg.control_every < 1 or cfg.n_weight < 1:
        raise ValueError("control_every and n_weight must be >= 1")
    if cfg.guided and p.force_cap > 0:
        raise ValueError("guided dynamics are implemented without a force cap")
    base = [int(x) for x in np.atleast_1d(cfg.seed)]
    gen = np.random.default_rng(base + [0])     # resampling + initial noise streams
    M = len(sample_idx)
    pop = ABPPopulation(p, g, M, seed=base + [1])  # its master generator seeds children's streams
    pop.pos = np.ascontiguousarray(reservoir.pos[sample_idx])
    pop.theta = np.ascontiguousarray(reservoir.theta[sample_idx])
    pop.rng = new_rng_states(M, gen)

    n_total = max(int(round(horizon / p.dt)), 0)
    n_w = max(int(cfg.n_weight), 1)
    n_k = max(int(np.ceil(n_total / n_w)), 1) if n_total > 0 else 0
    lo, hi, L = g.lo, g.hi, g.L

    def U_at(remaining):
        return twist_energy(pop.pos, pop.theta, max(remaining, 0.0), lam, twist_v, twist_Dt,
                            p.Dr, lo, hi, L)

    U_prev = U_at(n_total * p.dt)
    logw = -U_prev.copy()
    lnZ_acc = 0.0
    n_res = 0
    ess_min = 1.0
    ancestors = np.arange(M)
    steps_done = 0
    var_logw_max = 0.0
    trace = []
    for k in range(n_k + 1):
        if k > 0:
            n = min(n_w, n_total - steps_done)
            if n > 0:
                if cfg.guided:
                    logPQ = np.zeros(pop.M)
                    failed = np.zeros(pop.M, np.bool_)
                    advance_controlled(pop.pos, pop.theta, pop.rng, n, (n_total - steps_done) * p.dt,
                                       float(lam), float(twist_v), float(twist_Dt), float(L),
                                       float(p.v0), float(p.Dt), float(p.Dr), float(p.eps), float(p.dt),
                                       bool(p.interacting), float(lo), float(hi), int(cfg.control_every),
                                       float(cfg.s_floor_steps * p.dt), float(cfg.fine_steps * p.dt),
                                       logPQ, failed)
                    check_failed(failed, "advance_controlled")
                    logw += logPQ
                else:
                    pop.advance(n)
                steps_done += n
            U_new = U_at((n_total - steps_done) * p.dt)
            logw += -U_new + U_prev
            U_prev = U_new
        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w * w) / M
        ess_min = min(ess_min, ess)
        var_logw_max = max(var_logw_max, float(np.var(logw)))
        if record_traj:
            trace.append((steps_done * p.dt, lnZ_acc + _logmeanexp(logw), ess))
        if ess < cfg.ess_threshold and k < n_k:
            lnZ_acc += _logmeanexp(logw)
            idx = systematic_resample(w, gen)
            pop.resample(idx)
            U_prev = U_prev[idx]
            ancestors = ancestors[idx]
            logw[:] = 0.0
            n_res += 1
    lnZ = lnZ_acc + _logmeanexp(logw)
    N_final = count_in_square(pop.pos, lo, hi)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return dict(lnZ=float(lnZ), N=N_final, w=w, ess_final=float(1.0 / np.sum(w * w) / M),
                ess_min=float(ess_min), n_resample=n_res, n_ancestors=int(np.unique(ancestors).size),
                var_logw_max=var_logw_max, steps=n_total, trace=np.array(trace))


def run_lambda_grid(params, geom, smc_cfg, reservoir, lambdas, n_replicates, horizon,
                    twist_v, twist_Dt, verbose=True):
    """Runs every lambda with ``n_replicates`` independent replicate runs.

    Replicate r uses the reservoir samples [r*M, (r+1)*M) for every lambda, so
    different replicates are independent while different lambdas share x_0.
    """
    M = smc_cfg.M
    if reservoir.n_samples < n_replicates * M:
        raise ValueError("reservoir too small for the requested replicates")
    K = reservoir.N_tot + 1
    lnZ = np.zeros((len(lambdas), n_replicates))
    hist = np.zeros((len(lambdas), n_replicates, K))
    diag = []
    for a, lam in enumerate(lambdas):
        t0 = time.time()
        for r in range(n_replicates):
            idx = np.arange(r * M, (r + 1) * M)
            # distinct SeedSequence entropy per (lambda, replicate): no stream collisions
            cfg_r = SMCConfig(**{**smc_cfg.to_dict(), "seed": [int(smc_cfg.seed), 0xB, a, r]})
            out = run_twisted_smc(params, geom, cfg_r, reservoir, idx, float(lam), horizon,
                                  twist_v, twist_Dt)
            lnZ[a, r] = out["lnZ"]
            hist[a, r] = np.bincount(out["N"], weights=out["w"], minlength=K)
            diag.append(dict(lam=float(lam), rep=r, ess_min=out["ess_min"],
                             ess_final=out["ess_final"], n_resample=out["n_resample"],
                             n_ancestors=out["n_ancestors"], var_logw_max=out["var_logw_max"]))
        if verbose:
            d = [x for x in diag if x["lam"] == float(lam)]
            print(f"  lambda={lam:+.3f}  lnZ={np.mean(lnZ[a]):9.3f} +- "
                  f"{np.std(lnZ[a], ddof=1) / np.sqrt(n_replicates) if n_replicates > 1 else 0:.3f}  "
                  f"<N>={np.sum(hist[a].mean(0) * np.arange(K)):8.2f}  "
                  f"ESSmin={np.mean([x['ess_min'] for x in d]):.2f}  "
                  f"resamplings={np.mean([x['n_resample'] for x in d]):.1f}  "
                  f"ancestors={np.mean([x['n_ancestors'] for x in d]):.0f}  ({time.time() - t0:.0f}s)",
                  flush=True)
    return dict(lambdas=np.asarray(lambdas, float), lnZ_reps=lnZ, hist_reps=hist, diagnostics=diag)


def combine_replicates(lnZ_reps):
    """ln of the replicate-averaged Z (unbiased in Z) and a jackknife standard error."""
    R = lnZ_reps.shape[-1]
    lnZ = logsumexp(lnZ_reps, axis=-1) - np.log(R)
    if R < 2:
        return lnZ, np.full_like(lnZ, np.nan)
    jack = np.stack([logsumexp(np.delete(lnZ_reps, r, axis=-1), axis=-1) - np.log(R - 1)
                     for r in range(R)], axis=-1)
    err = np.sqrt((R - 1) / R * np.sum((jack - jack.mean(-1, keepdims=True)) ** 2, axis=-1))
    return lnZ, err

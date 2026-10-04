#!/usr/bin/env python
"""Production driver: static large deviations of the subvolume density of 2D ABPs.

For every subvolume side ell (box side L = kappa * ell, so v/V is fixed) this
  1. builds a reservoir of independent steady-state configurations (unbiased
     dynamics) and records brute-force histograms of N_v and its autocorrelation;
  2. measures effective self-propulsion parameters for the one-body twist;
  3. runs Doob-guided, twisted Feynman-Kac SMC for every biasing field lambda,
     with independent replicates, estimating Z_v(lambda) = E[exp(lambda N_v)]
     and sampling the tilted ensembles;
  4. saves everything to <out>/ell_<ell>.npz (resumable: existing files are skipped).

Run ``scripts/analyze_subvolume_ldp.py <out>`` afterwards for the rate functions,
finite-size scaling and figures.

Examples
--------
    python scripts/run_subvolume_ldp.py --out results/abp_pe24 --ells 6 8 10 12 14
    python scripts/run_subvolume_ldp.py --preset ideal --out results/ideal --ells 6 8 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from abp_ldp.simulator import ABPParams, Geometry  # noqa: E402
from abp_ldp.smc import (ReservoirConfig, Reservoir, SMCConfig, combine_replicates,  # noqa: E402
                         run_lambda_grid)

PRESETS = {
    # homogeneous active fluid below MIPS: Pe = 3 v0 / (sigma Dr) = 24, phi = pi rho0 / 4 = 0.31
    "default": dict(rho0=0.4, v0=24.0, Dr=3.0, dt=1e-4, interacting=True),
    # non-interacting ABPs: exact binomial reference for validation
    "ideal": dict(rho0=0.4, v0=24.0, Dr=3.0, dt=1e-4, interacting=False),
    # passive WCA fluid (equilibrium reference; slow: tau_v ~ ell^2 / (2 pi^2 Dt))
    "passive": dict(rho0=0.4, v0=0.0, Dr=3.0, dt=1e-4, interacting=True),
    # inside the MIPS binodal: Pe = 120, phi = pi rho0 / 4 = 0.8, eps = 1 (as in the GNN notebook)
    "mips": dict(rho0=4 * 0.8 / 3.141592653589793, v0=120.0, Dr=3.0, dt=2e-5, eps=1.0, interacting=True),
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--preset", default="default", choices=sorted(PRESETS))
    ap.add_argument("--ells", type=float, nargs="+", default=[6, 8, 10, 12, 14])
    ap.add_argument("--kappa", type=float, default=3.0, help="L / ell (fixed)")
    ap.add_argument("--rho0", type=float)
    ap.add_argument("--v0", type=float)
    ap.add_argument("--Dr", type=float)
    ap.add_argument("--dt", type=float)
    ap.add_argument("--eps", type=float, help="WCA strength")
    ap.add_argument("--biasing", type=float, nargs="+",
                    default=[-1.25, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.25],
                    help="biasing fields lambda (conjugate to N_v)")
    ap.add_argument("--M", type=int, default=128, help="replicas per SMC run")
    ap.add_argument("--R", type=int, default=3, help="independent replicate runs per lambda")
    ap.add_argument("--horizon-factor", type=float, default=2.5, help="horizon T in units of tau_v")
    ap.add_argument("--t-burn", type=float, default=5.0)
    ap.add_argument("--t-prod", type=float, default=2.0)
    ap.add_argument("--t-prod-max", type=float, default=16.0,
                    help="production is extended (doubling) until the N_v ACF decays, up to this time")
    ap.add_argument("--n-extra", type=int, default=32,
                    help="extra replicas (never x_0) that set tau_v and the twist parameters")
    ap.add_argument("--no-smc", action="store_true",
                    help="unbiased (brute-force) sampling only: M*R independent replicas, no tilted runs")
    ap.add_argument("--unguided", action="store_true", help="twist only, no Doob control")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    a = parse()
    pr = dict(PRESETS[a.preset])
    for k in ("rho0", "v0", "Dr", "dt", "eps"):
        if getattr(a, k) is not None:
            pr[k] = getattr(a, k)
    params = ABPParams(**pr)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "run_config.json"), "w") as fh:
        json.dump(dict(args=vars(a), params=params.to_dict()), fh, indent=2)
    print(f"ABP parameters: {params.to_dict()}  Pe={params.peclet:.1f}", flush=True)
    for i, ell in enumerate(a.ells):
        fn = os.path.join(a.out, f"ell_{ell:g}.npz")
        if os.path.exists(fn) and not a.overwrite:
            print(f"[ell={ell:g}] exists, skipping", flush=True)
            continue
        t0 = time.time()
        geom = Geometry(ell=ell, kappa=a.kappa)
        print(f"[ell={ell:g}] L={geom.L:g} v={geom.v:g} V={geom.V:g} N={geom.n_particles(params.rho0)}",
              flush=True)
        rcfg = ReservoirConfig(M=a.M * a.R, t_burn=a.t_burn, t_prod=a.t_prod, t_prod_max=a.t_prod_max,
                               n_extra=a.n_extra, seed=a.seed + 100 * i)
        # reservoir streams use SeedSequence([seed, k]); SMC streams use ([seed, lambda, rep, k]),
        # so the two families can never collide
        res = Reservoir(params, geom, rcfg).build()
        if params.interacting:
            v_eff, Dt_eff, s_lag = res.measure_self_propagation()
        else:
            v_eff, Dt_eff, s_lag = params.v0, params.Dt, 0.0
        horizon = a.horizon_factor * res.tau_v
        print(f"[ell={ell:g}] twist parameters v_eff={v_eff:.3f} Dt_eff={Dt_eff:.3f}; "
              f"horizon T={horizon:.3f}", flush=True)
        scfg = SMCConfig(M=a.M, guided=not a.unguided, seed=a.seed + 7 + 100 * i)
        lams = [] if a.no_smc else a.biasing
        if lams:
            out = run_lambda_grid(params, geom, scfg, res, lams, a.R, horizon, v_eff, Dt_eff)
            lnZ, err = combine_replicates(out["lnZ_reps"])
        else:
            out = dict(lambdas=np.zeros(0), lnZ_reps=np.zeros((0, a.R)),
                       hist_reps=np.zeros((0, a.R, res.N_tot + 1)), diagnostics=[])
            lnZ, err = np.zeros(0), np.zeros(0)
        np.savez_compressed(
            fn,
            params=json.dumps(params.to_dict()), geometry=json.dumps(geom.to_dict()),
            smc_config=json.dumps(scfg.to_dict()), reservoir_config=json.dumps(rcfg.to_dict()),
            N_tot=res.N_tot, lambdas=out["lambdas"], lnZ_reps=out["lnZ_reps"], hist_reps=out["hist_reps"],
            lnZ=lnZ, lnZ_err=err, diagnostics=json.dumps(out["diagnostics"]),
            hist_tiles=res.hist_tiles, hist_tiles_blocks=res.hist_tiles_blocks, hist_center=res.hist_center,
            acf=res.acf, acf_t=res.acf_t, tau_v=res.tau_v, var_N=res.var_N, mean_N=res.mean_N,
            v_eff=v_eff, Dt_eff=Dt_eff, horizon=horizon, t_prod_used=res.t_prod_used,
            snapshot_pos=res.pos[:4].astype(np.float32), snapshot_theta=res.theta[:4].astype(np.float32),
            wall_time=time.time() - t0,
        )
        print(f"[ell={ell:g}] done in {time.time() - t0:.0f}s -> {fn}", flush=True)


if __name__ == "__main__":
    main()

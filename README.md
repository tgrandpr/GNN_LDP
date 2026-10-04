# GNN_LDP
Implement a scalable graph neural network algorithm for dynamical large deviation calculations

## Static large deviations of the subvolume density of active Brownian particles (`abp_ldp/`)

The theory behind the code is summarised in [`docs/THEORY.md`](docs/THEORY.md); results for a
homogeneous state (Pe = 24) and a state inside the MIPS binodal (Pe = 120, phi = 0.8) are in
[`docs/RESULTS.md`](docs/RESULTS.md).

This package computes the large-deviation function of the **particle density in a
subvolume** of a two-dimensional system of active Brownian particles (ABPs), with the
**subvolume area `v` as the large parameter** (not time). The observable is the
instantaneous number `N_v` of particles in a central `ell x ell` square of an `L x L`
periodic box:

```
P_v(N_v = rho v)   ~  exp[-v I(rho)]                        (v = ell^2 -> infinity)
psi_v(lambda)      =  (1/v) ln < exp(lambda N_v) >_ss  ->  psi(lambda)
I(rho)             =  sup_lambda [lambda rho - psi(lambda)]  (where I is convex)
```

`<.>_ss` is the nonequilibrium steady state of the ABPs. The biasing field `lambda`
(called `biasing` in the scripts, as in the GNN notebooks) tilts the statistics of the
subvolume density fluctuations. To show the large-deviation scaling, the box is scaled with
the subvolume, `L = kappa * ell` (default `kappa = 3`, so `f = v/V = 1/9` is fixed), and
`-(1/v) ln P_v` is shown to collapse as `ell` grows.

### Model

Overdamped ABPs integrated with Euler–Maruyama (Ito), units `sigma = mu = Dt = 1`:

```
dr_i     = [v0 e(theta_i) + F_i] dt + sqrt(2 Dt) dW_i,    dtheta_i = sqrt(2 Dr) dW'_i
```

`F_i` is the WCA repulsion (`eps = 1`, cutoff `2^(1/6)`). The default preset is `rho0 = 0.4`
(`phi = pi rho0 / 4 = 0.31`), `v0 = 24`, `Dr = 3`, `Pe = 3 v0 / (sigma Dr) = 24`, `dt = 1e-4`.
That is a homogeneous active fluid below motility-induced phase separation. The simulator
(`abp_ldp/simulator.py`) is numba-parallel over replicas. Each replica has its own cell list
and its own SplitMix64 random stream, so a run is reproducible from its seed.

### Method: why this algorithm

The steady state of ABPs is **not** a Boltzmann distribution and is not known. Three
consequences:

* Metropolis Monte Carlo on `P_ss(x) exp(lambda N_v(x))` is impossible.
* Adding a bias force to the dynamics changes the steady state in an unknown way, so it
  cannot be reweighted afterwards.
* Plain cloning with a fixed tilt fails exponentially in `v`. The unbiased dynamics keeps
  relaxing the tilted population back to `P_ss`, and the weights must pay for it. The variance
  grows like `exp(lambda^2 Var N_v)`. This is verified in the development tests.

We use a **finite-horizon, Doob-guided, twisted Feynman–Kac SMC** (`abp_ldp/smc.py`,
`abp_ldp/twist.py`). It is exact for any steady state:

1. A **reservoir** of independent steady-state configurations `x_0 ~ P_ss` comes from
   unbiased runs, one sample per independent trajectory.
2. `M` replicas evolve for a horizon `T ~ 2.5 tau_v` (`tau_v` = 1/e time of the `N_v`
   autocorrelation) under a *guided* dynamics. The guided dynamics adds a control force
   `u_i = 2 Dt grad_i phi` and a torque `w_i = 2 Dr d phi / d theta_i`. Here
   `phi_s(r, theta) = ln[1 + (e^lambda - 1) p_s(r, theta)]`, and `p_s` is the probability that
   a free ABP starting at `(r, theta)` is inside the subvolume after the remaining time `s`
   (Gaussian approximation with the exact free-ABP moments). For non-interacting ABPs this
   is the exact Doob transform of the terminal tilt `exp(lambda N_v(x_T))`. For interacting
   ABPs it uses effective `(v_eff, Dt_eff)` measured from tagged particles. This is the
   static analogue of the learned control forces in `gnn_active_Brownian_particles.ipynb`.
3. The weights combine the **exact discrete-time Girsanov factor** `dP/dQ` of every
   Euler–Maruyama step with the **twist** `exp(-U_k(x_k) + U_{k-1}(x_{k-1}))`, where
   `U_k = -sum_i phi_{T - t_k}(r_i, theta_i)`. At the final time `U_K = -lambda N_v` exactly.
   Replicas are resampled (systematic resampling) when the ESS drops below `M/2`.

The weights telescope to `exp(lambda N_v(x_T)) dP/dQ`. Because `x_0 ~ P_ss` and the
unbiased dynamics preserve `P_ss`:

* the product of mean weights is an **unbiased estimator of `Z_v(lambda) = <exp(lambda N_v)>`**;
* the final weighted population samples the tilted ensemble `P_ss exp(lambda N_v)/Z`.

This holds for any twist and control. Their quality only affects the variance. Errors come
from `R` independent replicate runs, averaging `Z` (not `ln Z`) with jackknife errors.

`ln P_v(N)` over the full density range is reconstructed by WHAM. WHAM combines the tilted
histograms of all `lambda` and the unbiased histogram from the reservoir (all `kappa^2`
tiled placements of the subvolume; the steady state is translation invariant).
`abp_ldp/analysis.py` then gives:

* `I_v(rho)`;
* the Legendre pairs `(rho_lambda, lambda rho_lambda - psi_v(lambda))`;
* `1/ell` extrapolations at fixed `lambda` and at fixed `rho`;
* an estimate of the infinite-reservoir (`f -> 0`) rate function. It assumes additivity,
  `I_f(rho) = I_0(rho) + ((1 - f)/f) I_0(rho_out)`.

### Validation

* `tests/`: forces vs brute force and vs `-grad U`; the cell list; free-ABP MSD and
  orientation decay; RNG reproducibility; twist limits and gradients; and SMC vs the
  **exact binomial** `ln Z = N ln(1 - f + f e^lambda)` for non-interacting ABPs.
  Run with `python -m pytest -q`.
* `--preset ideal` repeats the production protocol for non-interacting ABPs, where
  `P_v(N)` is exactly binomial, at every `ell` (validation panel of `fig_validation.png`).
* For interacting ABPs, SMC+WHAM is compared with unbiased brute-force histograms wherever
  the latter are reliable.

### Usage

```bash
pip install numpy scipy numba matplotlib pytest
python -m pytest -q
# production (resumable; one npz per subvolume size)
python scripts/run_subvolume_ldp.py --out results/abp_pe24 --ells 6 8 10 12 14
python scripts/run_subvolume_ldp.py --preset ideal --out results/ideal --ells 6 8 10 12 14
# analysis + figures
python scripts/analyze_subvolume_ldp.py results/abp_pe24 --ideal results/ideal
```

Main options of `run_subvolume_ldp.py`:

| option | meaning |
|---|---|
| `--biasing` | list of `lambda` values |
| `--M` | replicas per run |
| `--R` | independent replicates |
| `--horizon-factor` | `T / tau_v` |
| `--kappa` | `L / ell` |
| `--rho0`, `--v0`, `--Dr`, `--dt` | model parameters |
| `--unguided` | twist without control, for comparison |
| `--preset mips` | inside the MIPS binodal: Pe = 120, phi = 0.8, eps = 1, dt = 2e-5 |
| `--no-smc` | unbiased (brute-force) sampling only |
| `--eps`, `--n-extra`, `--t-prod-max` | WCA strength, tuning replicas, maximum production time |

Outputs in the results directory:

| file | content |
|---|---|
| `fig_volume_scaling.png` | `-ln P_v` vs `rho` (unscaled) and `-(1/v) ln P_v` (collapse) |
| `fig_scgf.png` | `psi_v(lambda)` for all `v`, and its `1/ell` extrapolation |
| `fig_rate_function.png` | `I_v(rho)`, Legendre points, the `v -> infinity` limit, the Gaussian approximation |
| `fig_validation.png` | SMC vs brute force (interacting) and vs exact binomial (ideal) |
| `fig_diagnostics.png` | ESS, population diversity, `N_v` autocorrelation |
| `fig_coexistence_scaling.png` | `-ln P_v` per unit volume vs per unit length (bulk vs interfaces) |
| `fig_snapshots.png` | a steady-state configuration per size, with the subvolume outlined |
| `summary.json`, `summary.md` | all numbers |

### Caveats

* **Finite reservoir.** At fixed `kappa` the limit is the rate function at fixed
  `f = v/V`, not the infinite-reservoir one. Both are reported.
* **Persistence length.** At `v0 = 24` the persistence length is `v0/Dr = 8 sigma`. The
  effective value under collisions is about 5. Subvolumes with `ell <~ 14` are only a few
  persistence lengths across, so `1/ell` corrections are sizeable.
* **Cost.** Cost grows roughly like `ell^4` per run at fixed `M`: `N`, `tau_v ~ ell^2`.
  The variance of `ln Z` grows like `v I / M`, so the largest `|lambda|` at the largest `ell`
  need larger `M`.
* **Non-convex regions.** Where `I(rho)` is non-convex (e.g. near freezing at high `rho`, or
  inside MIPS coexistence), exponential tilting only gives the convex hull. Inside MIPS
  coexistence at fixed `f` there is no `exp(-v I)` scaling at all (interface physics,
  `-ln P ~ ell`). The MIPS regime needs umbrella-type biases and a smaller `dt`
  (the GNN notebook uses `8e-6` at `v0 = 100`).

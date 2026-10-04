# Subvolume-density large deviations: summary

Parameters: {'rho0': 1.0185916357881302, 'v0': 120.0, 'Dt': 1.0, 'Dr': 3.0, 'eps': 1.0, 'dt': 2e-05, 'interacting': True, 'force_cap': 0.0}  
kappa = L/ell = 3, f = v/V = 0.1111, global density rho_bar = 1.0187

| ell | v | N | chi_v = Var(N_v)/v | <N_v>/v (rho_bar) | tau_v | v_eff | Dt_eff | horizon | max abs(lnZ_TI - lnZ_WHAM) | wall time [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 64 | 587 | 3.0718 | 1.0409 (1.0191) | 0.559 | 29.88 | 35.90 | 1.40 | nan | 1123 |
| 12 | 144 | 1320 | 16.4262 | 1.0399 (1.0185) | 7.153 | 23.88 | 48.29 | 17.88 | nan | 2202 |
| 16 | 256 | 2347 | 32.8751 | 0.9811 (1.0187) | 42.597 | 18.83 | 52.20 | 106.49 | nan | 2513 |

lnZ_TI: trapezoidal thermodynamic integration of <N_v>_lambda over the lambda grid (a coarse-grid consistency check, accurate to O(dlambda^2 Var')).

SMC ln Z vs WHAM ln Z (flag * = SMC estimate degenerate: inconsistent with WHAM or violating convexity; WHAM is used):

- ell=8: 
- ell=12: 
- ell=16: 

psi_v(lambda) = (1/v) ln E[exp(lambda N_v)] (WHAM normalisation, per-replicate errors):

| lambda | ell=8 | ell=12 | ell=16 | v -> inf |
|---|---|---|---|---|
| +0.00 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0002 |

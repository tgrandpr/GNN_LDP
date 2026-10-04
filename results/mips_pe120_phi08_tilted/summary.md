# Subvolume-density large deviations: summary

Parameters: {'rho0': 1.0185916357881302, 'v0': 120.0, 'Dt': 1.0, 'Dr': 3.0, 'eps': 1.0, 'dt': 2e-05, 'interacting': True, 'force_cap': 0.0}  
kappa = L/ell = 3, f = v/V = 0.1111, global density rho_bar = 1.0191

| ell | v | N | chi_v = Var(N_v)/v | <N_v>/v (rho_bar) | tau_v | v_eff | Dt_eff | horizon | max abs(lnZ_TI - lnZ_WHAM) | wall time [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 64 | 587 | 3.1528 | 1.0217 (1.0191) | 0.866 | 31.85 | 34.46 | 2.17 | 1.69 | 10206 |

lnZ_TI: trapezoidal thermodynamic integration of <N_v>_lambda over the lambda grid (a coarse-grid consistency check, accurate to O(dlambda^2 Var')).

SMC ln Z vs WHAM ln Z (flag * = SMC estimate degenerate: inconsistent with WHAM or violating convexity; WHAM is used):

- ell=8: -1.00: -23.44/-25.64, -0.50: -17.76/-17.56, -0.25: -11.17/-11.02, +0.25: 19.47/19.69, +0.50: 40.50/41.90*, +1.00: 81.75/89.49*

psi_v(lambda) = (1/v) ln E[exp(lambda N_v)] (WHAM normalisation, per-replicate errors):

| lambda | ell=8 | v -> inf |
|---|---|---|
| -1.00 | -0.4007 ± 0.0247 | nan ± nan |
| -0.50 | -0.2744 ± 0.0068 | nan ± nan |
| -0.25 | -0.1722 ± 0.0011 | nan ± nan |
| +0.00 | 0.0000 ± 0.0000 | nan ± nan |
| +0.25 | 0.3077 ± 0.0001 | nan ± nan |
| +0.50 | 0.6547 ± 0.0010 | nan ± nan |
| +1.00 | 1.3983 ± 0.0061 | nan ± nan |

# Theory note: static large deviations of the subvolume density in active Brownian particles

This note explains what `abp_ldp/` computes and why the algorithm is exact. Every
symbol below appears in the code under the same name.

## 1. Model and observable

There are $N$ active Brownian particles in an $L\times L$ periodic box, in units $\sigma=\mu=D_t=1$:

$$
\mathrm d\mathbf r_i=\big[v_0\,\mathbf e(\theta_i)+\mathbf F_i\big]\mathrm dt+\sqrt{2D_t}\,\mathrm d\mathbf W_i,
\qquad
\mathrm d\theta_i=\sqrt{2D_r}\,\mathrm dW_i' ,
$$

where $\mathbf F_i$ is the WCA force. We integrate with Euler–Maruyama at step $\Delta t$. All
statements below hold exactly for this **discrete-time chain**, whose steady state we call
$P_{ss}$.

The observable is the instantaneous number of particles in a central square of side $\ell$
(area $v=\ell^2$),

$$
N_v(x)=\sum_i \mathbf 1\big[\mathbf r_i\in[\tfrac{L-\ell}{2},\tfrac{L+\ell}{2})^2\big],\qquad \rho_v=N_v/v .
$$

## 2. Large deviations with the volume as the large parameter

The large parameter is the subvolume area $v$, not time. Nothing is time-integrated:

$$
P_v(\rho_v=\rho)\asymp e^{-v\,I(\rho)},\qquad
\psi_v(\lambda)=\frac1v\ln\big\langle e^{\lambda N_v}\big\rangle_{ss}\xrightarrow[v\to\infty]{}\psi(\lambda),
\qquad I(\rho)=\sup_\lambda\big[\lambda\rho-\psi(\lambda)\big].
$$

The last relation (Gärtner–Ellis) holds where $I$ is convex. The field $\lambda$
(`biasing` in the scripts) tilts the subvolume density fluctuations. The tilted ensemble is

$$
\pi_\lambda(x)=\frac{P_{ss}(x)\,e^{\lambda N_v(x)}}{Z_v(\lambda)},\qquad
Z_v(\lambda)=\langle e^{\lambda N_v}\rangle_{ss},\qquad
\langle\rho_v\rangle_\lambda=\psi_v'(\lambda).
$$

**Box scaling.** We scale the box with the subvolume, $L=\kappa\ell$, so $f=v/V=\kappa^{-2}$
is fixed. The limit is then the rate function $I_f$ of a subvolume coupled to a finite
reservoir. For ideal particles $N_v\sim\mathrm{Bin}(N,f)$, which gives

$$
I_f(\rho)=\rho\ln\frac{\rho}{\bar\rho}+\frac{1-f}{f}\,\rho_{\rm out}\ln\frac{\rho_{\rm out}}{\bar\rho},
\qquad \rho_{\rm out}=\frac{\bar\rho-f\rho}{1-f} .
$$

This reduces to the Poisson result $I_0=\rho\ln(\rho/\bar\rho)-\rho+\bar\rho$ as $f\to0$.

If the static fluctuations are additive (exact in equilibrium with short-range forces, a
*hypothesis* for ABPs), then in general

$$
I_f(\rho)=I_0(\rho)+\frac{1-f}{f}\,I_0(\rho_{\rm out}).
$$

`analysis.deconvolve_finite_reservoir` inverts this relation. It fixes the gauge
$I_0(\bar\rho)=I_0'(\bar\rho)=0$ and uses a damped iteration.

**Finite-size corrections.** Write $P_v(\rho v)\simeq C(\rho)\,v^{-1/2}e^{-vI(\rho)-\ell\,b(\rho)}$.
Here $b$ is a boundary (perimeter) term, nonzero for interacting particles. Shifting the
minimum of $-\frac1v\ln P_v$ to zero cancels the $\ln v$ prefactor, so the leading correction
is $O(1/\ell)$. Both $\psi_v$ and $I_v$ are therefore extrapolated linearly in $1/\ell$.

## 3. Why the tilted ensemble is hard to sample for an active system

$P_{ss}$ is a nonequilibrium steady state. It is not Boltzmann and it is unknown. That rules
out three standard approaches:

* **Metropolis Monte Carlo** on $P_{ss}e^{\lambda N_v}$ needs $P_{ss}$ itself.
* **A bias potential $U$ in the dynamics** does not produce $P_{ss}e^{-U}$. With the
  steady-state current $\mathbf J_{ss}\neq0$, one finds
  $\mathcal L_U^\dagger(P_{ss}e^{-U})=e^{-U}\,\nabla U\cdot\mathbf J_{ss}\neq0$.
* **Cloning with a tilt held fixed in time** has weights that fight the dynamics: the
  unbiased dynamics keeps relaxing the tilted population back to $P_{ss}$. The relative
  variance of the weights grows like
  $Z(\lambda)Z(-\lambda)\simeq e^{\lambda^2\,\mathrm{Var}\,N_v}$, i.e. exponentially in $v$.
  We observed this collapse directly in the development tests.

## 4. Feynman–Kac identity with a terminal tilt

Draw $x_0\sim P_{ss}$ and evolve with the unbiased chain $P$ up to time $T$. Since $P$
preserves $P_{ss}$,

$$
\mathbb E\big[e^{\lambda N_v(x_T)}\big]=Z_v(\lambda),\qquad
\mathbb E\big[e^{\lambda N_v(x_T)}\delta(x_T-x)\big]=Z_v(\lambda)\,\pi_\lambda(x).
$$

Pick any intermediate *twists* $U_0,\dots,U_K$ (functions of the configuration) with
$U_K=-\lambda N_v$ exactly. The weights

$$
G_0=e^{-U_0(x_0)},\qquad G_k=e^{-U_k(x_{t_k})+U_{k-1}(x_{t_{k-1}})}
$$

telescope to $e^{\lambda N_v(x_T)}$. Sequential Monte Carlo with resampling then makes the
product of the mean weights over the resampling epochs an **unbiased** estimator of $Z_v$, and
its final weighted population samples $\pi_\lambda$.

This holds **for any twist**: the twist only controls the variance. Ramp speed and holding
time matter only through the variance too, which is why $\lambda$ is never held fixed (§3).

**Optimal twist.** The minimum-variance twist is the value function
$h_s(x)=\ln\mathbb E[e^{\lambda N_v(x_T)}\mid x_{T-s}=x]$. For non-interacting ABPs it is
exactly one-body:

$$
h_s(x)=\sum_i\phi_s(\mathbf r_i,\theta_i),\qquad
\phi_s=\ln\!\big[1+(e^\lambda-1)\,p_s(\mathbf r,\theta)\big],
$$

where $p_s$ is the probability that a free ABP starting at $(\mathbf r,\theta)$ is in the
subvolume after time $s$. We approximate the displacement by a Gaussian with the exact free-ABP
moments:

* mean $\frac{v}{D_r}(1-e^{-D_r s})\,\mathbf e(\theta)$;
* mean square displacement $4D_ts+2\frac{v^2}{D_r^2}(D_rs-1+e^{-D_rs})$.

The required periodic images are included. For interacting particles $v$ and $D_t$ are
replaced by effective values $(v_{\rm eff},D_{t,\rm eff})$, measured from tagged-particle
displacements on replicas that are never used as $x_0$ (`twist.py`).

## 5. Doob-guided dynamics and exact Girsanov weights

Variance falls much further if the dynamics are also steered by the Doob drift of the same
value function. The guided chain $Q$ adds a control force and a torque:

$$
\mathbf u_i=2D_t\,\nabla_{\mathbf r_i}\phi_s,\qquad w_i=2D_r\,\partial_{\theta_i}\phi_s .
$$

For non-interacting particles this is exactly the Doob transform of the terminal tilt, and
the importance weights become constant. Every Euler–Maruyama step has Gaussian transition
densities under both $P$ and $Q$. Using the standard normals $\boldsymbol\xi,\eta$ drawn by
$Q$, the log-likelihood ratio of one step is exact for the discrete chains:

$$
\ln\frac{\mathrm dP}{\mathrm dQ}
=-\mathbf u\cdot\boldsymbol\xi\sqrt{\tfrac{\Delta t}{2D_t}}-\frac{\Delta t\,|\mathbf u|^2}{4D_t}
-w\,\eta\sqrt{\tfrac{\Delta t}{2D_r}}-\frac{\Delta t\,w^2}{4D_r}.
$$

The total weight is $\prod_k G_k\times\prod_{\rm steps}\mathrm dP/\mathrm dQ$, so the estimator
stays unbiased for any control. This is the static, analytic counterpart of the control
forces learned by the GNNs in `gnn_active_Brownian_particles.ipynb`. A learned one-body or
graph control could replace $\phi_s$ to reduce the variance further.

## 6. Reconstructing $P_v(N)$ and the rate function

Each $\lambda$ yields the estimate $\hat Z_v(\lambda)$. We average $Z$, not $\ln Z$, over $R$
independent replicate runs, with jackknife errors. Each run also yields a weighted histogram
of $N_v$ sampled from $\pi_\lambda$. The tilted histograms and the unbiased brute-force
histogram are combined by WHAM. The unbiased histogram uses all $\kappa^2$ tiled placements of
the subvolume, which is valid because $P_{ss}$ is translation invariant. WHAM gives
$\ln P_v(N)$ over the whole density range, and

$$
I_v(\rho)=-\frac1v\Big[\ln P_v(\rho v)-\max_N\ln P_v(N)\Big].
$$

Independently, every tilted ensemble gives one point of the Legendre transform,

$$
\big(\rho_\lambda,\ I_\lambda\big)=\big(\langle N_v\rangle_\lambda/v,\ \ \lambda\rho_\lambda-\psi_v(\lambda)\big).
$$

Data collapse of $-\frac1v\ln P_v$ for increasing $v$, and convergence of $\psi_v$ in $1/\ell$,
demonstrate the volume large-deviation principle.

**Limitation.** Where $I$ is non-convex (near freezing, or inside motility-induced phase
separation), exponential tilting only reaches the convex hull. Inside the coexistence region
at fixed $f$, $-\ln P_v$ scales like $\ell$ (interfaces) rather than $v$.

## 7. Checks built into the code

* **Exact reference:** non-interacting ABPs have $N_v\sim\mathrm{Bin}(N,f)$, so
  $\ln Z_v=N\ln(1-f+fe^\lambda)$. SMC must reproduce it at every $\ell$ and $\lambda$
  (`tests/test_smc.py`, `--preset ideal`).
* **Brute force:** for interacting ABPs, SMC+WHAM must agree with the unbiased histograms
  wherever those are reliable.
* **Exact identities:**
  * $\psi_v(0)=0$;
  * $\psi_v'(0)=\bar\rho$ (translation invariance);
  * thermodynamic integration $\ln Z(\lambda)=\int_0^\lambda\langle N_v\rangle_{\lambda'}\,\mathrm d\lambda'$.
* **Diagnostics:** effective sample size, number of distinct $t=0$ ancestors, the $N_v$
  autocorrelation and its time $\tau_v$ (the horizon is $T\approx2.5\,\tau_v$), and
  simulator blow-up detection.

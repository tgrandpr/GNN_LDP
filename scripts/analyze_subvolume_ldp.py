#!/usr/bin/env python
"""Analysis of subvolume-density large deviations with the subvolume area v as the
large parameter.

Reads the ell_<ell>.npz files written by run_subvolume_ldp.py and produces

  * psi_v(lambda) = (1/v) ln E[exp(lambda N_v)]  for every v, and its v -> infinity limit
  * ln P_v(N) from WHAM over all tilted ensembles (+ the unbiased brute-force window)
  * I_v(rho)  = -(1/v) [ln P_v(rho v) - max ln P_v]  and the Legendre points of the
    tilted ensembles, rho_lambda = <N_v>_lambda / v,  I = lambda rho_lambda - psi_v(lambda)
  * finite-size extrapolation in 1/ell at fixed lambda and at fixed rho
  * the f -> 0 (infinite reservoir) rate function from the fixed-f one (additivity assumption)
  * figures (PNG) and summary.json / summary.md

Usage:
    python scripts/analyze_subvolume_ldp.py results/abp_pe24 [--ideal results/ideal]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from abp_ldp import analysis as an  # noqa: E402

# ---- palette: ordinal blue ramp for subvolume size, orange for the v -> inf limit ----
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]
ORANGE = "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def size_colors(n):
    idx = np.round(np.linspace(0, len(BLUE_RAMP) - 1, n)).astype(int) if n > 1 else [len(BLUE_RAMP) - 3]
    return [BLUE_RAMP[i] for i in idx]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6, "lines.markersize": 5,
        "errorbar.capsize": 0,
    })


# --------------------------------------------------------------------------
# Loading and per-size analysis
# --------------------------------------------------------------------------

def load_run(fn):
    d = np.load(fn, allow_pickle=False)
    geom = json.loads(str(d["geometry"]))
    params = json.loads(str(d["params"]))
    smc = json.loads(str(d["smc_config"]))
    rcfg = json.loads(str(d["reservoir_config"]))
    diag = json.loads(str(d["diagnostics"]))
    r = dict(file=fn, geom=geom, params=params, smc=smc, rcfg=rcfg, diag=diag)
    for k in ("N_tot", "lambdas", "lnZ_reps", "hist_reps", "lnZ", "lnZ_err", "hist_tiles",
              "hist_tiles_blocks", "hist_center", "acf", "acf_t", "tau_v", "var_N", "mean_N",
              "v_eff", "Dt_eff", "horizon", "wall_time"):
        r[k] = d[k]
    r["t_prod_used"] = float(d["t_prod_used"]) if "t_prod_used" in d else rcfg["t_prod"]
    r["snapshot_pos"] = d["snapshot_pos"] if "snapshot_pos" in d else None
    r["N_tot"] = int(r["N_tot"])
    r["ell"], r["v"], r["V"] = geom["ell"], geom["v"], geom["V"]
    r["f"] = r["v"] / r["V"]
    r["rho_bar"] = r["N_tot"] / r["V"]
    return r


def analyze_run(r):
    v, K = r["v"], r["N_tot"] + 1
    N = np.arange(K)
    lam = np.asarray(r["lambdas"], float)
    M = r["smc"]["M"]
    # pooled tilted histograms (each replicate's weighted histogram is normalised)
    hist = r["hist_reps"].mean(1)
    n_eff = np.zeros(lam.size)
    for dct in r["diag"]:
        a = int(np.argmin(np.abs(lam - dct["lam"])))
        n_eff[a] += max(1.0, min(dct["ess_final"] * M, dct["n_ancestors"]))
    # unbiased brute-force window (all kappa^2 tiled placements)
    bf = r["hist_tiles"] / r["hist_tiles"].sum()
    rc = r["rcfg"]
    n_tiles = max(1.0, round(r["geom"]["kappa"]) ** 2)
    n_eff_bf = rc["M"] * r["t_prod_used"] / max(2.0 * float(r["tau_v"]), rc["measure_every"]) * 0.5 * n_tiles
    H = np.vstack([bf * n_eff_bf, hist * n_eff[:, None]])
    U = np.vstack([np.zeros(K), -lam[:, None] * N[None, :]])
    lnP, fw = an.wham(H, U, f_init=np.concatenate([[0.0], -r["lnZ"]]))
    lnZ_wham = -(fw[1:] - fw[0])
    # Primary normalisation: self-consistent WHAM (set by the overlap of the tilted histograms),
    # with errors from per-replicate WHAM.  The SMC ln Z is kept as a cross-check: at the largest
    # tilts its finite-population (weight-degeneracy) bias makes it too low.
    R = r["hist_reps"].shape[1]
    lnZ_wham_reps = np.zeros((lam.size, R))
    for k in range(R):
        n_eff_k = np.zeros(lam.size)
        for dct in r["diag"]:
            if dct["rep"] == k:
                a = int(np.argmin(np.abs(lam - dct["lam"])))
                n_eff_k[a] = max(1.0, min(dct["ess_final"] * M, dct["n_ancestors"]))
        Hk = np.vstack([bf * n_eff_bf, r["hist_reps"][:, k] * n_eff_k[:, None]])
        _, fk = an.wham(Hk, U, f_init=np.concatenate([[0.0], -r["lnZ_reps"][:, k]]))
        lnZ_wham_reps[:, k] = -(fk[1:] - fk[0])
    lnZ_wham_err = (np.std(lnZ_wham_reps, axis=1, ddof=1) / np.sqrt(R)) if R > 1 else np.zeros(lam.size)
    # flag SMC ln Z that is inconsistent with WHAM or violates convexity:
    # ln Z(l2) >= ln Z(l1) + (l2 - l1) <N>_{l1}
    meanN_t = (hist * N).sum(1)
    smc_flag = np.abs(r["lnZ"] - lnZ_wham) > 3 * np.maximum(r["lnZ_err"], lnZ_wham_err) + 0.5
    order_l = np.argsort(lam)
    lam_s, lnZ_s, mN_s = lam[order_l], r["lnZ"][order_l], meanN_t[order_l]
    lam_full = np.concatenate([lam_s, [0.0]])
    lnZ_full = np.concatenate([lnZ_s, [0.0]])
    mN_full = np.concatenate([mN_s, [float(r["mean_N"])]])
    of = np.argsort(lam_full)
    lam_full, lnZ_full, mN_full = lam_full[of], lnZ_full[of], mN_full[of]
    conv_viol = np.zeros(lam.size, bool)
    for a in range(lam.size):
        j = int(np.flatnonzero(np.isclose(lam_full, lam[a]))[0])
        nb = j - 1 if lam[a] > 0 else j + 1     # neighbour towards lambda = 0
        bound = lnZ_full[nb] + (lam_full[j] - lam_full[nb]) * mN_full[nb]
        conv_viol[a] = lnZ_full[j] < bound - 3 * max(r["lnZ_err"][a], 1e-3)
    smc_flag |= conv_viol
    # tilted windows only, normalised with the SMC ln Z: independent of the brute-force data
    win_only = an.window_estimates(hist, -lam[:, None] * N[None, :], r["lnZ"])
    with np.errstate(divide="ignore"):
        w_cnt = hist * n_eff[:, None] * (~smc_flag)[:, None]
    lnP_tilt = np.full(K, -np.inf)
    good = w_cnt.sum(0) > 0
    num = np.where(np.isfinite(win_only), w_cnt * np.exp(np.where(np.isfinite(win_only), win_only, 0.0)), 0.0)
    lnP_tilt[good] = np.log(np.maximum(num[:, good].sum(0), 1e-300)) - np.log(w_cnt[:, good].sum(0))
    tilt_counts = w_cnt.sum(0)
    # exact identities: psi_v'(0) = rho_bar and thermodynamic integration vs SMC ln Z
    lam_ti = np.concatenate([lam, [0.0]])
    mean_ti = np.concatenate([(hist * N).sum(1), [float(r["mean_N"])]])
    lnZ_ti = an.thermodynamic_integration(lam_ti, mean_ti)[:-1]
    rho, I = an.rate_function(lnP, v)
    lnP_max = np.max(lnP[np.isfinite(lnP)])
    with np.errstate(divide="ignore"):
        lnP_bf = np.log(bf)
    mean_rho = (hist * N).sum(1) / v
    rho_l, I_l = an.tilted_rate_points(lam, lnZ_wham, mean_rho, v)
    # include lambda = 0 in psi (primary: WHAM normalisation)
    lam0 = np.concatenate([lam, [0.0]])
    psi0 = np.concatenate([lnZ_wham, [0.0]]) / v
    err0 = np.concatenate([lnZ_wham_err, [0.0]]) / v
    psi_smc0 = np.concatenate([r["lnZ"], [0.0]]) / v
    flag0 = np.concatenate([smc_flag, [False]])
    o = np.argsort(lam0)
    win_lnP = an.window_estimates(hist, -lam[:, None] * N[None, :], r["lnZ"])
    return dict(lnP=lnP, lnP_bf=lnP_bf, lnP_max=lnP_max, rho=rho, I=I, lnZ_wham=lnZ_wham,
                lnP_tilt=lnP_tilt, tilt_counts=tilt_counts, lnZ_ti=lnZ_ti,
                rho_mean_unbiased=float(r["mean_N"]) / v,
                mean_rho=mean_rho, rho_l=rho_l, I_l=I_l, I_l_err=lnZ_wham_err / v,
                lam=lam0[o], psi=psi0[o], psi_err=err0[o], psi_smc=psi_smc0[o], smc_flag=flag0[o],
                lnZ_wham_err=lnZ_wham_err, smc_flag_t=smc_flag,
                chi=r["var_N"] / v, win_lnP=win_lnP, n_eff=n_eff)


def exact_ideal(r):
    """Exact finite-v quantities for non-interacting particles (binomial)."""
    lnP = an.binomial_lnP(r["N_tot"], r["f"])
    return lnP, (lambda lam: an.binomial_lnZ(r["N_tot"], r["f"], lam) / r["v"])


# --------------------------------------------------------------------------
# Finite-size scaling
# --------------------------------------------------------------------------

def fit_inverse_ell(ells, y, yerr=None, order=1):
    """Weighted least squares y = a0 + a1/ell (+ a2/ell^2); returns a0, err(a0)."""
    ells = np.asarray(ells, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < order + 1:
        return np.nan, np.nan, None
    x = 1.0 / ells[ok]
    A = np.vander(x, order + 1, increasing=True)
    w = np.ones(ok.sum()) if yerr is None else 1.0 / np.maximum(np.asarray(yerr, float)[ok], 1e-6) ** 2
    Aw = A * np.sqrt(w)[:, None]
    yw = y[ok] * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    cov = np.linalg.pinv(Aw.T @ Aw)
    dof = ok.sum() - (order + 1)
    if dof > 0:
        chi2 = np.sum((Aw @ coef - yw) ** 2) / dof
        # with known errors inflate by the reduced chi^2 if > 1; without errors use the residual variance
        cov = cov * (max(chi2, 1.0) if yerr is not None else chi2)
    elif yerr is None:
        cov = cov * np.nan
    return float(coef[0]), float(np.sqrt(cov[0, 0])), coef


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def spread_labels(values, min_gap):
    """Vertical label positions that keep at least min_gap between neighbours."""
    v = np.asarray(values, float)
    order = np.argsort(v)
    pos = v.copy()
    for k in range(1, order.size):
        i, j = order[k - 1], order[k]
        if pos[j] - pos[i] < min_gap:
            pos[j] = pos[i] + min_gap
    return pos


def fig_scgf(runs, res, ext, path, ideal_ref=True):
    cols = size_colors(len(runs))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    for r, q, c in zip(runs, res, cols):
        a.errorbar(q["lam"], q["psi"], yerr=q["psi_err"], color=c, marker="o", ms=4, lw=1.2,
                   label=f"$\\ell={r['ell']:g}$ ($v={r['v']:g}$)")
        fl = q["smc_flag"]
        if np.any(fl):
            a.plot(q["lam"][fl], q["psi_smc"][fl], ls="none", marker="o", ms=6, mfc="none", mec=c, mew=1.0)
    a.plot([], [], ls="none", marker="o", mfc="none", mec=INK2, label="SMC $\\ln\\hat Z/v$ (flagged: degenerate)")
    a.errorbar(ext["lam"], ext["psi_inf"], yerr=ext["psi_inf_err"], color=ORANGE, lw=1.6, marker="D", ms=4.5,
               label=r"$v\to\infty$ (fit in $1/\ell$)")
    if ideal_ref:
        r0 = runs[-1]
        lg = np.linspace(ext["lam"].min(), ext["lam"].max(), 200)
        a.plot(lg, r0["rho_bar"] / r0["f"] * np.log1p(r0["f"] * np.expm1(lg)), color=MUTED, ls="--",
               lw=1.2, label="ideal gas, same $\\bar\\rho$, $f$")
    a.set_xlabel(r"biasing field $\lambda$ (conjugate to $N_v$)")
    a.set_ylabel(r"$\psi_v(\lambda)=\frac{1}{v}\ln\langle e^{\lambda N_v}\rangle$")
    a.set_title("Static SCGF of the subvolume density")
    a.legend(loc="upper left")
    b = ax[1]
    sel = [l for l in (-1.0, -0.5, 0.5, 1.0) if np.any(np.isclose(ext["lam"], l))]
    xs = np.linspace(0, 1.05 / min(r["ell"] for r in runs), 50)
    lc = [BLUE_RAMP[2], BLUE_RAMP[5], BLUE_RAMP[7], BLUE_RAMP[9]]
    for l, c in zip(sel, lc):
        j = int(np.argmin(np.abs(ext["lam"] - l)))
        y = [q["psi"][int(np.argmin(np.abs(q["lam"] - l)))] for q in res]
        e = [q["psi_err"][int(np.argmin(np.abs(q["lam"] - l)))] for q in res]
        b.errorbar([1 / r["ell"] for r in runs], y, yerr=e, color=c, marker="o", ls="none", ms=4.5)
        coef = ext["coef"][j]
        if coef is not None:
            b.plot(xs, np.polyval(coef[::-1], xs), color=c, lw=1.0)
        b.plot([0], [ext["psi_inf"][j]], marker="D", color=ORANGE, ms=5)
    if sel:
        lo_, hi_ = b.get_ylim()
        yv = [ext["psi_inf"][int(np.argmin(np.abs(ext["lam"] - l)))] for l in sel]
        for l, y0, yl in zip(sel, yv, spread_labels(yv, 0.06 * (hi_ - lo_))):
            b.annotate(f"$\\lambda={l:+g}$", (0, y0), xytext=(0.006, yl), textcoords="data",
                       color=INK2, fontsize=8, va="center")
    b.set_xlabel(r"$1/\ell$")
    b.set_ylabel(r"$\psi_v(\lambda)$")
    b.set_title(r"Convergence as $v=\ell^2\to\infty$")
    b.set_xlim(left=-0.005)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_rate(runs, res, ext, path, exact=None):
    cols = size_colors(len(runs))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    for r, q, c in zip(runs, res, cols):
        a.plot(q["rho"], q["I"], color=c, lw=1.4, label=f"$\\ell={r['ell']:g}$")
        a.errorbar(q["rho_l"], q["I_l"], yerr=q["I_l_err"], color=c, marker="o", ms=4, ls="none")
    a.plot(ext["rho_grid"], ext["I_inf"], color=ORANGE, lw=1.6, label=r"$v\to\infty$ (WHAM, fit in $1/\ell$)")
    a.errorbar(ext["rho_l_inf"], ext["I_l_inf"], xerr=ext["rho_l_inf_err"], yerr=ext["I_l_inf_err"],
               color=ORANGE, marker="D", ms=5, ls="none", lw=1.0, label=r"$v\to\infty$ (Legendre pairs)")
    if np.any(np.isfinite(ext["I0_inf"])):
        a.plot(ext["rho_grid"], ext["I0_inf"], color=ORANGE, lw=1.4, ls="--",
               label=r"$v\to\infty$, $f\to0$ (additivity)")
    rb, chi = ext["rho_bar"], ext["chi_inf"]
    a.plot(ext["rho_grid"], (ext["rho_grid"] - rb) ** 2 / (2 * chi), color=MUTED, ls=":", lw=1.3,
           label=r"Gaussian $(\rho-\bar\rho)^2/2\chi$")
    if exact is not None:
        a.plot(exact[0], exact[1], color=INK, ls="--", lw=1.0, label="exact (ideal gas)")
    a.set_xlabel(r"subvolume density $\rho=N_v/v$")
    a.set_ylabel(r"$I_v(\rho)=-\frac{1}{v}\ln P_v(\rho)$ (min. shifted)")
    a.set_title("Rate function: lines = WHAM, points = tilted ensembles")
    top = np.nanmax(ext["I_inf"]) * 1.25 if np.any(np.isfinite(ext["I_inf"])) else None
    a.set_ylim(-0.01, top)
    xmax = max(np.nanmax(q["rho"]) for q in res)
    a.set_xlim(left=-0.02, right=xmax + 0.45)
    a.legend(loc="upper right", ncol=1, fontsize=7.5)
    b = ax[1]
    xs = np.linspace(0, 1.05 / min(r["ell"] for r in runs), 50)
    lc = [BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[7], BLUE_RAMP[9]]
    labs = []
    for k, (rs, c) in enumerate(zip(ext["rho_sel"], lc)):
        y = [np.interp(rs, q["rho"], q["I"], left=np.nan, right=np.nan) for q in res]
        b.plot([1 / r["ell"] for r in runs], y, color=c, marker="o", ls="none", ms=4.5)
        coef = ext["coef_rho"][k]
        if coef is not None:
            b.plot(xs, np.polyval(coef[::-1], xs), color=c, lw=1.0)
            b.plot([0], [coef[0]], marker="D", color=ORANGE, ms=5)
            labs.append((rs, coef[0]))
    if labs:
        lo_, hi_ = b.get_ylim()
        ypos = spread_labels([y for _, y in labs], 0.06 * (hi_ - lo_))
        for (rs, y0), yl in zip(labs, ypos):
            b.annotate(f"$\\rho={rs:.2f}$", (0, y0), xytext=(0.006, yl), textcoords="data",
                       color=INK2, fontsize=8, va="center")
    b.set_xlabel(r"$1/\ell$")
    b.set_ylabel(r"$I_v(\rho)$")
    b.set_title(r"Finite-size extrapolation at fixed $\rho$")
    b.set_xlim(left=-0.005)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_collapse(runs, res, path):
    """The point of the exercise: -ln P_v grows linearly with v; divided by v it collapses."""
    cols = size_colors(len(runs))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for r, q, c in zip(runs, res, cols):
        ok = np.isfinite(q["lnP"])
        N = np.arange(q["lnP"].size)[ok]
        ax[0].plot(N / r["v"], -(q["lnP"][ok] - q["lnP_max"]), color=c, lw=1.4, label=f"$v={r['v']:g}$")
        ax[1].plot(N / r["v"], -(q["lnP"][ok] - q["lnP_max"]) / r["v"], color=c, lw=1.4,
                   label=f"$v={r['v']:g}$")
    ax[0].set_ylabel(r"$-\ln P_v(\rho)+\ln P_v^{\max}$")
    ax[0].set_title(r"Unscaled: tails sharpen with $v$")
    ax[1].set_ylabel(r"$-\frac{1}{v}\,[\ln P_v(\rho)-\ln P_v^{\max}]$")
    ax[1].set_title(r"Scaled by the volume: data collapse $P_v\asymp e^{-vI(\rho)}$")
    for a in ax:
        a.set_xlabel(r"$\rho=N_v/v$")
        a.legend(loc="upper center", ncol=3)
    ymax = max(np.nanmax(-(q["lnP"][np.isfinite(q["lnP"])] - q["lnP_max"])) for q in res)
    ax[0].set_ylim(-0.5, ymax * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def detect_binodal(lnP, v, min_dip=1.0):
    """Two-peak structure of ln P_v(N): returns (rho_gas, rho_liquid, dip) or None.

    Local maxima of a lightly smoothed ln P separated by a minimum at least
    ``min_dip`` below the lower peak signal phase coexistence."""
    ok = np.isfinite(lnP)
    N = np.arange(lnP.size)[ok]
    y = lnP[ok]
    if y.size < 7:
        return None
    k = max(1, int(round(0.02 * v)))
    ys = np.convolve(y, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    ys[:k] = y[:k]
    ys[-k:] = y[-k:]
    peaks = [i for i in range(1, ys.size - 1) if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]]
    best = None
    for a in range(len(peaks)):
        for b in range(a + 1, len(peaks)):
            i, j = peaks[a], peaks[b]
            dip = min(ys[i], ys[j]) - ys[i:j + 1].min()
            if dip >= min_dip and (best is None or dip > best[2]):
                best = (N[i] / v, N[j] / v, float(dip))
    return best


def fig_coexistence(runs, res, path):
    """-ln P_v unscaled, per unit volume (bulk LDP) and per unit length (interfaces)."""
    cols = size_colors(len(runs))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    for r, q, c in zip(runs, res, cols):
        ok = np.isfinite(q["lnP"])
        N = np.arange(q["lnP"].size)[ok]
        y = -(q["lnP"][ok] - q["lnP_max"])
        lab = f"$\\ell={r['ell']:g}$"
        ax[0].plot(N / r["v"], y, color=c, lw=1.4, label=lab)
        ax[1].plot(N / r["v"], y / r["v"], color=c, lw=1.4, label=lab)
        ax[2].plot(N / r["v"], y / r["ell"], color=c, lw=1.4, label=lab)
    bn = detect_binodal(res[-1]["lnP"], runs[-1]["v"])
    if bn is not None:
        for a in ax:
            for x in bn[:2]:
                a.axvline(x, color=MUTED, ls=":", lw=1.0)
        ax[1].annotate(f"peaks at $\\rho\\approx{bn[0]:.2f}$ and ${bn[1]:.2f}$", (0.5, 0.92),
                       xycoords="axes fraction", ha="center", color=INK2, fontsize=8.5)
    ax[0].set_ylabel(r"$-\ln P_v+\ln P_v^{\max}$")
    ax[1].set_ylabel(r"$-\frac{1}{v}[\ln P_v-\ln P_v^{\max}]$  (bulk: $v=\ell^2$)")
    ax[2].set_ylabel(r"$-\frac{1}{\ell}[\ln P_v-\ln P_v^{\max}]$  (interfaces)")
    ax[0].set_title("Unscaled")
    ax[1].set_title(r"Volume scaling: collapses outside coexistence")
    ax[2].set_title(r"Length scaling: collapses inside coexistence")
    for a in ax:
        a.set_xlabel(r"$\rho=N_v/v$")
        a.legend(loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_snapshots(runs, path):
    """One steady-state configuration per subvolume size, subvolume outlined."""
    sel = [r for r in runs if r.get("snapshot_pos") is not None]
    if not sel:
        return
    fig, ax = plt.subplots(1, len(sel), figsize=(3.6 * len(sel), 3.8), squeeze=False)
    for a, r in zip(ax[0], sel):
        pos = r["snapshot_pos"][0]
        g = r["geom"]
        a.scatter(pos[0], pos[1], s=max(0.5, 2600.0 / g["L"] ** 2), color=BLUE_RAMP[6], lw=0)
        lo, hi = g["lo"], g["hi"]
        a.plot([lo, hi, hi, lo, lo], [lo, lo, hi, hi, lo], color=ORANGE, lw=1.6)
        a.set_xlim(0, g["L"])
        a.set_ylim(0, g["L"])
        a.set_aspect("equal")
        a.grid(False)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"$\\ell={r['ell']:g}$, $L={g['L']:g}$, $N={r['N_tot']}$")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_validation(runs, res, path, ideal=None):
    cols = size_colors(len(runs))
    n = 2 if ideal is not None else 1
    fig, ax = plt.subplots(1, n, figsize=(5.4 * n, 3.9), squeeze=False)
    a = ax[0, 0]
    for r, q, c in zip(runs, res, cols):
        ok = np.isfinite(q["lnP_bf"]) & np.isfinite(q["lnP_tilt"])
        cnt = r["hist_tiles"]
        ok &= (cnt >= 20) & (q["tilt_counts"] >= 5.0)   # enough unbiased and tilted samples
        N = np.arange(q["lnP"].size)[ok]
        # brute-force errors from block variation
        blk = r["hist_tiles_blocks"]
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.log(blk / blk.sum(1, keepdims=True))
        e = np.nanstd(np.where(np.isfinite(lb), lb, np.nan), axis=0) / np.sqrt(blk.shape[0])
        a.errorbar(N / r["v"], q["lnP_tilt"][ok] - q["lnP_bf"][ok], yerr=e[ok], color=c, marker="o", ms=3,
                   ls="none", lw=0.8, label=f"$\\ell={r['ell']:g}$")
    a.axhline(0, color=AXIS, lw=1)
    a.set_xlabel(r"$\rho=N_v/v$")
    a.set_ylabel(r"$\ln P_v^{\rm tilted\ SMC}-\ln P_v^{\rm brute\ force}$")
    a.set_title("Tilted ensembles only vs unbiased histograms")
    a.legend(ncol=2)
    if ideal is not None:
        b = ax[0, 1]
        iruns, ires = ideal
        icols = size_colors(len(iruns))
        for r, q, c in zip(iruns, ires, icols):
            lam = np.asarray(r["lambdas"], float)
            ex = an.binomial_lnZ(r["N_tot"], r["f"], lam)
            b.errorbar(lam, q["lnZ_wham"] - ex, yerr=q["lnZ_wham_err"], color=c, marker="o", ms=3.5, lw=1.0,
                       label=f"$\\ell={r['ell']:g}$")
            b.plot(lam + 0.03, r["lnZ"] - ex, ls="none", marker="o", ms=4, mfc="none", mec=c, mew=0.9)
        b.plot([], [], ls="none", marker="o", mfc="none", mec=INK2, label="direct SMC estimate")
        b.axhline(0, color=AXIS, lw=1)
        b.set_xlabel(r"$\lambda$")
        b.set_ylabel(r"$\ln Z_v(\lambda)-\ln Z_v^{\rm exact}(\lambda)$")
        b.set_title("Ideal ABPs vs exact binomial (filled: WHAM)")
        b.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_diagnostics(runs, path):
    cols = size_colors(len(runs))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for r, c in zip(runs, cols):
        lam = np.array([d["lam"] for d in r["diag"]])
        ul = np.unique(lam)
        M = r["smc"]["M"]
        ess = [np.mean([d["ess_min"] for d in r["diag"] if d["lam"] == l]) for l in ul]
        anc = [np.mean([d["n_ancestors"] for d in r["diag"] if d["lam"] == l]) / M for l in ul]
        ax[0].plot(ul, ess, color=c, marker="o", ms=3.5, label=f"$\\ell={r['ell']:g}$")
        ax[1].plot(ul, anc, color=c, marker="o", ms=3.5)
        ax[2].plot(r["acf_t"], r["acf"], color=c, lw=1.3)
    ax[0].set_ylabel("min ESS / M during a run")
    ax[1].set_ylabel(r"distinct $t=0$ ancestors / M")
    for a in ax[:2]:
        a.set_xlabel(r"$\lambda$")
        a.set_ylim(0, 1.05)
    ax[2].set_xlabel("lag t")
    ax[2].set_ylabel(r"$C_{N_v}(t)$ (unbiased)")
    ax[2].axhline(np.exp(-1), color=MUTED, ls=":", lw=1)
    ax[0].legend(ncol=2)
    ax[0].set_title("SMC weight degeneracy")
    ax[1].set_title("Population diversity at the final time")
    ax[2].set_title(r"Subvolume-count autocorrelation ($\tau_v$ at $1/e$)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resdir")
    ap.add_argument("--ideal", help="directory with an ideal-gas run (validation panel)")
    ap.add_argument("--min-ell-fit", type=float, default=0.0, help="smallest ell used in the 1/ell fits")
    ap.add_argument("--fit-order", type=int, default=1)
    a = ap.parse_args()
    style()
    files = sorted(glob.glob(os.path.join(a.resdir, "ell_*.npz")),
                   key=lambda s: float(os.path.basename(s)[4:-4]))
    if not files:
        sys.exit(f"no ell_*.npz in {a.resdir}")
    runs = [load_run(f) for f in files]
    res = [analyze_run(r) for r in runs]
    is_ideal = not runs[0]["params"]["interacting"]

    # ---- extrapolation at fixed lambda ----
    fit_runs = [i for i, r in enumerate(runs) if r["ell"] >= a.min_ell_fit]
    lam_grid = res[0]["lam"]
    psi_inf, psi_inf_err, coefs = [], [], []
    for j, l in enumerate(lam_grid):
        y = [res[i]["psi"][int(np.argmin(np.abs(res[i]["lam"] - l)))] for i in fit_runs]
        e = [max(res[i]["psi_err"][int(np.argmin(np.abs(res[i]["lam"] - l)))], 1e-4) for i in fit_runs]
        a0, e0, coef = fit_inverse_ell([runs[i]["ell"] for i in fit_runs], y, e, order=a.fit_order)
        psi_inf.append(a0)
        psi_inf_err.append(e0)
        coefs.append(coef)
    # ---- Legendre pairs at fixed lambda: rho_v(lambda) = <N_v>_lambda / v, I = lambda rho - psi ----
    lam_t = np.asarray(runs[0]["lambdas"], float)
    rho_l_inf, rho_l_inf_err, I_l_inf, I_l_inf_err = [], [], [], []
    for j, l in enumerate(lam_t):
        yr = [res[i]["mean_rho"][j] for i in fit_runs]
        a0, e0, _ = fit_inverse_ell([runs[i]["ell"] for i in fit_runs], yr, None, order=a.fit_order)
        rho_l_inf.append(a0)
        rho_l_inf_err.append(e0)
        yi = [res[i]["I_l"][j] for i in fit_runs]
        ei = [max(res[i]["I_l_err"][j], 1e-4) for i in fit_runs]
        a1, e1, _ = fit_inverse_ell([runs[i]["ell"] for i in fit_runs], yi, ei, order=a.fit_order)
        I_l_inf.append(a1)
        I_l_inf_err.append(e1)
    # ---- extrapolation at fixed rho (WHAM curves) ----
    lo = max(np.min(q["rho"]) for q in res)
    hi = min(np.max(q["rho"]) for q in res)
    rho_grid = np.round(np.arange(np.ceil(lo * 50) / 50, hi + 1e-9, 0.02), 4)
    I_inf, I_inf_err = [], []
    for rg in rho_grid:
        y = [np.interp(rg, res[i]["rho"], res[i]["I"], left=np.nan, right=np.nan) for i in fit_runs]
        a0, e0, _ = fit_inverse_ell([runs[i]["ell"] for i in fit_runs], y, None, order=a.fit_order)
        I_inf.append(a0)
        I_inf_err.append(e0)
    I_inf = np.array(I_inf)
    # trust the point-wise extrapolation only inside the range spanned by the extrapolated
    # Legendre pairs (where every size is sampled by the tilted ensembles)
    if len(rho_l_inf):
        okr = np.isfinite(rho_l_inf)
        if okr.any():
            lo_l, hi_l = np.nanmin(np.array(rho_l_inf)[okr]), np.nanmax(np.array(rho_l_inf)[okr])
            I_inf = np.where((rho_grid >= lo_l) & (rho_grid <= hi_l), I_inf, np.nan)
    rho_bar = runs[-1]["rho_bar"]
    f = runs[-1]["f"]
    chi_vals = [r["var_N"] / r["v"] for r in runs]
    chi_inf, chi_inf_err, _ = fit_inverse_ell([runs[i]["ell"] for i in fit_runs],
                                              [chi_vals[i] for i in fit_runs], None, order=a.fit_order)
    I0_inf = an.deconvolve_finite_reservoir(rho_grid, I_inf, rho_bar, f) if f < 0.5 else np.full_like(I_inf, np.nan)
    fin = rho_grid[np.isfinite(I_inf)]
    if fin.size:
        lo_s, hi_s = fin.min(), fin.max()
        rho_sel = [rho_bar - 0.8 * (rho_bar - lo_s), rho_bar - 0.4 * (rho_bar - lo_s),
                   rho_bar + 0.4 * (hi_s - rho_bar), rho_bar + 0.8 * (hi_s - rho_bar)]
    else:
        rho_sel = []
    rho_sel = [float(np.round(x, 2)) for x in rho_sel]
    coef_rho = []
    for rs in rho_sel:
        y = [np.interp(rs, res[i]["rho"], res[i]["I"], left=np.nan, right=np.nan) for i in fit_runs]
        coef_rho.append(fit_inverse_ell([runs[i]["ell"] for i in fit_runs], y, None, order=a.fit_order)[2])
    ext = dict(lam=lam_grid, psi_inf=np.array(psi_inf), psi_inf_err=np.array(psi_inf_err), coef=coefs,
               rho_grid=rho_grid, I_inf=I_inf, I_inf_err=np.array(I_inf_err), I0_inf=I0_inf,
               rho_bar=rho_bar, chi_inf=chi_inf, rho_sel=rho_sel, coef_rho=coef_rho,
               rho_l_inf=np.array(rho_l_inf), rho_l_inf_err=np.array(rho_l_inf_err),
               I_l_inf=np.array(I_l_inf), I_l_inf_err=np.array(I_l_inf_err))

    exact = None
    if is_ideal:
        rr = np.linspace(max(rho_grid.min(), 1e-3), rho_grid.max(), 300)
        exact = (rr, an.ideal_rate_function(rr, rho_bar, f))

    ideal = None
    if a.ideal:
        ifiles = sorted(glob.glob(os.path.join(a.ideal, "ell_*.npz")),
                        key=lambda s: float(os.path.basename(s)[4:-4]))
        if ifiles:
            iruns = [load_run(x) for x in ifiles]
            ideal = (iruns, [analyze_run(x) for x in iruns])

    out = a.resdir
    fig_scgf(runs, res, ext, os.path.join(out, "fig_scgf.png"))
    fig_rate(runs, res, ext, os.path.join(out, "fig_rate_function.png"), exact=exact)
    fig_collapse(runs, res, os.path.join(out, "fig_volume_scaling.png"))
    fig_validation(runs, res, os.path.join(out, "fig_validation.png"), ideal=ideal)
    fig_diagnostics(runs, os.path.join(out, "fig_diagnostics.png"))
    fig_coexistence(runs, res, os.path.join(out, "fig_coexistence_scaling.png"))
    fig_snapshots(runs, os.path.join(out, "fig_snapshots.png"))
    binodals = [detect_binodal(q["lnP"], r["v"]) for r, q in zip(runs, res)]

    # ---- summary ----
    summary = dict(
        params=runs[0]["params"], kappa=runs[0]["geom"]["kappa"], f=f, rho_bar=rho_bar,
        sizes=[dict(ell=r["ell"], v=r["v"], N_tot=r["N_tot"], chi_v=r["var_N"] / r["v"],
                    mean_N=float(r["mean_N"]), tau_v=float(r["tau_v"]), v_eff=float(r["v_eff"]),
                    Dt_eff=float(r["Dt_eff"]), horizon=float(r["horizon"]), wall_time_s=float(r["wall_time"]),
                    rho_bar=r["rho_bar"], rho_mean_unbiased=q["rho_mean_unbiased"],
                    lnZ_smc=r["lnZ"].tolist(), lnZ_smc_err=r["lnZ_err"].tolist(),
                    lnZ_wham=q["lnZ_wham"].tolist(), lnZ_wham_err=q["lnZ_wham_err"].tolist(),
                    smc_flag=q["smc_flag_t"].tolist(), lnZ_thermo_integration=q["lnZ_ti"].tolist(),
                    lambdas=q["lam"].tolist(), psi=q["psi"].tolist(), psi_err=q["psi_err"].tolist(),
                    rho_lambda=q["mean_rho"].tolist(), I_lambda=q["I_l"].tolist())
               for r, q in zip(runs, res)],
        two_peak_structure=[None if b is None else dict(ell=r["ell"], rho_low=b[0], rho_high=b[1],
                                                        lnP_dip=b[2]) for r, b in zip(runs, binodals)],
        extrapolated=dict(lambdas=lam_grid.tolist(), psi_inf=ext["psi_inf"].tolist(),
                          psi_inf_err=ext["psi_inf_err"].tolist(), rho=rho_grid.tolist(),
                          I_inf=I_inf.tolist(), I0_inf=np.where(np.isfinite(I0_inf), I0_inf, None).tolist(),
                          chi_inf=chi_inf, chi_inf_err=chi_inf_err, fit_order=a.fit_order,
                          legendre_lambda=lam_t.tolist(), legendre_rho=ext["rho_l_inf"].tolist(),
                          legendre_rho_err=ext["rho_l_inf_err"].tolist(), legendre_I=ext["I_l_inf"].tolist(),
                          legendre_I_err=ext["I_l_inf_err"].tolist(),
                          ells_in_fit=[runs[i]["ell"] for i in fit_runs]),
    )
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    lines = ["# Subvolume-density large deviations: summary", "",
             f"Parameters: {runs[0]['params']}  ", f"kappa = L/ell = {runs[0]['geom']['kappa']:g}, "
             f"f = v/V = {f:.4f}, global density rho_bar = {rho_bar:.4f}", "",
             "| ell | v | N | chi_v = Var(N_v)/v | <N_v>/v (rho_bar) | tau_v | v_eff | Dt_eff | horizon | "
             "max abs(lnZ_TI - lnZ_WHAM) | wall time [s] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, q in zip(runs, res):
        dti = np.max(np.abs(q["lnZ_ti"] - q["lnZ_wham"])) if len(r["lnZ"]) else float("nan")
        lines.append(f"| {r['ell']:g} | {r['v']:g} | {r['N_tot']} | {r['var_N'] / r['v']:.4f} | "
                     f"{q['rho_mean_unbiased']:.4f} ({r['rho_bar']:.4f}) | "
                     f"{float(r['tau_v']):.3f} | {float(r['v_eff']):.2f} | {float(r['Dt_eff']):.2f} | "
                     f"{float(r['horizon']):.2f} | {dti:.2f} | {float(r['wall_time']):.0f} |")
    for r, b in zip(runs, binodals):
        if b is not None:
            lines.append(f"\nell={r['ell']:g}: two-peak P_v (phase coexistence): rho ~ {b[0]:.3f} and "
                         f"{b[1]:.3f}, ln P dip {b[2]:.2f}")
    lines += ["", "lnZ_TI: trapezoidal thermodynamic integration of <N_v>_lambda over the lambda grid "
              "(a coarse-grid consistency check, accurate to O(dlambda^2 Var'))."]
    lines += ["", "SMC ln Z vs WHAM ln Z (flag * = SMC estimate degenerate: inconsistent with WHAM or "
              "violating convexity; WHAM is used):", ""]
    for r, q in zip(runs, res):
        lam_t = np.asarray(r["lambdas"], float)
        ent = [f"{l:+.2f}: {z:.2f}/{w:.2f}{'*' if f_ else ''}"
               for l, z, w, f_ in zip(lam_t, r["lnZ"], q["lnZ_wham"], q["smc_flag_t"])]
        lines.append(f"- ell={r['ell']:g}: " + ", ".join(ent))
    lines += ["", "psi_v(lambda) = (1/v) ln E[exp(lambda N_v)] (WHAM normalisation, per-replicate errors):", "",
              "| lambda | " + " | ".join(f"ell={r['ell']:g}" for r in runs) + " | v -> inf |",
              "|---" * (len(runs) + 2) + "|"]
    for j, l in enumerate(lam_grid):
        row = [f"{q['psi'][int(np.argmin(np.abs(q['lam'] - l)))]:.4f} ± "
               f"{q['psi_err'][int(np.argmin(np.abs(q['lam'] - l)))]:.4f}" for q in res]
        lines.append(f"| {l:+.2f} | " + " | ".join(row) + f" | {ext['psi_inf'][j]:.4f} ± {ext['psi_inf_err'][j]:.4f} |")
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

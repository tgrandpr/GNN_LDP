#!/usr/bin/env python
"""Rate function of the subvolume density inside the MIPS binodal for several sizes.

    python scripts/plot_mips_ldf.py results/mips_pe120_phi08 \
        --tilted results/mips_pe120_phi08_tilted --out results/fig_mips_ldf_three_sizes.png

Left: I_v(rho) = -(1/v)[ln P_v(rho) - max ln P_v]. Right: zoom on the coexistence plateau.
Sizes whose N_v autocorrelation time exceeds half the production time are drawn dashed
(unconverged).
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import analyze_subvolume_ldp as A  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_dir(d):
    files = sorted(glob.glob(os.path.join(d, "ell_*.npz")), key=lambda s: float(os.path.basename(s)[4:-4]))
    runs = [A.load_run(f) for f in files]
    return runs, [A.analyze_run(r) for r in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mips")
    ap.add_argument("--tilted", help="directory with tilted runs (replaces the same ell)")
    ap.add_argument("--out", default="results/fig_mips_ldf_three_sizes.png")
    a = ap.parse_args()
    A.style()
    runs, res = load_dir(a.mips)
    tilted = {}
    if a.tilted:
        tr, tq = load_dir(a.tilted)
        tilted = {r["ell"]: (r, q) for r, q in zip(tr, tq)}
    cols = A.size_colors(len(runs))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    rho_bar = runs[0]["rho_bar"]
    for r, q, c in zip(runs, res, cols):
        src = "unbiased"
        if r["ell"] in tilted:
            r, q = tilted[r["ell"]]
            src = "tilted + unbiased"
        unconv = float(r["tau_v"]) > 0.5 * r["t_prod_used"]
        lab = f"$\\ell={r['ell']:g}$ ($v={r['v']:g}$, {src}" + (", unconverged)" if unconv else ")")
        for x in ax:
            x.plot(q["rho"], q["I"], color=c, lw=1.8, ls="--" if unconv else "-", label=lab)
    for x in ax:
        x.axvline(rho_bar, color=A.MUTED, ls=":", lw=1.0)
        x.set_xlabel(r"subvolume density $\rho=N_v/v$")
    ax[0].set_ylabel(r"$I_v(\rho)=-\frac{1}{v}\,[\ln P_v(\rho)-\ln P_v^{\max}]$")
    ax[0].set_ylim(-0.005, 0.3)
    ax[0].set_title("LDF of the subvolume density inside the MIPS binodal")
    ax[0].annotate(r"global $\bar\rho$", (rho_bar, 0.12), xytext=(4, 0), textcoords="offset points",
                   color=A.INK2, fontsize=8.5)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].set_ylim(-0.001, 0.03)
    ax[1].set_xlim(0.15, 1.6)
    ax[1].set_ylabel(r"$I_v(\rho)$ (zoom)")
    ax[1].set_title("Zoom on the coexistence plateau")
    p = runs[0]["params"]
    fig.suptitle(f"Pe $={p['v0']:g}$, $\\phi={np.pi * p['rho0'] / 4:.2f}$, "
                 f"$\\varepsilon={p['eps']:g}$, $L=3\\ell$", fontsize=10, color=A.INK2, y=0.995)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Side-by-side volume-scaled rate functions: homogeneous state vs inside the MIPS binodal.

    python scripts/plot_two_regimes.py results/abp_pe24 results/mips_pe120_phi08 \
        --tilted results/mips_pe120_phi08_tilted --out results/fig_two_regimes.png
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
    ap.add_argument("homogeneous")
    ap.add_argument("mips")
    ap.add_argument("--tilted", help="MIPS directory with tilted runs (replaces the same ell)")
    ap.add_argument("--out", default="results/fig_two_regimes.png")
    a = ap.parse_args()
    A.style()
    hr, hq = load_dir(a.homogeneous)
    mr, mq = load_dir(a.mips)
    tilted = {}
    if a.tilted:
        tr, tq = load_dir(a.tilted)
        tilted = {r["ell"]: (r, q) for r, q in zip(tr, tq)}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    cols = A.size_colors(len(hr))
    for r, q, c in zip(hr, hq, cols):
        ax[0].plot(q["rho"], q["I"], color=c, lw=1.5, label=f"$\\ell={r['ell']:g}$")
    p = hr[0]["params"]
    ax[0].set_title(f"Homogeneous: Pe$={p['v0']:g}$, $\\phi={np.pi * p['rho0'] / 4:.2f}$")
    ax[0].set_ylim(-0.005, 0.3)
    mcols = A.size_colors(len(mr))
    for r, q, c in zip(mr, mq, mcols):
        rr, qq, lab = r, q, f"$\\ell={r['ell']:g}$ (unbiased)"
        if r["ell"] in tilted:
            rr, qq = tilted[r["ell"]]
            lab = f"$\\ell={r['ell']:g}$ (tilted + unbiased)"
        unconverged = float(rr["tau_v"]) > 0.5 * rr["t_prod_used"]
        ax[1].plot(qq["rho"], qq["I"], color=c, lw=1.5, ls="--" if unconverged else "-",
                   label=lab + (", unconverged" if unconverged else ""))
    p = mr[0]["params"]
    ax[1].set_title(f"Inside the MIPS binodal: Pe$={p['v0']:g}$, $\\phi={np.pi * p['rho0'] / 4:.2f}$")
    ax[1].set_ylim(-0.005, 0.3)
    for x in ax:
        x.set_xlabel(r"subvolume density $\rho=N_v/v$")
        x.set_ylabel(r"$I_v(\rho)=-\frac{1}{v}[\ln P_v(\rho)-\ln P_v^{\max}]$")
        x.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

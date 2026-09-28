#!/usr/bin/env python3
"""Combined cortical-depth figure. a: DBSI anisotropic-fiber fraction from cortex through the
white surface into superficial white matter, 3HG vs matched crown; b: the 3HG-crown difference
across that depth (fiber converges sub-WM). c-h: equivolumetric intracortical depth profiles of
six microstructure metrics (white d=0 -> pial d=1), region-matched, FDR-marked. Specimen is the
unit; medial-wall 3HG excluded. Intracortical profiling only -- no histological-layer claim."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import natstyle as ns
import fig_fiberconv as fc                                   # reuse sub-WM profile sampling
ns.use()

DEPTH = (_env.ROOT + "/surf/three_hinge/depth")
RES = (_env.ROOT + "/surf/three_hinge/results")
SUBJ = [1, 2, 3, 4, 5, 6]
ORDER = [("fiber", "DBSI fiber"), ("MD", "MD"), ("hindered", "DBSI hindered"),
         ("water", "DBSI free water"), ("restricted", "DBSI restricted"), ("FA", "FA")]
INNER, MID, OUTER = slice(0, 3), slice(3, 6), slice(6, 9)


def load_depth():
    rho = np.load(f"{DEPTH}/Subject5_depth.npz")["rho"]
    Yh, Yc, dWP = {}, {}, {}
    for m, _ in ORDER:
        Yh[m] = np.array([np.load(f"{DEPTH}/Subject{s}_depth.npz")[f"Yh_{m}"] for s in SUBJ])
        Yc[m] = np.array([np.load(f"{DEPTH}/Subject{s}_depth.npz")[f"Yc_{m}"] for s in SUBJ])
        dWP[m] = np.array([np.load(f"{DEPTH}/Subject{s}_depth.npz")[f"dWP_{m}"] for s in SUBJ])
    return rho, Yh, Yc, dWP


def main():
    # ---- sub-WM fiber profiles (panels a,b); cached so layout tweaks don't resample ----
    cache = f"{RES}/_depthAB_cache.npz"
    if os.path.exists(cache):
        z = np.load(cache); Hs, Cs = z["Hs"], z["Cs"]
    else:
        Hs, Cs = [], []
        for s in SUBJ:
            hm, cm, _, _ = fc.profiles(s)
            Hs.append(hm); Cs.append(cm)
        Hs, Cs = np.array(Hs), np.array(Cs)
        np.savez(cache, Hs=Hs, Cs=Cs)
    dH = Hs[:, fc.SUBWM].mean(1); dC = Cs[:, fc.SUBWM].mean(1)
    pAB = stats.wilcoxon(dH - dC).pvalue
    D = fc.DEPTHS

    rho, Yh, Yc, dWP = load_depth()

    fig = plt.figure(figsize=(11.4, 9.2), facecolor="white")
    gsT = gridspec.GridSpec(1, 2, left=0.085, right=0.985, top=0.955, bottom=0.71, wspace=0.24)
    gsG = gridspec.GridSpec(2, 3, left=0.075, right=0.985, top=0.60, bottom=0.06, wspace=0.30, hspace=0.36)

    # panel a: fiber profile from cortex, across the white surface, into superficial WM
    axA = fig.add_subplot(gsT[0])
    axA.axvspan(0, D.max(), color="#f0eeea", zorder=0)
    axA.axvline(0, color=ns.INK, lw=0.8)
    for M, col, lab in [(Cs, ns.C_CROWN, "gyral crown"), (Hs, ns.C_3HG, "3HG")]:
        mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(6)
        axA.plot(D, mu, color=col, lw=2.4, label=lab, solid_capstyle="round", zorder=3)
        axA.fill_between(D, mu - se, mu + se, color=col, alpha=.18, lw=0, zorder=2)
    _y0, _y1 = axA.get_ylim(); axA.set_ylim(_y0, _y1 + 0.17 * (_y1 - _y0))   # headroom for labels
    axA.set_xlabel("distance from white surface  (mm)")
    axA.set_ylabel("DBSI fiber fraction")
    axA.text(0.14, 0.96, "gray matter", transform=axA.transAxes, ha="center", va="top",
             fontsize=9.5, color=ns.INK)
    axA.text(0.64, 0.96, "white matter", transform=axA.transAxes, ha="center", va="top",
             fontsize=9.5, color=ns.INK)
    axA.legend(loc="lower right", handlelength=1.2)
    ns.panel(axA, "a", x=-0.13, y=1.05)

    # panel b: 3HG - crown difference across the same cortex->WM distance
    axB = fig.add_subplot(gsT[1])
    diff = Hs - Cs; mu = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(6)
    axB.axvspan(0, D.max(), color="#f0eeea", zorder=0)
    axB.axhline(0, color=ns.INK, lw=0.8, zorder=1)
    axB.fill_between(D, 0, mu, where=(mu > 0), color=ns.C_3HG, alpha=.14, zorder=1)
    axB.fill_between(D, mu - se, mu + se, color=ns.C_3HG, alpha=.22, lw=0, zorder=2)
    axB.plot(D, mu, color=ns.C_3HG, lw=2.4, solid_capstyle="round", zorder=3)
    axB.axvline(0, color=ns.INK, lw=0.8, zorder=1)
    _y0, _y1 = axB.get_ylim(); axB.set_ylim(_y0, _y1 + 0.17 * (_y1 - _y0))   # headroom for labels
    axB.set_xlabel("distance from white surface  (mm)")
    axB.set_ylabel("fiber-fraction difference")
    axB.text(0.14, 0.96, "gray matter", transform=axB.transAxes, ha="center", va="top",
             fontsize=9.5, color=ns.INK)
    axB.text(0.64, 0.96, "white matter", transform=axB.transAxes, ha="center", va="top",
             fontsize=9.5, color=ns.INK)
    axB.text(0.97, 0.10, f"P = {pAB:.3f}", transform=axB.transAxes, ha="right", va="bottom",
             fontsize=9, color=ns.INK)
    ns.panel(axB, "b", x=-0.15, y=1.05)
    for a in (axA, axB):
        for sp in ("top", "right"): a.spines[sp].set_visible(False)

    # panels c-h: equivolumetric intracortical depth profiles (white surface -> pial surface)
    print(f"{'metric':14s} {'inner':>7s} {'mid':>7s} {'outer':>7s} {'slope p':>8s}")
    for i, (m, lab) in enumerate(ORDER):
        ax = fig.add_subplot(gsG[i // 3, i % 3])
        for Y, col, name in [(Yc[m], ns.C_CROWN, "crown"), (Yh[m], ns.C_3HG, "3HG")]:
            mu = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(6)
            ax.plot(rho, mu, color=col, lw=2.2, label=name, solid_capstyle="round", zorder=3)
            ax.fill_between(rho, mu - se, mu + se, color=col, alpha=.16, lw=0, zorder=2)
        d = dWP[m]; pv = [stats.wilcoxon(d[:, k]).pvalue for k in range(d.shape[1])]
        q = multipletests(pv, method="fdr_bh")[1]
        ytop = ax.get_ylim()[1]
        for k in range(len(rho)):
            if q[k] < 0.05:
                ax.plot(rho[k], ytop, marker="v", color=ns.INK, ms=4, clip_on=False, zorder=5)
        ax.set_title(lab, fontsize=11.5, pad=3)
        ns.panel(ax, "cdefgh"[i], x=-0.2, y=1.06)
        if m == "MD":                                    # avoid the 0.00008-style tick clutter
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.yaxis.get_offset_text().set(size=8.5, color=ns.INK)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["white\n(0)", "0.5", "pial\n(1)"], fontsize=9)
        if i >= 3: ax.set_xlabel("normalized cortical depth")
        if i == 0: ax.legend(loc="best", handlelength=1.0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        inner, mid, outer = d[:, INNER].mean(), d[:, MID].mean(), d[:, OUTER].mean()
        slopes = np.array([np.polyfit(rho, d[j], 1)[0] for j in range(6)])
        print(f"{lab:14s} {inner:+7.2f} {mid:+7.2f} {outer:+7.2f} {stats.wilcoxon(slopes).pvalue:8.3f}")

    fig.savefig(f"{RES}/FIG2_depth.png", dpi=300); fig.savefig(f"{RES}/FIG2_depth.pdf")
    from PIL import Image
    print("wrote FIG2_depth.png", Image.open(f"{RES}/FIG2_depth.png").size,
          "| sub-WM p=%.3f" % pAB)


if __name__ == "__main__":
    main()

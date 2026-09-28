#!/usr/bin/env python3
"""Supplementary: matching and confound diagnostics for the folding-convergence ladder.

 a  host-sulc percentile distributions of the three rungs, before and after the decile
    matching (vertices pooled over the six specimens; the matched distributions coincide).
 b  |mean curvature| after matching - deliberately not matched, and still separated:
    junction-ness IS local curvature, which is why it must stay out of the covariates.
 c  raw skeleton-vertex counts per degree and specimen: the skeleton stops at degree 4,
    which is why the ladder's top rung is 4HG+ and no finer order is separable.
 d  the slope the claim rests on: per-specimen OLS beta of the landmark z value on skeleton
    degree (2-4), with host-sulc percentile and DK parcel always in the model, with and
    without |curvature| - the slope does not move.
Data: matching_diag.py -> results/matching_diag.npz."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import natstyle as ns, suppstyle as ss
ns.nature()
RES = (_env.ROOT + "/surf/three_hinge/results")
DPI = 1600
RUNGS = ["crest", "3HG", "4HG+"]
RCOL = {"crest": ss.BLUE, "3HG": ss.ORANGE, "4HG+": ss.GREEN}
METS = ["MD", "AD", "RD", "FA", "restricted", "fiber", "hindered", "water"]
MLAB = {"restricted": "restr.", "hindered": "hind.", "water": "water", "fiber": "fiber",
        "MD": "MD", "AD": "AD", "RD": "RD", "FA": "FA"}


def kpanel(ax, z, keys, xs, xlabel):
    for r in RUNGS:
        v = z[f"{keys}|{r}"]; v = v[np.isfinite(v)]
        kd = stats.gaussian_kde(v)
        d = kd(xs)
        ax.fill_between(xs, 0, d, color=RCOL[r], alpha=.45, lw=0, zorder=2)
        ax.plot(xs, d, color=RCOL[r], lw=1.2, zorder=3, label=r)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, None)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=7, labelpad=2)
    ax.set_ylabel("Density", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ss.frame(ax)


def main():
    z = np.load(f"{RES}/matching_diag.npz")
    fig = plt.figure(figsize=(ns.W2, 96 * ns.MM), facecolor="white")
    B1, H1 = 0.615, 0.315
    B2, H2 = 0.130, 0.315

    axA1 = fig.add_axes([0.055, B1, 0.240, H1])
    kpanel(axA1, z, "raw", np.linspace(0, 100, 200), "Fold depth (sulc percentile)")
    axA1.text(0.97, 0.92, "before matching", transform=axA1.transAxes, ha="right",
              fontsize=6.2)
    axA1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False, fontsize=5.8,
                handlelength=1.0, labelspacing=.3)
    axA2 = fig.add_axes([0.365, B1, 0.240, H1])
    kpanel(axA2, z, "mat", np.linspace(0, 100, 200), "Fold depth (sulc percentile)")
    axA2.text(0.97, 0.92, "after matching", transform=axA2.transAxes, ha="right", fontsize=6.2)

    axB = fig.add_axes([0.720, B1, 0.262, H1])
    kpanel(axB, z, "cur", np.linspace(0, 0.95, 220), "|mean curvature| (mm$^{-1}$)")

    # ---- c: counts per degree ----
    axC = fig.add_axes([0.055, B2, 0.240, H2])
    C = z["counts"]; degs = z["degrees"]
    keep = C.sum(0) > 0
    for k, d in enumerate(degs[keep]):
        v = C[:, keep][:, k]
        axC.bar(k, np.mean(v), 0.6, facecolor=ss.BLUE, edgecolor="black", linewidth=0.8,
                zorder=2)
        axC.scatter(np.full(6, k) + np.linspace(-.12, .12, 6), v, s=6, color="black",
                    zorder=4, lw=0)
    axC.set_yscale("log")
    axC.set_xticks(range(int(keep.sum())))
    axC.set_xticklabels([str(d) for d in degs[keep]], fontsize=6.4)
    axC.set_ylim(10, 5e4)
    axC.set_xlabel("Skeleton degree", fontsize=7, labelpad=2)
    axC.set_ylabel("Vertices per hemisphere", fontsize=7, labelpad=2)
    axC.tick_params(labelsize=6); axC.minorticks_off()
    ss.frame(axC)

    # ---- d: beta_degree, two models ----
    axD = fig.add_axes([0.385, B2, 0.597, H2])
    axD.axhline(0, color="black", lw=.7, ls=(0, (4, 2.5)), zorder=1)
    for i, m in enumerate(METS):
        for k, (tag, col, off) in enumerate((("base", ss.BLUE, -0.16),
                                             ("curv", ss.ORANGE, 0.16))):
            b = z[f"beta|{m}|{tag}"]
            mu, se = b.mean(), b.std(ddof=1) / np.sqrt(len(b))
            axD.errorbar(i + off, mu, yerr=se, fmt="o", ms=4.2, mfc=col, mec="black",
                         mew=0.6, ecolor="black", elinewidth=0.9, capsize=2.0, zorder=4)
    axD.set_xticks(range(len(METS)))
    axD.set_xticklabels([MLAB[m] for m in METS], fontsize=6.4)
    axD.set_xlim(-0.6, len(METS) - 0.4)
    axD.set_ylim(-0.135, 0.150)
    axD.set_ylabel(r"$\beta_{\rm degree}$  (SD per order)", fontsize=7, labelpad=2)
    axD.tick_params(labelsize=6)
    ss.frame(axD)
    ss.legend_swatch(axD, [(ss.BLUE, "degree + sulc + region"),
                           (ss.ORANGE, "degree + sulc + region + |curvature|")],
                     loc="upper left", bbox_to_anchor=(0.015, 0.98))

    for letter, x, y in (("a", 0.012, B1 + H1 + 0.035), ("b", 0.675, B1 + H1 + 0.035),
                         ("c", 0.012, B2 + H2 + 0.035), ("d", 0.340, B2 + H2 + 0.035)):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="bottom")

    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{RES}/FIGS_matching.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image
    w, h = Image.open(f"{RES}/FIGS_matching.png").size
    print(f"wrote FIGS_matching.png  {w/DPI*25.4:.0f} x {h/DPI*25.4:.0f} mm")


if __name__ == "__main__":
    main()

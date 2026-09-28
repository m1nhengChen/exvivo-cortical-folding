#!/usr/bin/env python3
"""Supplementary: why the published gyral-bias correction is not applied here.

The correction in common use rescales node STRENGTH by an assumed linear function of `sulc`, with
the crown-to-wall ratio taken from histology in another dataset. Four panels, four reasons that is
the wrong instrument for this analysis - and one honest caveat, which is panel d.

 a  gyral bias exists in this data and is measured, not assumed: streamline endpoint density on the
    white surface against fold depth, with the two node classes' own depth distributions beneath it
    on the same axis, so "where we sit on the bias curve" is one reading rather than two.
 b  how far that difference propagates: endpoint density -> streamlines collected per node -> node
    degree. It attenuates by an order of magnitude.
 c  what each estimator does when the sampling difference is controlled. Pair incidence loses its
    effect; the seed-normalized contrast cannot, because f divides by the seed's own streamline
    count by construction.

b and c share an x axis so the 1.50 -> 1.06 attenuation and the estimator spread can be compared by
eye instead of by reading numbers off two different scales.
Data: gyral_bias.py -> results/gyral_bias.npz."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import natstyle as ns
ns.nature()
# One raster resolution for every supplementary figure. These were set per-figure as each
# was written and drifted to 800 / 1200 / 1600 dpi, which shows as three different line
# weights when they are printed side by side.
DPI = 1600
RES = (_env.ROOT + "/surf/three_hinge/results")
VAL = (_env.ROOT + "/surf/three_hinge/validate")
FW, FH = ns.W2, 74 * ns.MM


def main():
    import suppstyle as ss
    z = np.load(f"{RES}/gyral_bias.npz")
    d, e = z["dens"], z["edges"]
    x = (e[:-1] + e[1:]) / 2
    sh, sc = z["sulc_h"], z["sulc_c"]
    mh, mc = np.median(sh), np.median(sc)
    ors, nsd, deg = z["ors"], z["nseed"], z["degree"]
    mu = np.nanmean(d, 0); se = np.nanstd(d, 0, ddof=1) / np.sqrt(len(d))
    fd = np.interp([mh, mc], x, mu)

    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    TOP, BOT = 0.905, 0.185

    # ---------------------------------------------------------------- a
    HA = (TOP - BOT) * 0.70
    axA = fig.add_axes([0.070, TOP - HA, 0.255, HA])
    axM = fig.add_axes([0.070, BOT, 0.255, (TOP - BOT) - HA - 0.055])
    axA.fill_between(x, mu - se, mu + se, color="black", alpha=.16, lw=0, zorder=2)
    axA.plot(x, mu, color="black", lw=1.3, solid_capstyle="round", zorder=3)
    for m, col in ((mh, ss.ORANGE), (mc, ss.BLUE)):
        y = float(np.interp(m, x, mu))
        axA.plot([m], [y], marker="o", ms=4.6, mfc=col, mec="black", mew=.6, zorder=6,
                 linestyle="none")
        axA.plot([m, m], [0, y], color=col, lw=.7, ls=(0, (2, 1.8)), zorder=4)
    axA.annotate("", xy=(mc + 1.0, fd[0]), xytext=(mc + 1.0, fd[1]),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=.7, shrinkA=0, shrinkB=0))
    axA.plot([mh, mc + 1.0], [fd[0], fd[0]], color="black", lw=.5, zorder=3)
    axA.text(mc + 3.5, (fd[0] + fd[1]) / 2, f"{fd[0]/fd[1]:.2f}×", fontsize=7,
             fontweight="bold", va="center", ha="left")
    axA.set_xlim(0, 100); axA.set_ylim(0, mu.max() * 1.14)
    axA.set_xticklabels([])
    axA.set_ylabel("Endpoint density (mm$^{-2}$)", fontsize=7.5, labelpad=2)
    axA.set_yticks([0, 4, 8, 12]); axA.tick_params(labelsize=6)
    ss.frame(axA)
    ns.ptitle(axA, "", "a", lx=-0.235, y=1.02)

    bins = np.linspace(0, 100, 101)
    for v, col in ((sc, ss.BLUE), (sh, ss.ORANGE)):
        h, _ = np.histogram(v, bins=bins, density=True)
        axM.fill_between(bins[:-1], 0, h, step="post", color=col, alpha=.80, lw=0, zorder=2)
    axM.set_xlim(0, 100); axM.set_yticks([])
    axM.set_xticks([0, 25, 50, 75, 100]); axM.tick_params(labelsize=6)
    axM.set_xlabel("Fold depth (sulc percentile)", fontsize=7.5, labelpad=2)
    axM.set_ylim(0, axM.get_ylim()[1] * 1.30)
    axM.text(2.0, axM.get_ylim()[1] * .93, "3HG", color=ss.ORANGE, fontsize=6.4,
             fontweight="bold", ha="left", va="top")
    axM.text(mc + 6.5, axM.get_ylim()[1] * .93, "Crown", color=ss.BLUE, fontsize=6.4,
             fontweight="bold", ha="left", va="top")
    ss.frame(axM)

    # ---------------------------------------------------------------- b, c: vertical bars,
    # black-edged fills, specimen dots, ratio + stars above each bar
    YR = (0.92, 1.75)

    def vbars(ax, items, letter, title, xlabel_note=None, lx=-0.24):
        ax.axhline(1, color="black", lw=.7, ls=(0, (4, 2.5)), zorder=1)
        for i, (lab, v, col) in enumerate(items):
            v = np.atleast_1d(np.asarray(v, float))
            m = float(np.exp(np.mean(np.log(v))))
            ax.bar(i, m - 1.0, 0.58, bottom=1.0, facecolor=col, edgecolor="black",
                   linewidth=0.8, zorder=2)
            txt = f"{m:.2f}×"
            if len(v) > 1:
                ax.scatter(np.full(len(v), i) + np.linspace(-.13, .13, len(v)), v, s=6,
                           color="black", zorder=4, lw=0)
                sd = m * np.std(np.log(v), ddof=1) / np.sqrt(len(v))
                ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="black", elinewidth=.9,
                            capsize=2.0, zorder=5)
                p = stats.ttest_1samp(np.log(v), 0).pvalue
                txt += ss.stars(p)
            ax.text(i, YR[1] - 0.015, txt, ha="center", va="top", fontsize=6.2,
                    fontweight="bold")
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([t[0] for t in items], fontsize=6.2, rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.65, len(items) - 0.35)
        ax.set_ylim(*YR); ax.set_yticks([1.0, 1.25, 1.5])
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylabel(xlabel_note, fontsize=7.5, labelpad=2)
        ss.frame(ax)
        ns.ptitle(ax, "", letter, lx=lx, y=1.02)

    axB = fig.add_axes([0.435, BOT, 0.235, TOP - BOT])
    vbars(axB, [("Endpoint density", fd[0] / fd[1], ss.ORANGE),
                ("Streamlines per node", nsd[:, 0] / nsd[:, 1], ss.ORANGE),
                ("Node degree", deg[:, 0] / deg[:, 1], ss.ORANGE)],
          "b", "How far the difference propagates", "3HG / crown", lx=-0.17)

    sc_ = np.load(f"{VAL}/seed_control.npz")
    seed = sc_["f_hh"] / sc_["f_ch"]
    axC = fig.add_axes([0.775, BOT, 0.215, TOP - BOT])
    vbars(axC, [("Pair incidence", np.exp(ors[:, 0]), ss.BLUE),
                ("+ seed count", np.exp(ors[:, 2]), ss.GRAY),
                ("Seed contrast", seed, ss.BLUE)],
          "c", "What each estimator sees", "Enrichment", lx=-0.20)

    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{RES}/FIGS_gyralbias.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image
    w, h = Image.open(f"{RES}/FIGS_gyralbias.png").size
    print(f"wrote FIGS_gyralbias.png  {w/DPI*25.4:.0f} x {h/DPI*25.4:.0f} mm, {w} x {h} px")
    print(f"  endpoint density over the class band: {fd[0]/fd[1]:.2f}x")
    print(f"  streamlines per node: {np.exp(np.mean(np.log(nsd[:,0]/nsd[:,1]))):.3f}x")
    print(f"  node degree:          {np.exp(np.mean(np.log(deg[:,0]/deg[:,1]))):.3f}x")


if __name__ == "__main__":
    main()

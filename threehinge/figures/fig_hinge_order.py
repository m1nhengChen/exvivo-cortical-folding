#!/usr/bin/env python3
"""Panel: cortical microstructure tracks folding convergence.

x is the folding-convergence ladder read off the gyral skeleton: a crest point with no junction,
a three-way junction (3HG), a four-way-or-higher junction. The first rung pools off-skeleton gyral
crown with degree-2 skeleton points - a crown vertex sits a median 1.0 mm from the skeleton and the
2 mm measuring disks overlap, so they are one tissue, not two rungs. All rungs are resampled onto a
common fold-depth (sulc) histogram before anything is measured, because the raw rungs sit at
different depths. Lines are the group mean of the within-specimen z, shading the s.e.m. over six specimens.
Metrics that fall with convergence are drawn in the crown colour, those that rise in the 3HG
colour, matching the class colours used elsewhere in the figure.
Data: hinge_order.py -> results/hinge_order.npz."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import natstyle as ns

RES = (_env.ROOT + "/surf/three_hinge/results")
RUNGS = ["crest", "3HG", "4HG+"]        # keys into hinge_order.npz
RUNG_LABEL = ["Crest", "3HG", "4HG+"]   # display only; the npz keys stay lower case
# (key in the npz, short label, colour) - falling metrics in the crown blue, rising in 3HG orange
LINES = [("ADC", "MD", ns.BLUE),
         ("hindered_ratio", "Hind.", "#6E8DB6"),
         ("water_ratio", "Water", ns.BLUE_L),
         ("fiber_ratio", "Fiber", ns.C_3HG),
         ("restricted_ratio", "Restr.", "#E5A084")]


def order_panel(ax, letter, lx=-0.19, ly=1.10):
    z = np.load(f"{RES}/hinge_order.npz")
    x = np.arange(len(RUNGS))
    ax.axhline(0, color=ns.MUTE, lw=.5, ls=(0, (2.5, 2)), zorder=1)
    ends = []
    for key, lab, col in LINES:
        M = np.array([z[f"{key}|{r}"] for r in RUNGS])          # (rung, specimen)
        mu = M.mean(1); se = M.std(1, ddof=1) / np.sqrt(M.shape[1])
        ax.fill_between(x, mu - se, mu + se, color=col, alpha=.15, lw=0, zorder=2)
        ax.plot(x, mu, color=col, lw=1.1, marker="o", ms=2.4, mec="white", mew=.4,
                solid_capstyle="round", zorder=3)
        ends.append([mu[-1], lab, col])
    # the falling metrics converge at 4HG+, so their end labels would sit on top of one another;
    # push them apart by a minimum gap while keeping their order and a leader to the true value
    GAP = 0.062
    ends.sort(key=lambda e: e[0])
    ypos = [e[0] for e in ends]
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + GAP)
    shift = max(0.0, (ypos[-1] - 0.33))                          # keep the stack inside the axes
    for (yv, lab, col), yl in zip(ends, ypos):
        yl -= shift
        if abs(yl - yv) > 1e-3:
            ax.plot([len(RUNGS) - .98, len(RUNGS) - .90], [yv, yl], color=col, lw=.45, zorder=3, clip_on=False)
        ax.text(len(RUNGS) - .87, yl, lab, color=col, fontsize=5.8, fontweight="bold",
                va="center", ha="left", clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(RUNG_LABEL, fontsize=6)
    ax.set_xlim(-.22, len(RUNGS) - .90); ax.set_ylim(-.36, .36); ax.set_yticks([-.3, 0, .3])
    ax.set_xlabel("Folding convergence", fontsize=7, labelpad=2)
    ax.set_ylabel("Microstructure (SD)", fontsize=7, labelpad=3)
    ns.nletter(ax, letter, x=lx, y=ly)


def main():
    ns.nature()
    fig = plt.figure(figsize=(ns.W1 * 0.8, 46 * ns.MM), facecolor="white")
    order_panel(fig.add_axes([0.20, 0.20, 0.62, 0.68]), "a", lx=-0.24, ly=1.06)
    fig.savefig(f"{RES}/FIG_hinge_order.png", dpi=800, bbox_inches="tight", pad_inches=0.02)
    print("wrote FIG_hinge_order.png")


if __name__ == "__main__":
    main()

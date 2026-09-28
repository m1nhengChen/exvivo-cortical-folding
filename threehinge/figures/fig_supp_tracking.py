#!/usr/bin/env python3
"""Supplementary: does the inter-hinge excess depend on how the tractogram was built?

Three questions, each holding as much fixed as it can.

 a  Within the published reconstruction, does the excess come from committing each seed to ONE
    discrete peak? max_cross sets how many initial directions are tracked from a seed in a crossing
    voxel - and three-hinge gyri are where the crossings are. Opening it from 1 to all raises the
    streamline count by 53% and does not lower the enrichment.
 b  Does it come from the propagation rule, or from the reconstruction as a whole? The first three
    rows share one fODF, one set of seeds, one stopping criterion, one angle and one step size, and
    differ only in how a direction is chosen at each step. The fourth is separate software with its
    own response function, FOD and stopping rule.
 c  What differs between the tractograms: where streamlines stop, as signed distance to the white
    surface. The nodes sit 2 mm inside it (dashed).

MRtrix iFOD2 was also run and is deliberately NOT in b. The estimator scores a connection from a
streamline's distal ENDPOINT, and without anatomical constraints iFOD2's FOD-amplitude stopping
rule does not fire at the grey/white boundary: its endpoints peak 2.75 mm outside the white surface
and only 44% of them fall inside it, against 66% and 63% for the two tractograms that are compared.
Its curve is kept in c so the exclusion is visible and checkable rather than silent. Note the order
of discovery honestly in the text: the incompatibility was identified after iFOD2 disagreed, and
the criterion has to be stated as a property of the estimator for the exclusion to be defensible."""
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
FW, FH = ns.W2, 66 * ns.MM
# The x range runs well past the largest specimen value (1.52) on purpose: the right-hand strip is
# a reserved column for the summary text, which otherwise sits on top of the scatter.
# Data range only. The numeric column lives OUTSIDE the axes (get_yaxis_transform, x > 1), which
# is how a forest plot is normally set: an in-axes column has to be padded past the widest dot and
# leaves a band of dead space between the data and the text.
XR = (0.92, 1.60)
ROWH = 0.62                                       # bar height, as a fraction of the row pitch


def vgroup(ax, items, letter, title, ylabel, ymax=1.86, lx=-0.20):
    """vertical violin+box per condition, star vs unity above each - example-figure style"""
    import suppstyle as ss
    ax.axhline(1, color="black", lw=.7, ls=(0, (4, 2.5)), zorder=1)
    for i, (lab, v, col) in enumerate(items):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        ss.violin_box(ax, i, v, col, width=0.62, tf=np.log, itf=np.exp)
        p = stats.ttest_1samp(np.log(v), 0).pvalue
        m = float(np.exp(np.mean(np.log(v))))
        ax.text(i, ymax * 0.995, ss.stars(p), ha="center", va="top", fontsize=7.5)
        ax.text(i, ymax * 0.945, f"{m:.2f}×", ha="center", va="top", fontsize=5.6)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([t[0] for t in items], fontsize=6.2, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.75, len(items) - 0.25)
    ax.set_ylim(0.84, ymax)
    ax.set_yticks([1.0, 1.25, 1.5]); ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel(ylabel, fontsize=7.5, labelpad=2)
    ss.frame(ax)
    ns.ptitle(ax, "", letter, lx=lx, y=1.02)


def main():
    import suppstyle as ss
    z = np.load(f"{VAL}/ablate_tracking.npz")
    A, B, AN = z["A"], z["B"], z["AN"]
    det = np.load(f"{VAL}/mrtrix_det.npz")
    ep = np.load(f"{VAL}/endpoint_depth.npz")

    fig = plt.figure(figsize=(FW, 70 * ns.MM), facecolor="white")
    BOT, TOP = 0.240, 0.880

    axA = fig.add_axes([0.062, BOT, 0.230, TOP - BOT])
    vgroup(axA, [(f"max_cross = {m}\n({int(AN[i].mean()/1000)}k)", A[i, :, 1], ss.BLUE)
                 for i, m in enumerate(["1", "2", "3", "all"])],
           "a", "Seeding in crossing voxels", "Seed contrast")

    axB = fig.add_axes([0.395, BOT, 0.230, TOP - BOT])
    vgroup(axB, [("Discrete peaks", B[0, :, 1], ss.BLUE),
                 ("Deterministic max.", B[1, :, 1], ss.ORANGE),
                 ("Probabilistic", B[2, :, 1], ss.GREEN),
                 ("MRtrix SD_Stream", det["seed"], ss.GRAY)],
           "b", "Propagation rule, and software", "Seed contrast")
    # separate software right of the divider: the first three rows share one fODF, one seed
    # set, one stopping rule
    axB.axvline(2.5, color="black", lw=.6, ls=(0, (2.5, 2)), zorder=1)

    axC = fig.add_axes([0.735, BOT, 0.250, TOP - BOT])
    x = (ep["edges"][:-1] + ep["edges"][1:]) / 2
    YTOP = float(np.nanmax([np.nanmean(ep["hist"][k], 0).max() for k in range(3)]))
    for i, (nm, col) in enumerate(zip(ep["srcs"], (ss.BLUE, ss.ORANGE, ss.GRAY))):
        m = np.nanmean(ep["hist"][i], 0)
        excluded = str(nm).endswith("iFOD2")
        lab = str(nm).replace("MRtrix ", "") + (" (excluded)" if excluded else "")
        axC.plot(x, m, color=col, lw=1.4, label=lab, ls=(0, (3, 1.5)) if excluded else "-",
                 solid_capstyle="round", zorder=3)
    axC.plot([0, 0], [0, YTOP * 1.04], color="black", lw=.6, zorder=1)
    axC.plot([-2, -2], [0, YTOP * 1.04], color="black", lw=.6, ls=(0, (2.5, 2)), zorder=1)
    axC.text(-2.25, 0.004, "node depth", fontsize=6, color="black", rotation=90, ha="right",
             va="bottom")
    axC.set_xlim(-9, 7)
    axC.set_ylim(0, YTOP * 1.34)
    axC.tick_params(labelsize=6)
    ss.frame(axC)
    axC.set_xlabel("Distance from white surface (mm)", fontsize=7.5, labelpad=2)
    axC.set_ylabel("Density", fontsize=7.5, labelpad=2)
    leg = axC.legend(loc="upper right", handlelength=1.2, fontsize=5.8, borderpad=.4,
                     labelspacing=.35, handletextpad=.5, frameon=True, edgecolor="black",
                     facecolor="white", framealpha=1, bbox_to_anchor=(0.99, 0.99))
    leg.get_frame().set_linewidth(0.7)
    ns.ptitle(axC, "", "c", lx=-0.22, y=1.02)

    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{RES}/FIGS_tracking.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    from PIL import Image
    w, h = Image.open(f"{RES}/FIGS_tracking.png").size
    print(f"wrote FIGS_tracking.png  {w/DPI*25.4:.0f} x {h/DPI*25.4:.0f} mm, {w} x {h} px")


if __name__ == "__main__":
    main()

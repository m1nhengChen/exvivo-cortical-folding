#!/usr/bin/env python3
"""Fig. 4 - three-hinge gyri preferentially interconnect. Nature spec: 183 mm double column,
5-7 pt type.

 a  node x node connectivity for one specimen, nodes ordered [3HG | matched crown].
 b  the three connection blocks resolved by connection length.
 c  the same, as enrichment ratios against connection length.
 d  the seed x target 2x2 the claim rests on, as a value matrix.
 e  enrichment under three contrasts, per specimen.
 f  the hinge network in the brain: the streamlines that link one 3HG to another.
 g  microstructure along the gyral-skeleton degree ladder (folding convergence).
 h  3HG-3HG enrichment against the distance/region matched null.
b-e are computed by validate/seed_control.py with one estimator throughout; a visualizes the same
streamlines resolved into spatially distinct patches (tracto/conn_matrix.py); f renders that same
selection in 3D (tracto/render_hinge_net.py); g-h come from fig_supp_geometry.py."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec, colors as mcolors
from matplotlib.patches import Rectangle
from PIL import Image
import natstyle as ns, fig_supp_geometry as sg, fig_hinge_order as ho
ns.nature()
TR = (_env.ROOT + "/surf/three_hinge/tracto")
VAL = (_env.ROOT + "/surf/three_hinge/validate")
DPI = 1600                     # one raster resolution across the whole figure set
RES = (_env.ROOT + "/surf/three_hinge/results")
SUBJ = [1, 2, 3, 4, 5, 6]; SHOW = 5

CMAP = ns.SEQ                               # value scale, same blue hue as the class colours
C_H, C_C = ns.C_3HG, ns.C_CROWN            # 3HG-3HG block / crown-crown block
C_M = ns.BLUE_L                             # 3HG-crown block, the intermediate tone
C_OK = ns.C_3HG                             # the enrichments are 3HG-ward quantities
STROKE = [pe.withStroke(linewidth=1.6, foreground="white")]


def pstr(p):
    if p >= 1e-3:
        return f"$P$ = {p:.3f}"
    e = int(np.floor(np.log10(p)))
    return f"$P$ = {p / 10 ** e:.0f}" + r" $\times$ 10$^{" + str(e) + "}$"


def band(ax, x, Y, col, lab, ls="-", lw=1.2, mu=None):
    """mean (or a supplied central curve) +- s.e.m. across specimens."""
    se = np.nanstd(Y, 0, ddof=1) / np.sqrt(Y.shape[0])
    mu = np.nanmean(Y, 0) if mu is None else mu
    ax.fill_between(x, mu - se, mu + se, color=col, alpha=.16, lw=0, zorder=2)
    ax.plot(x, mu, color=col, lw=lw, ls=ls, label=lab, solid_capstyle="round", zorder=3)
    return mu


PAIR_E = np.arange(10, 101, 15.0)                 # inter-node distance bins for panel b


def pair_stats():
    """pair-level incidence and strength, per specimen, from the node x node matrices

    The seed-level fraction f that panels d,e use is a STREAMLINE-WEIGHTED quantity: it multiplies
    how often a pair connects at all by how much of a seed's tractogram the connection carries.
    Those two move in opposite directions here - 3HG pairs connect 1.14x more often (5/6,
    P = 0.035) but carry 0.96x the streamlines once connected (2/6, P = 0.34) - so f cancels them
    and f_hh/f_cc lands at 1.03 (P = 0.42). Splitting them is the honest panel: the effect is in
    connection incidence, not connection strength."""
    inc, stre, curves = [], [], []
    for sid in SUBJ:
        z = np.load(f"{TR}/Subject{sid}/conn_matrix.npz")
        kind = z["kind"]; A = (((z["Mb"] if "Mb" in z.files else z["M"]) + (z["Mb"] if "Mb" in z.files else z["M"]).T) > 0); F = z["F"] + z["F"].T; D = z["nodedist"]
        h = np.where(kind == 0)[0]; c = np.where(kind == 1)[0]
        iu = lambda n: np.triu_indices(n, 1)
        blk = {"hh": (A[np.ix_(h, h)][iu(len(h))], F[np.ix_(h, h)][iu(len(h))],
                      D[np.ix_(h, h)][iu(len(h))]),
               "cc": (A[np.ix_(c, c)][iu(len(c))], F[np.ix_(c, c)][iu(len(c))],
                      D[np.ix_(c, c)][iu(len(c))]),
               "hc": (A[np.ix_(h, c)].ravel(), F[np.ix_(h, c)].ravel(), D[np.ix_(h, c)].ravel())}
        inc.append([blk[k][0].mean() for k in ("hh", "hc", "cc")])
        stre.append([blk[k][1][blk[k][0]].mean() for k in ("hh", "hc", "cc")])
        row = []
        for k in ("hh", "hc", "cc"):
            a, _, d = blk[k]; b = np.digitize(d, PAIR_E) - 1
            # 30 pairs is the floor for a per-specimen proportion to be worth plotting
            row.append([a[b == i].mean() if (b == i).sum() >= 30 else np.nan
                        for i in range(len(PAIR_E) - 1)])
        curves.append(row)
    return np.array(inc), np.array(stre), np.array(curves)


def main():
    z = np.load(f"{VAL}/seed_control.npz")
    fhh, fhc, fch, fcc = z["f_hh"], z["f_hc"], z["f_ch"], z["f_cc"]
    C, E = z["curves"], z["edges"]                     # C: (specimen, {hh,hc,ch,cc}, bin)
    x = (E[:-1] + E[1:]) / 2
    HH, HC, CH, CC = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    HCm = np.nanmean([HC, CH], 0)                      # 3HG-crown block, both directions
    # keep only lengths where every specimen contributes non-zero connections in all four
    # seed x target combinations; beyond ~70 mm too few streamlines survive per seed and the
    # per-seed fractions (and hence their ratios) become unstable.
    ok = np.all([(A > 0).all(0) for A in (HH, HC, CH, CC)], 0)
    x, HH, HC, CH, CC, HCm = x[ok], HH[:, ok], HC[:, ok], CH[:, ok], CC[:, ok], HCm[:, ok]
    XLIM = (E[0], x[-1] + (E[1] - E[0]) / 2)

    PI, PS, PC = pair_stats()
    tgt_c, seed_c = fhh / fhc, fhh / fch
    inter = tgt_c / (fch / fcc)
    P = [stats.ttest_1samp(np.log(v), 0).pvalue for v in (tgt_c, seed_c, inter)]

    # two rows: the a-e block keeps its own 88 mm geometry, the new bottom row adds BOTH mm.
    TOP_MM, BOT_MM = 95.0, 60.0        # + room for the per-panel titles
    FW, FH = ns.W2, (TOP_MM + BOT_MM) * ns.MM
    frac = BOT_MM / (TOP_MM + BOT_MM)                  # canvas fraction below the a-e block

    def yt(y):
        """old within-block fraction -> canvas fraction, so a-e keep their tuned layout"""
        return frac + y * (1 - frac)

    L, R = 0.055, 0.988
    T, B = yt(0.930), yt(0.150)
    # gap between the matrix and the right block: it must hold the colorbar AND the y-axis
    # decorations of b/d, which extend left of the gridspec edge.
    CB = 0.118
    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    # the matrix is square, so give panel a a box whose width equals its height in absolute units;
    # its height then matches the full b-e block rather than leaving slack in the grid cell.
    AH = T - B
    AW = AH * FH / FW                                  # square in absolute units, both rescaled
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], left=L + AW + CB, right=R, top=T, bottom=B,
                           wspace=0.38, hspace=0.55)   # room for the per-panel titles

    # ------------------------------------------------------------------ a: connectivity matrix
    axA = fig.add_axes([L, B, AW, AH])
    m = np.load(f"{TR}/Subject{SHOW}/conn_matrix.npz")
    F, kind, parc, pos = m["F"], m["kind"], m["parc"], m["pos"]
    n = len(kind) // 2; N = 2 * n
    order = np.lexsort((pos[:n, 1], parc[:n]))
    idx = np.r_[order, order + n]
    VMAX = 0.10
    im = axA.imshow(F[np.ix_(idx, idx)], cmap=CMAP, interpolation="nearest",
                    norm=mcolors.PowerNorm(0.32, vmin=0, vmax=VMAX))
    axA.axhline(n - .5, color="white", lw=1.3); axA.axvline(n - .5, color="white", lw=1.3)
    axA.axhline(n - .5, color=ns.INK, lw=0.6, ls=(0, (3, 2)))
    axA.axvline(n - .5, color=ns.INK, lw=0.6, ls=(0, (3, 2)))
    for px, py, lab, col in [(n * .5, n * .5, "3HG–3HG", C_H),
                             (n * 1.5, n * .5, "3HG–crown", ns.MUTE),
                             (n * .5, n * 1.5, "3HG–crown", ns.MUTE),
                             (n * 1.5, n * 1.5, "crown–crown", C_C)]:
        axA.text(px, py, lab, ha="center", va="center", fontsize=6.5, fontweight="bold",
                 color=col, path_effects=STROKE)
    bar, gap = 0.052 * N, 0.007 * N                    # class strips on both margins, labels inside
    for a, b, col, lab in [(0, n, C_H, "3HG"), (n, N, C_C, "Crown")]:
        axA.add_patch(Rectangle((-bar - gap, a - .5), bar, b - a, color=col, clip_on=False, lw=0))
        axA.add_patch(Rectangle((a - .5, -bar - gap), b - a, bar, color=col, clip_on=False, lw=0))
        axA.text(-bar / 2 - gap, (a + b) / 2 - .5, lab, rotation=90, ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold", clip_on=False)
        axA.text((a + b) / 2 - .5, -bar / 2 - gap, lab, ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold", clip_on=False)
    axA.set_xticks([]); axA.set_yticks([]); axA.set_xlim(-.5, N - .5); axA.set_ylim(N - .5, -.5)
    for sp in axA.spines.values(): sp.set_visible(False)
    # The matrix is aspect-constrained and anchored north, so it fills only the upper part of its
    # grid cell. A colorbar attached with ax=axA centres on that CELL, not on the image, and ends up
    # visibly low; place it against the image's drawn box instead.
    fig.canvas.draw()
    q = axA.get_position()
    cax = fig.add_axes([q.x1 + 0.006, q.y0 + q.height * 0.19, 0.0135, q.height * 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, VMAX]); cb.set_ticklabels(["0", "0.1"])
    cb.ax.yaxis.set_ticks_position("right"); cb.ax.yaxis.set_label_position("right")
    cb.set_label("Streamline fraction", fontsize=7, rotation=90)
    # the two tick labels sit at the very ends of the bar, so the label can be pulled in beside its
    # middle instead of clearing their width
    cb.ax.yaxis.set_label_coords(1.6, 0.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=1.5, labelsize=6, pad=1.0)

    # ------------------------------- b: probability a node pair is connected, vs their distance
    # Pair level, not seed level. The seed-level curves were too noisy to read at n = 6 - beyond
    # ~50 mm a seed puts only a handful of streamlines in a 10 mm bin - whereas each of these bins
    # holds 900-2600 pairs per specimen. The block ordering 3HG-3HG > 3HG-crown > crown-crown then
    # holds at EVERY distance, which the seed-level version did not.
    px = (PAIR_E[:-1] + PAIR_E[1:]) / 2
    axB = fig.add_subplot(gs[0, 0])
    keep = np.all(np.isfinite(PC), (0, 1))
    for j, (col, lab) in enumerate([(C_H, "3HG–3HG"), (C_M, "3HG–crown"), (C_C, "crown–crown")]):
        band(axB, px[keep], 100 * PC[:, j][:, keep], col, lab)
    axB.set_yscale("log")
    axB.set_xlim(PAIR_E[0], PAIR_E[-1]); axB.set_xticks([20, 50, 80])
    # log axis: one tick length per step size, so the two families are told apart at a glance -
    # the 10-step ticks (10, 20, 30) long, the 1-step ticks (1 ... 9) short
    axB.set_ylim(0.7, 38)
    axB.set_yticks([10, 20, 30]); axB.set_yticklabels(["10", "", "30"])
    axB.set_yticks(list(range(1, 10)), minor=True)
    axB.set_yticklabels(["1"] + [""] * 8, minor=True)
    axB.tick_params(axis="y", which="major", length=3.4, width=0.5)
    axB.tick_params(axis="y", which="minor", length=1.5, width=0.4, labelsize=6)
    axB.set_xlabel("Inter-node distance (mm)", fontsize=7, labelpad=1.5)
    axB.set_ylabel("Pairs connected (%)", fontsize=7, labelpad=2)
    axB.legend(loc="upper right", handlelength=1.0, fontsize=5.8, borderpad=.2,
               labelspacing=.28, handletextpad=.5)
    ns.ptitle(axB, "Incidence falls with distance", y=1.06)

    # -------------------------- c: the effect is in incidence, not strength - the two split apart
    axC = fig.add_subplot(gs[0, 1])
    axC.axhline(1.0, color=ns.INK, lw=.5, ls=(0, (2.5, 2)), zorder=1)
    RI = PI[:, 0] / PI[:, 2]                                   # 3HG / crown, per specimen
    RS = PS[:, 0] / PS[:, 2]
    for i, (v, lab) in enumerate([(RI, "Incidence"), (RS, "Strength")]):
        col = C_OK if np.exp(np.mean(np.log(v))) > 1 else ns.MUTE
        axC.scatter(np.full(len(v), i) + np.linspace(-.16, .16, len(v)), v, s=7, color=ns.INK,
                    zorder=4, linewidth=0)
        axC.bar(i, np.exp(np.mean(np.log(v))), 0.52, color=col, alpha=.85, zorder=2)
        axC.errorbar(i, np.exp(np.mean(np.log(v))),
                     yerr=np.exp(np.mean(np.log(v))) * np.std(np.log(v), ddof=1) / np.sqrt(len(v)),
                     fmt="none", ecolor=ns.INK, elinewidth=.8, capsize=1.8, zorder=5)
        p = stats.ttest_1samp(np.log(v), 0).pvalue
        axC.text(i, 1.62, f"{np.exp(np.mean(np.log(v))):.2f}×", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold")
        axC.text(i, 1.50, pstr(p), ha="center", va="bottom", fontsize=5.2)
    axC.set_xlim(-.6, 1.6); axC.set_xticks([0, 1])
    axC.set_xticklabels(["Pairs\nconnected", "Streamlines\nper connection"], fontsize=6.2,
                        linespacing=1.25)
    axC.set_ylim(0, 1.95); axC.set_yticks([0, 0.5, 1.0, 1.5])
    axC.set_ylabel("3HG / crown", fontsize=7, labelpad=2)
    axC.tick_params(axis="x", length=0)
    ns.ptitle(axC, "Incidence versus strength", y=1.06)

    # ------------------------------------------- d: the seed x target 2x2, as a value matrix
    axD = fig.add_subplot(gs[1, 0])
    G = np.array([[fhh.mean(), fhc.mean()], [fch.mean(), fcc.mean()]])
    S = np.array([[fhh, fhc], [fch, fcc]])
    axD.imshow(G, cmap=CMAP, norm=mcolors.PowerNorm(0.8, vmin=0, vmax=0.34), aspect="auto")
    for i in range(2):
        for j in range(2):
            axD.text(j, i - 0.13, f"{G[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                     fontweight="bold", color="white")
            axD.text(j, i + 0.17, f"± {S[i, j].std(ddof=1) / np.sqrt(6):.3f}", ha="center",
                     va="center", fontsize=5.6, color="#e4e4e4")
    axD.set_xticks([0, 1]); axD.set_yticks([0, 1])
    axD.set_xticklabels(["3HG", "Crown"], fontsize=6.5)
    axD.set_yticklabels(["3HG", "Crown"], fontsize=6.5, rotation=90, va="center")
    # labelpad clears the class tick labels, which sit between this title and the matrix
    axD.set_xlabel("Reaching", fontsize=7, labelpad=5)
    axD.set_ylabel("Seeded from", fontsize=7, labelpad=2, fontweight="bold")
    axD.xaxis.set_label_position("top"); axD.xaxis.tick_top()
    # colour the class labels only after tick_top(), which rebuilds the tick artists
    for t, col in zip(axD.get_xticklabels(), (C_H, C_C)): t.set_color(col); t.set_fontweight("bold")
    for t, col in zip(axD.get_yticklabels(), (C_H, C_C)): t.set_color(col); t.set_fontweight("bold")
    axD.tick_params(length=0)
    for sp in axD.spines.values(): sp.set_visible(False)
    axD.text(0.5, -0.075, f"Interaction   {pstr(P[2])}", transform=axD.transAxes, ha="center",
             va="top", fontsize=6.5)

    # ------------------------------------------- e: the three contrasts, per specimen
    axE = fig.add_subplot(gs[1, 1])
    Y = np.array([tgt_c, seed_c, inter])
    cols = ["#c9c9c4", C_OK, C_OK]          # grey = not coverage-controlled
    axE.axhline(1.0, color=ns.INK, lw=.5, ls=(0, (2.5, 2)), zorder=1)
    # geometric mean throughout: the tests are one-sample t on log ratios, and panel c already
    # summarises its ratios geometrically - the arithmetic mean printed 1.35/1.24/1.64 here while
    # the text carried 1.34/1.23/1.60, a figure-vs-manuscript mismatch waiting for a referee.
    for i in range(3):
        gm = np.exp(np.mean(np.log(Y[i])))
        axE.bar(i, gm, width=.62, color=cols[i], alpha=.85, edgecolor="white",
                linewidth=.5, zorder=2)
        axE.errorbar(i, gm, yerr=gm * np.std(np.log(Y[i]), ddof=1) / np.sqrt(6), fmt="none",
                     ecolor=ns.INK, elinewidth=.8, capsize=1.8, zorder=4)
        axE.scatter(np.full(6, i) + np.linspace(-.17, .17, 6), Y[i], s=5, color=ns.INK,
                    zorder=5, linewidth=0)
        axE.text(i, 5.02, f"{gm:.2f}×", ha="center", va="bottom", fontsize=6.5,
                 fontweight="bold")
        axE.text(i, 4.58, pstr(P[i]), ha="center", va="bottom", fontsize=5.2)
    # bracket above the two bars whose ratio is free of the target-coverage difference
    axE.plot([0.68, 2.32], [3.48, 3.48], color=ns.INK, lw=.7, zorder=6)
    for xx in (0.68, 2.32):
        axE.plot([xx, xx], [3.32, 3.48], color=ns.INK, lw=.7, zorder=6)
    axE.text(1.50, 3.58, "Coverage-controlled", ha="center", va="bottom", fontsize=6, color=ns.INK)
    axE.set_xlim(-.55, 2.55); axE.set_xticks([0, 1, 2])
    axE.set_xticklabels(["Target\ncontrast", "Seed\ncontrast", "Interaction"], fontsize=6.2,
                        linespacing=1.25)
    axE.set_ylim(0, 5.6); axE.set_yticks([0, 1, 2, 3, 4, 5])
    axE.set_ylabel("Enrichment", fontsize=7, labelpad=2)
    axE.tick_params(axis="x", length=0)

    # ------------------------------------------- f: the hinge network rendered in the brain
    # Blender/Cycles, two passes composited by tracto/compose_clay.py: the cortex rendered opaque
    # and then lightened toward the page, with the network laid over it and depth-faded. Rendering
    # the cortex translucent instead is what produced the earlier fog - a lateral ray crosses a
    # folded pial a dozen times and the alphas compound. The streamline selection is the cached
    # one of render_hinge_net.py: every fibre shown links one 3HG to a different 3HG.
    img = Image.open(f"{RES}/_net_clay.png")
    FW_, FB_ = 0.390, 0.052                       # panel width and bottom, figure fractions
    fh_ = (FW_ * FW / (img.size[0] / img.size[1])) / FH
    axF = fig.add_axes([0.032, FB_, FW_, fh_])
    # no in-panel key: the orange spheres are the only class drawn, and the legend names them
    axF.imshow(np.asarray(img), interpolation="lanczos"); axF.set_axis_off()

    # ------------------------------------------- g: the EDR null. The folding ladder that sat
    # beside it moved to the in-vivo validation figure (Fig. 4), which owns the folding story;
    # this figure is wiring only, so the null gets the width the ladder freed.
    GB, GH = 0.078, 0.272                          # g's bottom and height
    axH = fig.add_axes([0.562, GB, 0.418, GH])
    sg.edr_panel(axH, "", lx=-0.165)
    ns.ptitle(axH, "Against a distance-matched null", y=1.06)

    # Panel letters, one baseline per row. ptitle placed each letter in its own axes' fraction
    # space, and with three panel constructions per row (margin bars on a, a top-side axis on d,
    # an image box on f) the letters landed on three different lines.
    fig.canvas.draw()
    pos = {k: ax.get_position() for k, ax in
           (("a", axA), ("b", axB), ("c", axC), ("d", axD),
            ("e", axE), ("f", axF), ("g", axH))}
    y1 = pos["b"].y0 + 1.06 * pos["b"].height
    y2 = pos["d"].y0 + 1.26 * pos["d"].height          # clears d's top-side axis and its ticks
    y3 = max(pos["g"].y0 + 1.06 * pos["g"].height, pos["f"].y1 + 0.012)
    for letter, lx, y in (("a", -0.090, y1), ("b", -0.28, y1), ("c", -0.28, y1),
                          ("d", -0.28, y2), ("e", -0.28, y2),
                          ("f", -0.040, y3), ("g", -0.150, y3)):
        q = pos[letter]
        fig.text(q.x0 + lx * q.width, y, letter, fontsize=8, fontweight="bold", va="bottom")

    fig.savefig(f"{RES}/FIG5_connectivity.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{RES}/FIG5_connectivity.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote FIG5_connectivity.png", Image.open(f"{RES}/FIG5_connectivity.png").size)
    print("  block means over length:  3HG-3HG %.3f | 3HG-crown %.3f | crown-crown %.3f"
          % (np.nanmean(HH), np.nanmean(HCm), np.nanmean(CC)))
    r1, r2 = HH.mean(0) / CC.mean(0), HH.mean(0) / HCm.mean(0)
    print("  lengths kept: %.0f-%.0f mm (%d bins)" % (XLIM[0], XLIM[1], len(x)))
    print("  enrichment vs length: HH/CC %.2f-%.2f, HH/3HG-crown %.2f-%.2f"
          % (r1.min(), r1.max(), r2.min(), r2.max()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Small supplementary figure: 3HG density, primary vs non-primary cortex, both datasets.

 a,b  per unit cortical AREA (junctions / cm2 white surface): ex-vivo pairs, in-vivo violins.
 c,d  sensitivity analysis, per unit GYRAL SKELETON LENGTH (junctions / 100 mm of ridge):
      independent of surface area and of mesh sampling density. In vivo the enrichment
      survives (1.15x, P = 7e-97); ex vivo it does not (1.03x, P = 0.29) - there the area
      effect is carried by the denser ridge system of non-primary cortex (1.18x skeleton
      length per area, P = 0.0004). Shown, not hidden: the per-cm2 enrichment must not be
      read as independent of folding complexity.
Primary = DK precentral, postcentral, pericalcarine, transversetemporal; every test is a
one-sample t on the per-hemisphere/subject log rate ratio.
Data: primary_density.py, primary_density_skel.py."""
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


def main():
    ex = np.load(f"{RES}/primary_density_exvivo.npz")["data"]
    iv = np.load(f"{RES}/primary_density_invivo.npz")["data"]
    exp_, exn = ex[:, 0] / ex[:, 1] * 100, ex[:, 2] / ex[:, 3] * 100
    ivp, ivn = iv[:, 0] / iv[:, 1] * 100, iv[:, 2] / iv[:, 3] * 100

    exs = np.load(f"{RES}/primary_skel_exvivo.npz")["data"]
    ivs = np.load(f"{RES}/primary_skel_invivo.npz")["data"]
    esp, esn = exs[:, 0] / exs[:, 1] * 100, exs[:, 2] / exs[:, 3] * 100
    isp, isn = ivs[:, 0] / ivs[:, 1] * 100, ivs[:, 2] / ivs[:, 3] * 100

    fig = plt.figure(figsize=(120 * ns.MM, 110 * ns.MM), facecolor="white")

    # ---- a: ex vivo, paired ----
    ax = fig.add_axes([0.105, 0.585, 0.365, 0.355])
    for a, b in zip(exp_, exn):
        ax.plot([0, 1], [a, b], color="#b8b8b3", lw=0.8, zorder=2)
    ax.scatter(np.zeros(6), exp_, s=16, color=ss.BLUE, edgecolor="black", linewidth=0.6,
               zorder=4)
    ax.scatter(np.ones(6), exn, s=16, color=ss.ORANGE, edgecolor="black", linewidth=0.6,
               zorder=4)
    p = stats.ttest_1samp(np.log(exn / exp_), 0).pvalue
    gm = np.exp(np.mean(np.log(exn / exp_)))
    ax.set_ylim(0.30, 1.06)
    ss.sig_bracket(ax, 0, 1, 0.97, ss.stars(p))
    ax.text(0.5, 0.885, f"{gm:.2f}×", ha="center", va="bottom", fontsize=6, color="black")
    ax.set_xticks([0, 1]); ax.set_xlim(-0.45, 1.45)
    ax.set_xticklabels(["Primary", "Non-primary"], fontsize=6.6)
    ax.set_ylabel("3HG per cm$^2$", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ss.frame(ax)
    ns.ptitle(ax, "Ex vivo", "a", lx=-0.26, y=1.05)

    # ---- b: in vivo, violins ----
    ax = fig.add_axes([0.610, 0.585, 0.365, 0.355])
    ss.violin_box(ax, 0, ivp, ss.BLUE, width=0.62)
    ss.violin_box(ax, 1, ivn, ss.ORANGE, width=0.62)
    p = stats.ttest_1samp(np.log(ivn / ivp), 0).pvalue
    gm = np.exp(np.mean(np.log(ivn / ivp)))
    ax.set_ylim(0.075, 0.315)
    ss.sig_bracket(ax, 0, 1, 0.290, ss.stars(p))
    ax.text(0.5, 0.264, f"{gm:.2f}×", ha="center", va="bottom", fontsize=6, color="black")
    ax.set_xticks([0, 1]); ax.set_xlim(-0.65, 1.65)
    ax.set_xticklabels(["Primary", "Non-primary"], fontsize=6.6)
    ax.set_ylabel("3HG per cm$^2$", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ss.frame(ax)
    ns.ptitle(ax, "In vivo", "b", lx=-0.30, y=1.05)

    # ---- c: ex vivo, per skeleton length ----
    ax = fig.add_axes([0.105, 0.088, 0.365, 0.355])
    for a, b in zip(esp, esn):
        ax.plot([0, 1], [a, b], color="#b8b8b3", lw=0.8, zorder=2)
    ax.scatter(np.zeros(6), esp, s=16, color=ss.BLUE, edgecolor="black", linewidth=0.6,
               zorder=4)
    ax.scatter(np.ones(6), esn, s=16, color=ss.ORANGE, edgecolor="black", linewidth=0.6,
               zorder=4)
    p = stats.ttest_1samp(np.log(esn / esp), 0).pvalue
    gm = np.exp(np.mean(np.log(esn / esp)))
    ax.set_ylim(3.4, 7.6)
    ss.sig_bracket(ax, 0, 1, 7.05, "n.s.", fontsize=5.6)
    ax.text(0.5, 6.42, f"{gm:.2f}×", ha="center", va="bottom", fontsize=6, color="black")
    ax.set_xticks([0, 1]); ax.set_xlim(-0.45, 1.45)
    ax.set_xticklabels(["Primary", "Non-primary"], fontsize=6.6)
    ax.set_ylabel("3HG per 100 mm skeleton", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ss.frame(ax)
    ns.ptitle(ax, "Ex vivo", "c", lx=-0.26, y=1.05)

    # ---- d: in vivo, per skeleton length ----
    ax = fig.add_axes([0.610, 0.088, 0.365, 0.355])
    ss.violin_box(ax, 0, isp, ss.BLUE, width=0.62)
    ss.violin_box(ax, 1, isn, ss.ORANGE, width=0.62)
    p = stats.ttest_1samp(np.log(isn / isp), 0).pvalue
    gm = np.exp(np.mean(np.log(isn / isp)))
    ax.set_ylim(1.75, 4.05)
    ss.sig_bracket(ax, 0, 1, 3.82, ss.stars(p))
    ax.text(0.5, 3.53, f"{gm:.2f}×", ha="center", va="bottom", fontsize=6, color="black")
    ax.set_xticks([0, 1]); ax.set_xlim(-0.65, 1.65)
    ax.set_xticklabels(["Primary", "Non-primary"], fontsize=6.6)
    ax.set_ylabel("3HG per 100 mm skeleton", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ss.frame(ax)
    ns.ptitle(ax, "In vivo", "d", lx=-0.30, y=1.05)

    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{RES}/FIGS_primdens.{ext}", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.02)
    from PIL import Image
    w, h = Image.open(f"{RES}/FIGS_primdens.png").size
    print(f"wrote FIGS_primdens.png  {w/DPI*25.4:.0f} x {h/DPI*25.4:.0f} mm")


if __name__ == "__main__":
    main()

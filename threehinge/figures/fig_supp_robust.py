#!/usr/bin/env python3
"""Supplementary figure - the two robustness checks.

 a  the depth effect under three analyses - as sampled, restricted to voxels above 70% grey
    matter, and adjusted for the white-matter and non-brain fractions. Nothing moves.
 b  inter-hinge enrichment under ten tracking parameter sets, rebuilt from the stored CSD peaks.

The partial volume of the sampled voxels is not plotted: the 3HG and crown curves lie on top of
each other, and a figure cannot show 'identical' any better than the numbers can. It is written
out as source data instead, together with the per-specimen tracking results behind b:
    FIGS_robust_tissue_fraction.csv   specimen x depth x class, GM / WM / non-brain fractions
    FIGS_robust_tracking.csv          setting x specimen, streamline count and both contrasts
Data: depth/depth_pv.py, validate/tracto_sensitivity.py."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import natstyle as ns

D = (_env.ROOT + "/surf/three_hinge/depth")
VAL = (_env.ROOT + "/surf/three_hinge/validate")
RES = (_env.ROOT + "/surf/three_hinge/results")
# One raster resolution for every supplementary figure. These were set per-figure as each
# was written and drifted to 800 / 1200 / 1600 dpi, which shows as three different line
# weights when they are printed side by side.
DPI = 1600
SUBJ = [1, 2, 3, 4, 5, 6]
MET = [("MD", "MD"), ("hindered", "hindered"), ("water", "free water"), ("AD", "AD"),
       ("RD", "RD"), ("fiber", "fiber"), ("restricted", "restricted"), ("FA", "FA")]
import suppstyle as _ss
VAR3 = [("wpraw", "as sampled", _ss.BLUE),
        ("wppure", "GM > 70%", _ss.ORANGE),
        ("wpadj", "PV-adjusted", _ss.GREEN)]


def export_csv():
    """source data for the two robustness checks, in tidy form"""
    Z = [np.load(f"{D}/Subject{s}_depth_pv.npz") for s in SUBJ]
    rho = Z[0]["rho"]
    rows = ["specimen,depth_rho,class,gm_fraction,wm_fraction,nonbrain_fraction,"
            "frac_samples_ge70pct_gm"]
    for s_, z in zip(SUBJ, Z):
        for j, r in enumerate(rho):
            for cls, tag in (("h", "3HG"), ("c", "crown")):
                gm, wm = z[f"gm_{cls}"][j], z[f"wm_{cls}"][j]
                rows.append(f"{s_},{r:.2f},{tag},{gm:.4f},{wm:.4f},{1 - gm - wm:.4f},"
                            f"{z['npure'][j]:.4f}")
    open(f"{RES}/FIGS_robust_tissue_fraction.csv", "w").write("\n".join(rows) + "\n")

    z = np.load(f"{VAL}/tracto_sensitivity.npz")
    names = [str(n) for n in z["names"]]; sub = z["subj"]
    rows = ["setting,specimen,n_streamlines,target_contrast,seed_contrast"]
    for i, nm in enumerate(names):
        for j, sd in enumerate(sub):
            rows.append(f"{nm},{int(sd)},{int(z['nstream'][i, j])},"
                        f"{z['target'][i, j]:.4f},{z['seed'][i, j]:.4f}")
    open(f"{RES}/FIGS_robust_tracking.csv", "w").write("\n".join(rows) + "\n")
    print("wrote FIGS_robust_tissue_fraction.csv, FIGS_robust_tracking.csv")


def panel_stable(ax):
    """grouped violins, example-figure style: metric on x, the three analyses as
    blue / orange / green violins with black inner boxes"""
    import suppstyle as ss
    Z = [np.load(f"{D}/Subject{s}_depth_pv.npz") for s in SUBJ]
    ax.axhline(0, color="black", lw=.7, ls=(0, (4, 2.5)), zorder=1)
    G, O = 3.3, 0.95
    for i, (m, lab) in enumerate(MET):
        for j, (pre, vlab, col) in enumerate(VAR3):
            v = np.array([np.nanmean(z[f"{pre}_{m}"]) for z in Z])
            ss.violin_box(ax, i * G + (j - 1) * O, v, col, width=0.85)
    ax.set_xticks([i * G for i in range(len(MET))])
    ax.set_xticklabels([l for _, l in MET], fontsize=6.4, rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-1.9, (len(MET) - 1) * G + 1.9)
    ax.set_ylim(-0.40, 0.34)
    ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_ylabel("3HG − crown (SD)", fontsize=7.5, labelpad=2)
    ax.tick_params(axis="y", labelsize=6)
    ss.frame(ax)
    ss.legend_swatch(ax, [(c, t) for _, t, c in VAR3], loc="upper left",
                     bbox_to_anchor=(0.012, 0.985))
    ns.ptitle(ax, "", "a", lx=-0.115, y=1.02)


def panel_track(ax):
    """parameter set on x, the two contrasts as orange / blue violin pairs on a log axis"""
    import suppstyle as ss
    z = np.load(f"{VAL}/tracto_sensitivity.npz")
    names = [str(n) for n in z["names"]]; T, S = z["target"], z["seed"]
    ax.set_yscale("log")
    ax.axhline(1, color="black", lw=.7, ls=(0, (4, 2.5)), zorder=1)
    G, O = 2.6, 0.55
    for i, nm in enumerate(names):
        for arr, col, off in ((T[i], ss.ORANGE, -O), (S[i], ss.BLUE, +O)):
            v = arr[np.isfinite(arr)]
            ss.violin_box(ax, i * G + off, v, col, width=0.95, tf=np.log, itf=np.exp)
    lab = ["Main results" if n == "published" else n for n in names]
    ax.set_xticks([i * G for i in range(len(names))])
    ax.set_xticklabels(lab, fontsize=6.2, rotation=40, ha="right", rotation_mode="anchor")
    for t, n in zip(ax.get_xticklabels(), names):
        if n == "published":
            t.set_fontweight("bold")
    ax.set_xlim(-1.6, (len(names) - 1) * G + 1.6)
    ax.set_ylim(0.72, 3.1)
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["1", "2", "3"], fontsize=6)
    ax.minorticks_off()
    ax.set_ylabel("Enrichment (log scale)", fontsize=7.5, labelpad=2)
    ss.frame(ax)
    ss.legend_swatch(ax, [(ss.ORANGE, "target contrast"), (ss.BLUE, "seed contrast")],
                     loc="upper right", bbox_to_anchor=(0.988, 0.985))
    ns.ptitle(ax, "", "b", lx=-0.10, y=1.02)


def main():
    ns.nature()
    export_csv()
    fig = plt.figure(figsize=(ns.W2, 78 * ns.MM), facecolor="white")
    panel_stable(fig.add_axes([0.058, 0.230, 0.375, 0.630]))
    panel_track(fig.add_axes([0.530, 0.230, 0.452, 0.630]))
    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{RES}/FIGS_robust.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print("wrote FIGS_robust.png", Image.open(f"{RES}/FIGS_robust.png").size)


if __name__ == "__main__":
    main()

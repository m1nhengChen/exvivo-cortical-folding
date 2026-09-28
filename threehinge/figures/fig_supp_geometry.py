#!/usr/bin/env python3
"""Individual geometry and the wiring-distance null.

 own_panel  each host's tractography scored with its own 3HG configuration and with the five
            registered from other specimens, plus the group summary.
 edr_panel  EDR-style matched null: 3HG-3HG enrichment before and after distance/region
            matching, per specimen, against the matched-null baseline of 1.
Run as a script this writes the standalone supplementary figure; the two panel functions are
also imported by fig_connectivity_matrix.py, which carries them as the bottom row of Fig. 4.
Data from validate/own_vs_foreign.npz and validate/edr_null.npz."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import natstyle as ns
VAL = (_env.ROOT + "/surf/three_hinge/validate")
RES = (_env.ROOT + "/surf/three_hinge/results")
SUBJ = [1, 2, 3, 4, 5, 6]
C_OWN, C_OTHER = ns.C_3HG, ns.BLUE_L               # filled orange = own, open pale blue = others
C_FLASH = "#8f9aa6"                                # donor-only FLASH cohort, a neutral slate


def pstr(p):
    return f"$P$ = {p:.3f}" if p >= 1e-3 else f"$P$ = {p:.0e}"


def own_data():
    """own[i]; the donor values split by cohort; the mean over ALL donors; and the paired P."""
    z = np.load(f"{VAL}/own_vs_foreign.npz")
    E = z["Epub"]
    nsub = 6
    flash = set(z["flash"].tolist()) if "flash" in z.files else set()
    donor = z["donor"].tolist() if "donor" in z.files else list(range(1, E.shape[1] + 1))
    own = np.array([E[i, i] for i in range(nsub)])
    dwi = [np.array([E[i, j] for j, d in enumerate(donor) if d not in flash and j != i])
           for i in range(nsub)]
    fla = [np.array([E[i, j] for j, d in enumerate(donor) if d in flash]) for i in range(nsub)]
    fmean = np.array([np.concatenate([dwi[i], fla[i]]).mean() for i in range(nsub)])
    return own, dwi, fla, fmean, stats.ttest_rel(np.log(own), np.log(fmean))[1]


def edr_data():
    q = np.load(f"{VAL}/edr_null.npz")
    Eb, Ub = q["E_binary"], q["U_binary"]           # matched null vs the same ratio unmatched
    return Eb, Ub, np.exp(np.log(Eb).mean()), np.exp(np.log(Ub).mean()), \
        stats.ttest_1samp(np.log(Eb), 0)[1]


def own_panel(ax, letter, lx=-0.115, ly=1.10):
    """Own vs every transferred 3HG configuration, one column per host. Donors from the DWI cohort
    and from the surface-only FLASH cohort are given different markers: the FLASH donors were
    reconstructed by a different pipeline from a different acquisition, so they are the control
    that no shared processing step can explain."""
    own, dwi, fla, fmean, p = own_data()
    for i in range(6):
        for vals, mk, col, lab in ((dwi[i], "o", C_OTHER, "Other specimen"),
                                   (fla[i], "^", C_FLASH, "FLASH cohort")):
            if not len(vals):
                continue
            jit = np.linspace(-.15, .15, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(i + jit, vals, s=6.5, marker=mk, facecolor="white", edgecolor=col,
                       linewidth=.6, zorder=3, label=lab if i == 0 else None)
        ax.plot([i, i], [fmean[i], own[i]], color="#c8c5c0", lw=.55, zorder=2,
                solid_capstyle="round")
        ax.plot([i - .19, i + .19], [fmean[i]] * 2, color="#9a968f", lw=.9, zorder=3,
                solid_capstyle="butt")
        ax.scatter([i], [own[i]], s=17, color=C_OWN, edgecolor="white", linewidth=.5,
                   zorder=5, label="Own" if i == 0 else None)
    xg = 6.9                                        # group summary, set off to the right
    ax.axvline(6.15, color="#dcdad6", lw=.6, zorder=1)
    for xoff, v, col, fc in ((-.17, fmean, C_OTHER, "white"), (.17, own, C_OWN, C_OWN)):
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(6)
        ax.errorbar(xg + xoff, m, yerr=se, fmt="none", ecolor=ns.INK, elinewidth=.7, capsize=1.6,
                    zorder=4)
        ax.scatter([xg + xoff], [m], s=17, facecolor=fc, edgecolor=col if fc == "white" else "white",
                   linewidth=.65, zorder=5)
    ax.plot([xg - .17, xg + .17], [fmean.mean(), own.mean()], color=ns.MUTE, lw=.6, zorder=2)
    ax.set_xticks(list(range(6)) + [xg])
    ax.set_xticklabels([f"S{s}" for s in SUBJ] + ["Mean"], fontsize=6)
    lo = min(own.min(), min(v.min() for v in dwi + fla if len(v)))
    hi = max(own.max(), max(v.max() for v in dwi + fla if len(v)))
    pad = 0.06 * (hi - lo)
    ax.set_xlim(-.55, 7.5); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Host specimen", fontsize=7, labelpad=2)
    ax.set_ylabel("Hinge-network enrichment", fontsize=7, labelpad=3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handlelength=.8,
              fontsize=5.6, borderpad=0, labelspacing=.25, handletextpad=.3, columnspacing=.8,
              frameon=False)
    ax.text(0.985, 0.96, pstr(p), transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ns.nletter(ax, letter, x=lx, y=ly)
    return p


def edr_panel(ax, letter, lx=-0.19, ly=1.10):
    """Same grammar as own_panel: filled orange = the reported quantity, open pale blue = what it
    is compared with. The stem runs from the null baseline to the matched value, so the pale
    marker splits it into the part present without distance control and the part matching adds."""
    Eb, Ub, gm, gu, pb = edr_data()
    ax.axhspan(0.90, 1.0, color="#f2f0ec", zorder=0, lw=0)
    ax.axhline(1.0, color=ns.INK, lw=.6, zorder=3)
    for i in range(6):
        ax.plot([i, i], [1.0, Eb[i]], color=C_OWN, lw=1.1, zorder=2, solid_capstyle="round")
        ax.scatter([i], [Ub[i]], s=11, facecolor="white", edgecolor=C_OTHER, linewidth=.7,
                   zorder=4, label="Unmatched" if i == 0 else None)
        ax.scatter([i], [Eb[i]], s=15, color=C_OWN, edgecolor="white", linewidth=.5, zorder=5,
                   label="Distance matched" if i == 0 else None)
    for v, col, lab in ((gu, C_OTHER, f"{gu:.2f}×"), (gm, C_OWN, f"{gm:.2f}×")):
        ax.plot([-.55, 5.42], [v, v], color=col, lw=.6, ls=(0, (3, 2)), zorder=1)
        ax.text(5.55, v, lab, color=col, fontsize=6, fontweight="bold", ha="left", va="center",
                clip_on=False)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"S{s}" for s in SUBJ], fontsize=6)
    ax.set_xlim(-.55, 6.55); ax.set_ylim(0.90, 1.62); ax.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax.set_yticklabels(["1", "1.2", "1.4", "1.6"])
    ax.set_xlabel("Specimen", fontsize=7, labelpad=2)
    ax.set_ylabel("3HG–3HG enrichment", fontsize=7, labelpad=3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1, handlelength=.8,
              fontsize=6, borderpad=0, labelspacing=.25, handletextpad=.4, columnspacing=.9,
              frameon=False)
    ax.text(0.985, 0.96, pstr(pb), transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ns.nletter(ax, letter, x=lx, y=ly)
    return gm, pb


def main():
    ns.nature()
    fig = plt.figure(figsize=(ns.W2 * 0.74, 54 * ns.MM), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.30, 1.0], left=0.085, right=0.985,
                           top=0.84, bottom=0.155, wspace=0.40)
    p = own_panel(fig.add_subplot(gs[0]), "a")
    gm, pb = edr_panel(fig.add_subplot(gs[1]), "b")
    fig.savefig(f"{RES}/FIGS_geometry.png", dpi=800, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{RES}/FIGS_geometry.pdf", bbox_inches="tight", pad_inches=0.02)
    print("wrote FIGS_geometry.png", Image.open(f"{RES}/FIGS_geometry.png").size,
          f"| own-vs-foreign P={p:.4f} | EDR {gm:.2f}x P={pb:.4f}")


if __name__ == "__main__":
    main()

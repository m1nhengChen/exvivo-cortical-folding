#!/usr/bin/env python3
"""What the tent mapping does to the intensity distribution.

  a   the N4 T2w as acquired: WM sits BELOW GM (inverted against T1) and the two
      distributions overlap heavily, which is exactly why a global threshold fails
      and why mris_place_surface cannot be pointed at the raw image
  b   b50 after the tent mapping: the same voxels, now with WM at FreeSurfer's 110
      target and GM at 60, in the T1-like ordering every downstream binary assumes

Colours match vis_tent_pipeline.py -- WM blue, GM green -- so the two figures read as
one set.  Subject5 by default: it is the specimen whose raw contrast is worst
(the B1 dropout is visible across its whole inferior half), so it is the honest case
to show rather than the flattering one.

    ./vis_b50_hist.py 5
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Arial is not installed here; Liberation Sans is metric-compatible with it (identical
# advance widths) and Nimbus Sans is the Helvetica clone -- either satisfies a journal that
# asks for Arial/Helvetica.  fonttype 42 embeds TrueType instead of Type 3, which some
# print pipelines reject.
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "custom"      # else math falls back to DejaVu
matplotlib.rcParams["mathtext.cal"] = "Liberation Sans"
for _k in ("rm", "it", "bf"):
    matplotlib.rcParams[f"mathtext.{_k}"] = "Liberation Sans" + (":italic" if _k == "it" else
                                                                 ":bold" if _k == "bf" else "")

ROOT = (_env.ROOT + "/28211843")
CSF = (_env.ROOT + "/surf/csf")
FS = (_env.ROOT + "/surf/fs")
OUT = (_env.ROOT + "/surf/results")

# Tissue colours, not class colours: grey and white matter are a different axis from 3HG vs crown,
# so they keep their own hues - but at the chroma the rest of the plate uses. These were a primary
# blue and a pure green (#008300), which sat next to the coral/slate panels like a different paper.
# Separated in lightness, not just hue. The overlay in panel a is drawn over a b50 slice where
# white matter is bright and grey matter dark, so two mid-tone colours of similar value wash out
# whatever their hue; so both are taken bright and separated by hue, with the alpha raised to carry them.
WM, GM, ALL = "#7FA6E8", "#3FC9A4", "#CFCBC5"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8a85"


def curve(v, lo, hi, bins=180):
    """Bin edges must respect the data's quantisation: b50 is uint8, so anything finer
    than one bin per integer level leaves empty bins between levels and the curve combs."""
    h, e = np.histogram(v, bins=bins, range=(lo, hi), density=True)
    return (e[:-1] + e[1:]) / 2, h


def overlap(data, lo, hi, bins):
    """Overlap coefficient of the WM and GM densities, int min(f_wm, f_gm) dx.

    This is the quantity the whole tent mapping exists to reduce: it is the fraction of
    tissue a *global* intensity threshold cannot separate no matter where it is placed.
    """
    xw, yw = curve(data["wm"], lo, hi, bins)
    _, yg = curve(data["gm"], lo, hi, bins)
    return float(np.minimum(yw, yg).sum() * (xw[1] - xw[0]))


def panel(ax, data, lo, hi, bins, xlabel, letter, title, marks=(), labels=()):
    # densities are ~1e-2, which prints as 0.0175 on the axis; carry the 10^-3 in the
    # label instead so the ticks stay integers
    K = 1e3
    x, y = curve(data["all"], lo, hi, bins)
    ax.fill_between(x, y * K, color=ALL, lw=0, zorder=1)

    peak = {}
    for key, col in (("wm", WM), ("gm", GM)):
        x, y = curve(data[key], lo, hi, bins)
        y = y * K
        ax.fill_between(x, y, color=col, alpha=.13, lw=0, zorder=2)
        ax.plot(x, y, color=col, lw=1.5, zorder=3, solid_capstyle="round")
        i = int(np.argmax(y))
        peak[key] = (x[i], y[i])

    top = max(v for _, v in peak.values()) * 1.30      # headroom for the direct labels
    ax.set_ylim(0, top)

    for xv, lab in marks:
        ax.axvline(xv, color=RULE, lw=.7, ls=(0, (4, 3)), zorder=1, ymax=.90)
        ax.text(xv, top * .915, lab, ha="center", va="bottom", fontsize=8.2, color=MUTED)

    # direct labels sit just above each mode, inside the axes -- identity without a box
    for key, col, dx, ha in labels:
        px, py = peak[key]
        ax.text(px + dx, min(py * 1.06, top * .93), key.upper(), color=col, ha=ha,
                va="bottom", fontsize=9.4, fontweight="semibold", zorder=4)

    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_xlim(lo, hi)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.tick_params(labelsize=8.6, colors=MUTED, width=.7, length=3.0, direction="out")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
        ax.spines[sp].set_linewidth(.6)
    if title:
        ax.set_title(title, fontsize=9.6, color=MUTED, loc="left", pad=6)
    if letter:
        ax.text(-0.10, 1.17, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", color=INK)


def run(sid, out, dpi=400):
    s = f"Subject{sid}"
    seg = np.asarray(nib.load(f"{ROOT}/{s}/{s}_Mask/T2w_GM_WM_mask.nii.gz").dataobj)
    raw = np.asarray(nib.load(f"{CSF}/S{sid}.t2w_n4.nii.gz").dataobj).astype(np.float32)
    A = dict(all=raw[seg > 0], wm=raw[seg == 2], gm=raw[seg == 1])

    b50 = np.asarray(nib.load(f"{FS}/{s}/mri/brain.image.b50.mgz").dataobj).astype(np.float32)
    wm = np.asarray(nib.load(f"{FS}/{s}/mri/wm.mgz").dataobj) > 0
    rib = np.asarray(nib.load(f"{FS}/{s}/mri/ribbon.mgz").dataobj) > 0
    tis = np.asarray(nib.load(f"{FS}/{s}/mri/tissue.mgz").dataobj) > 0
    B = dict(all=b50[tis], wm=b50[wm], gm=b50[rib & ~wm])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    hi_a = float(np.percentile(A["all"], 99.5))
    ov_a, ov_b = overlap(A, 0, hi_a, 160), overlap(B, 0, 120, 120)

    panel(axes[0], A, 0, hi_a, 160, "T2w intensity", "",
          "",
          labels=(("wm", WM, -hi_a * .03, "right"), ("gm", GM, hi_a * .03, "left")))
    axes[0].set_ylabel("density  (×10$^{-3}$)", fontsize=9.5, color=INK)

    # b50 is uint8: one bin per intensity level, or the curve combs
    panel(axes[1], B, 0, 120, 120, "b50 intensity", "",
          "",
          marks=((60, "GM target"), (110, "WM target")),
          labels=(("gm", GM, -3, "right"), ("wm", WM, -3, "right")))

    leg = axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ALL)],
                         labels=["whole brain"], frameon=False, fontsize=9,
                         loc="upper left", bbox_to_anchor=(-0.01, 1.02),
                         handlelength=1.0, handleheight=.8, handletextpad=.5)
    for t in leg.get_texts():
        t.set_color(MUTED)

    axes[1].set_ylabel("density  (×10$^{-3}$)", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.195, wspace=0.20)
    fig.savefig(f"{out}.png", dpi=dpi)
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  ({s})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subject", nargs="?", type=int, default=5)
    p.add_argument("--out", default=f"{OUT}/b50_hist")
    a = p.parse_args()
    run(a.subject, a.out)

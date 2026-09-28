#!/usr/bin/env python3
"""The depth axis: how the cortex is sampled, what the profiles show, and the white matter beneath.

The second half of the old FIG1_combined, split out and given the two explanatory panels it was
missing. FIG1_signature_ad carries the effect sizes; this carries the depth structure.

 a  ONE anatomical plane, five views of it: T2w, DWI and DBSI fiber; the equivolumetric shells
    cut through a three-hinge junction on that plane; and the tractogram on it, in voxel space.
    The orange box on the images and on the tractogram is the region the shells are drawn from.
 b  DBSI fiber fraction from cortex into superficial white matter
 c  the 3HG - crown fiber difference across that depth
 d  equivolumetric intracortical trajectories, six metrics, with the tested difference beneath
Nature double column, 183 mm."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patheffects as pe
import natstyle as ns
from fig_depth_combined import load_depth, ORDER, RES
import fig_fiberconv as fc
ns.nature()
OUT = RES; FW, FH = ns.W2, 190 * ns.MM
DPI = 1600                     # one raster resolution across the whole figure set
SUBJ_IMG = 6                         # best diffusion image quality of the six (see render_voxels.py)

# Row spacing. Panel a's lower cells end at 0.575; the two blocks below were hung well clear of
# it and of each other, leaving 10 mm and 18 mm of white where 6 mm and 10 mm read as separation.
# The bottom margin looks after itself - bbox_inches="tight" crops to the lowest drawn element.
R2_TOP, R2_H = 0.494, 0.152          # the two fiber plots
R3_TOP = 0.256                       # trajectories + strips


def img(path, ax, crop=None):
    im = Image.open(path)
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    a = np.asarray(bg)
    if crop:
        h, w = a.shape[:2]
        a = a[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
    ax.imshow(a, interpolation="lanczos"); ax.set_axis_off()
    return a.shape


def row_images(fig):
    """a: one plane, told as context then detail

    Five equal views in a row gave every view the same weight and no order, so the reader had no
    way in - there was nothing to say WHERE the detail views are. Now the slice is shown once,
    large, with the detail region boxed on it, and the six small views are all that same box: the
    three contrasts the analysis reads, the equivolumetric shells placed on it, the tractogram
    running through it, and the depth scale the shells are coloured by."""
    from matplotlib.collections import LineCollection
    z = np.load(f"{RES}/_voxels_S{SUBJ_IMG}.npz", allow_pickle=True)
    bx = np.load(f"{RES}/_lamina_S{SUBJ_IMG}_box.npz")
    ext = z["extent"]; bu, bv, bh = z["box"]
    bext = [bu - bh, bu + bh, bv - bh, bv + bh]

    # GY has to clear a title: the second row's captions were landing against the first row's
    # frames. 7 pt of type plus its pad is about 3.5 mm, which on this canvas is 0.019.
    #
    # Grid: three small views on top, two large ones underneath spanning the same width. The
    # earlier 3 x 2 grid had a sixth cell holding nothing but the depth colour key, which left a
    # 30 x 30 mm hole in the corner of the figure. The key now lives inside the shells view it
    # belongs to, and the two constructed views - the tractogram and the shells - take the space,
    # which is also where the detail actually needs the size.
    GX, GY, GAPC = 0.006, 0.026, 0.020
    X0, X1 = 0.028, 0.980
    # solve CX + GAPC + (3 SX + 2 GX) = X1 - X0 with the context square and as tall as the grid
    SX = (X1 - X0 - GAPC - 2.5 * GX - GY * FH / FW) / 5.5
    SY = SX * FW / FH
    BW = 1.5 * SX + 0.5 * GX                         # bottom pair: half the three-column width
    BH = BW * FW / FH
    HG = SY + GY + BH                                # grid height, which the context matches
    CX = HG * FH / FW
    TOP = 0.955
    fig.text(0.010, TOP + 0.010, "a", fontsize=8, fontweight="bold", va="bottom")

    def dot(ax, ms):
        """the 3HG the whole panel is built around, in every view of it

        Marked everywhere rather than only on the shells: it is the same landmark in all six, and
        without it the detail views are just a piece of cortex."""
        ax.plot([bu], [bv], marker="o", ms=ms, mfc=ns.C_3HG, mec="white", mew=0.5,
                linestyle="none", zorder=8, clip_on=False)

    def on_data(poly, A):
        """split a polyline into the runs of it that lie on pixels the image actually has

        The tractogram is defined over the diffusion mask, but the panels show that mask eroded by
        about a millimetre to keep the DBSI edge speckle out. Drawn unclipped, streamlines ran off
        onto blank canvas past the edge of every view - the same thing the ribbon contours did."""
        ny, nx = A.shape
        ix = ((poly[:, 0] - bext[0]) / (bext[1] - bext[0]) * nx).astype(int)
        iy = ((poly[:, 1] - bext[2]) / (bext[3] - bext[2]) * ny).astype(int)
        good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        good[good] &= np.isfinite(A[iy[good], ix[good]])
        if good.all():
            return [poly]
        idx = np.where(good)[0]
        if not len(idx):
            return []
        return [poly[g] for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(g) > 1]

    def ribbon(ax, A, label=False):
        """the white and pial boundaries on a detail view, clipped to where the image has data

        Drawn on every detail view, not just one: the point of the row is that these are all the
        same tissue, and the ribbon outline is what lets a reader carry the cortex from one
        contrast to the next - and see, in the tractography view, that the tracts stop at the
        white boundary. The surfaces cover the whole hemisphere but the diffusion mask does not,
        so an unclipped contour runs on over blank canvas."""
        ny, nx = A.shape

        def finite_at(P):
            ix = ((P[:, 0] - bext[0]) / (bext[1] - bext[0]) * nx).astype(int)
            iy = ((P[:, 1] - bext[2]) / (bext[3] - bext[2]) * ny).astype(int)
            inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            out = np.zeros(len(P), bool)
            out[inb] = np.isfinite(A[iy[inb], ix[inb]])
            return out

        for arr, col in ((bx["cw"], ns.C_CROWN), (bx["cp"], ns.C_3HG)):
            # both endpoints AND the midpoint on imaged pixels. Midpoint-only let a segment whose
            # far end sat past the eroded mask draw in full, so contours overhung the image edge
            # by up to half a segment - one to two millimetres at this segment length.
            good = finite_at(arr[:, 0]) & finite_at(arr.mean(1)) & finite_at(arr[:, 1])
            ax.add_collection(LineCollection(list(arr[good]), colors=col, linewidths=0.7,
                                             zorder=6))
        if label:
            for yy, lab, col in ((0.140, "White", ns.C_CROWN), (0.040, "Pial", ns.C_3HG)):
                ax.text(0.965, yy, lab, transform=ax.transAxes, fontsize=7, color=col,
                        fontweight="bold", va="bottom", ha="right", zorder=9,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

    def frame(ax, col=ns.INK, lw=0.5):
        ax.set_axis_on()
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(col); sp.set_linewidth(lw)

    # ---- context ----
    axc = fig.add_axes([X0, TOP - HG, CX, HG])
    axc.imshow(z["T2w"], cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext,
               interpolation="lanczos")
    axc.plot([bext[0], bext[1], bext[1], bext[0], bext[0]],
             [bext[2], bext[2], bext[3], bext[3], bext[2]], color=ns.C_3HG, lw=0.9, zorder=5)
    axc.set_xlim(ext[0], ext[1]); axc.set_ylim(ext[2], ext[3])
    dot(axc, 3.2)
    frame(axc)
    axc.plot([0.958], [0.058], marker="o", ms=3.2, mfc=ns.C_3HG, mec="white", mew=0.5,
             linestyle="none", transform=axc.transAxes, clip_on=False, zorder=9)
    axc.text(0.930, 0.058, "3HG landmark", transform=axc.transAxes, fontsize=6.5, va="center",
             ha="right", color=ns.C_3HG, fontweight="bold")

    GX0 = X0 + CX + GAPC
    # Top row: the three contrasts the analysis reads, at the size a contrast needs. Bottom row:
    # the two constructed views, at half the row instead of a third of it.
    cells = [("d_T2w", "T2w", "gray"), ("d_DWI", "DWI", "gray"), ("d_DBSI", "DBSI fiber", "gray"),
             ("tracts", "Tractography", None), ("lamina", "Equivolumetric shells", None)]
    for n, (key, title, cmap) in enumerate(cells):
        if n < 3:
            cx, cy, cw, ch = GX0 + n * (SX + GX), TOP - SY, SX, SY
        else:
            cx, cy, cw, ch = GX0 + (n - 3) * (BW + GX), TOP - SY - GY - BH, BW, BH
        ax = fig.add_axes([cx, cy, cw, ch])
        if key == "lamina":
            img(f"{RES}/_lamina_S{SUBJ_IMG}.png", ax)
            frame(ax, ns.C_3HG, 0.9)
            # the depth scale, inside the view it describes. The upper-left of this render is
            # blank canvas; the backing patch is there in case a specimen's ribbons reach it.
            from matplotlib.patches import Rectangle
            from render_lamina import LAYER_CMAP, NSH
            ax.add_patch(Rectangle((0.025, 0.845), 0.46, 0.135, transform=ax.transAxes,
                                   facecolor="white", edgecolor="none", alpha=.85, zorder=6))
            kx = ax.inset_axes([0.045, 0.905, 0.42, 0.038])
            # a child axes has zorder 5 by default, which put the colour bar UNDER the alpha-0.85
            # backing patch - it rendered as a washed-out pastel strip that no longer matched the
            # shells it is the key for
            kx.set_zorder(7)
            kx.imshow(np.arange(NSH)[None, :], cmap=LAYER_CMAP(), aspect="auto")
            kx.set_xticks([]); kx.set_yticks([])
            for sp in kx.spines.values():
                sp.set_visible(False)
            ax.text(0.045, 0.895, "White", fontsize=6.5, ha="left", va="top",
                    transform=ax.transAxes, color=ns.C_CROWN, fontweight="bold", zorder=8)
            ax.text(0.465, 0.895, "Pial", fontsize=6.5, ha="right", va="top",
                    transform=ax.transAxes, color=ns.C_3HG, fontweight="bold", zorder=8)
        elif key == "tracts":
            # the detail DWI, not the context view zoomed: same grid as the other small views
            ax.imshow(np.ma.masked_invalid(z["d_DWI"]), cmap="gray", vmin=0, vmax=1.25,
                      origin="lower", extent=bext, interpolation="lanczos")
            m = z["inbox"].astype(bool)
            raw = z["segs"][m]; rawc = np.clip(z["cols"][m], 0, 1)
            segs, cols = [], []
            for poly, c in zip(raw, rawc):
                for piece in on_data(np.asarray(poly), z["d_DWI"]):
                    segs.append(piece); cols.append(c)
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.28, alpha=0.85))
            ax.set_xlim(bext[0], bext[1]); ax.set_ylim(bext[2], bext[3])
            ribbon(ax, z["d_DWI"])
            dot(ax, 4.5)
            frame(ax, ns.C_3HG, 0.9)
            print(f"  panel a tractography: {len(raw)} segments in the box, "
                  f"{len(segs)} pieces after clipping to imaged tissue", flush=True)
        else:
            A = z[key]
            ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0, vmax=1, origin="lower",
                      extent=bext, interpolation="lanczos")
            ax.set_xlim(bext[0], bext[1]); ax.set_ylim(bext[2], bext[3])
            ribbon(ax, A, label=(key == "d_T2w"))
            dot(ax, 4.5)
            frame(ax)
        ax.set_title(title, fontsize=6.8, pad=2)


def row_fiber(fig):
    """d,e: the sub-white-matter fiber result, on the same depth axis as a"""
    z = np.load(f"{OUT}/_depthAB_cache.npz"); Hs, Cs = z["Hs"], z["Cs"]
    D = fc.DEPTHS
    p = stats.wilcoxon(Hs[:, fc.SUBWM].mean(1) - Cs[:, fc.SUBWM].mean(1)).pvalue
    for tag, x0, w in (("b", 0.075, 0.385), ("c", 0.585, 0.385)):
        ax = fig.add_axes([x0, R2_TOP - R2_H, w, R2_H])
        ax.axvspan(0, D.max(), color="#f2f0ec", zorder=0)
        ax.axvline(0, color=ns.INK, lw=.5)
        if tag == "b":
            for M, col, lab in [(Cs, ns.C_CROWN, "Crown"), (Hs, ns.C_3HG, "3HG")]:
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(6)
                ax.plot(D, mu, color=col, lw=1.3, label=lab, solid_capstyle="round", zorder=3)
                ax.fill_between(D, mu - se, mu + se, color=col, alpha=.16, lw=0, zorder=2)
            ax.set_ylabel("DBSI fiber fraction", fontsize=7, labelpad=2)
            ax.legend(loc="lower right", handlelength=.9, fontsize=6.5, borderpad=.2,
                      labelspacing=.3, handletextpad=.4)
        else:
            diff = Hs - Cs; mu = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(6)
            ax.axhline(0, color=ns.INK, lw=.5, zorder=1)
            ax.fill_between(D, 0, mu, where=(mu > 0), color=ns.C_3HG, alpha=.13, zorder=1)
            ax.fill_between(D, mu - se, mu + se, color=ns.C_3HG, alpha=.22, lw=0, zorder=2)
            ax.plot(D, mu, color=ns.C_3HG, lw=1.3, solid_capstyle="round", zorder=3)
            ax.set_ylabel("Fiber-fraction difference", fontsize=7, labelpad=2)
            ax.text(.96, .08, f"$P$ = {p:.3f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6.5)
        y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + .16 * (y1 - y0))
        ax.text(.14, .965, "Cortex", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
        ax.text(.64, .965, "White matter", transform=ax.transAxes, ha="center", va="top",
                fontsize=6.5)
        ax.set_xlabel("Distance from white surface (mm)", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        fig.text(x0 - 0.058, R2_TOP + 0.012, tag, fontsize=8, fontweight="bold", va="bottom")
        fig.text(x0 + w / 2, R2_TOP + 0.012,
                 "Fiber fraction into the white matter" if tag == "b"
                 else "3HG minus crown across that depth",
                 fontsize=7.5, ha="center", va="bottom")


def row_depth(fig):
    """f: the trajectories, each with the quantity depth_stats.py actually tests beneath it"""
    rho, Yh, Yc, dWP = load_depth()
    TOP, STRIP, GAP, BOT = 0.130, 0.048, 0.011, 0.067
    for i, (m, lab) in enumerate(ORDER):
        x0 = 0.062 + i * 0.157
        ax = fig.add_axes([x0, BOT + STRIP + GAP, 0.124, TOP])
        for Y, col in [(Yc[m], ns.C_CROWN), (Yh[m], ns.C_3HG)]:
            mu = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(6)
            ax.plot(rho, mu, color=col, lw=1.0, solid_capstyle="round", zorder=3)
            ax.fill_between(rho, mu - se, mu + se, color=col, alpha=.16, lw=0, zorder=2)
        ax.fill_between(rho, Yc[m].mean(0), Yh[m].mean(0), color=ns.MUTE, alpha=.22, lw=0, zorder=1)
        ax.set_title(lab, fontsize=7, pad=2)
        if m == "MD":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.yaxis.get_offset_text().set(size=5.5, color=ns.INK)
        ax.set_xticks([0, .5, 1]); ax.set_xticklabels([]); ax.tick_params(labelsize=6)
        if i == 0:
            fig.text(0.012, R3_TOP + 0.024, "d", fontsize=8, fontweight="bold", va="bottom")
            fig.text(0.53, R3_TOP + 0.024, "Intracortical depth trajectories", fontsize=7.5,
                     ha="center", va="bottom")
        sx = fig.add_axes([x0, BOT, 0.124, STRIP])
        Dm = dWP[m]; mu = np.nanmean(Dm, 0); se = np.nanstd(Dm, 0, ddof=1) / np.sqrt(Dm.shape[0])
        sx.axhline(0, color=ns.INK, lw=.4, ls=(0, (2.5, 2)), zorder=1)
        sx.fill_between(rho, 0, mu, where=(mu > 0), color=ns.C_3HG, alpha=.30, lw=0,
                        interpolate=True, zorder=2)
        sx.fill_between(rho, 0, mu, where=(mu < 0), color=ns.C_CROWN, alpha=.30, lw=0,
                        interpolate=True, zorder=2)
        sx.fill_between(rho, mu - se, mu + se, color=ns.MUTE, alpha=.20, lw=0, zorder=2)
        sx.plot(rho, mu, color=ns.INK, lw=.9, solid_capstyle="round", zorder=3)
        sx.set_ylim(-0.42, 0.42); sx.set_yticks([-0.3, 0, 0.3])
        sx.set_xticks([0, .5, 1]); sx.set_xticklabels(["white", "0.5", "pial"], fontsize=6)
        sx.tick_params(labelsize=5.5)
        if i == 0:
            sx.set_ylabel("3HG − crown\n(SD)", fontsize=6.2, labelpad=2)
    # one axis label for the row, centred under the six columns, instead of six copies
    fig.text(0.062 + (5 * 0.157 + 0.124) / 2, BOT - 0.038, "Cortical depth", fontsize=7,
             ha="center", va="top")


def main():
    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    row_images(fig); row_fiber(fig); row_depth(fig)
    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{OUT}/FIG2_depth_wm.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    w, h = Image.open(f"{OUT}/FIG2_depth_wm.png").size
    print(f"wrote FIG2_depth_wm.png  {w/DPI*25.4:.0f} x {h/DPI*25.4:.0f} mm")


if __name__ == "__main__":
    main()

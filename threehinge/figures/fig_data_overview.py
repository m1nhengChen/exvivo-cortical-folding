#!/usr/bin/env python3
"""Data and preprocessing overview: why the raw contrast fails, what the tent mapping does to it,
and the six reconstructions it yields.

 a  the pipeline on three specimens (rows, unlabelled - the point is the transformation
    along each row, not which hemisphere it ran on) - original T2w, tent mapping, b50, and the segmentation the
    surfaces are placed against
 b  what that does to the intensity distribution: T2w has white matter BELOW grey and the two
    modes overlap, b50 puts white at FreeSurfer's 110 target and grey at 60, in the T1-like
    ordering every downstream binary assumes
 c  the resulting cohort - two para-sagittal sections and the pial surface for each hemisphere

The three parts are REDRAWN into one canvas rather than composited from their finished PNGs. Each
source script sets its own type - 25 pt bold column heads in the pipeline, 9.5 pt axis labels in
the histograms, 7.5 pt in the cohort - and pasting the images together would carry three different
typographic systems into one figure. Importing the panel functions and driving them from here
gives one type scale, one font, and one set of panel letters.

    python3 fig_data_overview.py           # tall: a, b and c stacked      -> FIG_data_overview
    python3 fig_data_overview.py --wide    # a beside b, c underneath      -> ..._wide
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import natstyle as ns
import vis_tent_pipeline as vt
import vis_b50_hist as vh
import fig_cohort as fc

DPI = 1600                     # one raster resolution across the whole figure set
OUT = (_env.ROOT + "/surf/three_hinge/results")
FONTS = ["Arial", "Liberation Sans", "Nimbus Sans", "Helvetica", "DejaVu Sans"]
PIPE_SUBJ = [2, 4, 5]
# One type scale for the whole plate. At 182 x 247 mm this figure is a full page, so the 5-7 pt
# band that suits an 89 mm panel reads as undersized here; these are the sizes in points.
FS_HEAD, FS_ROW, FS_AX, FS_TICK, FS_LET, FS_SUBJ = 8.5, 8.5, 8.2, 7.2, 9.5, 8.5
COLS = ["Original T2w", "Tent mapping", "Segmentation", "B50"]


def shrink_text(ax, k):
    """The imported panel functions hardcode 8-9.5 pt for a 178 mm figure of their own. Scale
    every text they left behind, plus the ticks, onto this plate's 6-7 pt scale."""
    for t in ax.texts + [ax.xaxis.label, ax.yaxis.label]:
        t.set_fontsize(t.get_fontsize() * k)
    ax.tick_params(labelsize=FS_TICK, width=.6, length=2.4)


def pipeline_rows():
    rows = []
    for sid in PIPE_SUBJ:
        t2, tent, b50, wm, gm = vt.load(sid)
        x = vt.best_x(wm, gm)
        # column 2 carries the segmentation overlay, drawn on the same b50 slice as column 3
        rows.append((sid, [vt.sag(t2, x), vt.sag(tent, x), vt.sag(b50, x), vt.sag(b50, x)],
                     (vt.sag(gm, x).astype(bool), vt.sag(wm, x).astype(bool))))
    hw = (max(im.shape[0] for _, ims, _ in rows for im in ims),
          max(im.shape[1] for _, ims, _ in rows for im in ims))
    rows = [(sid, [vt.pad_to(im, hw) for im in ims],
             tuple(vt.pad_to(m, hw) for m in seg)) for sid, ims, seg in rows]
    # pad_to centres every slice on a canvas sized for the largest, which leaves a black border
    # around all of them. Crop back to the union of the content - a crop, not a resize, so the
    # 0.5 mm/pixel scale and the true relative brain sizes survive.
    acc = np.zeros(hw, bool)
    for _, ims, _ in rows:
        for im in ims:
            acc |= im > 0
    r0, r1 = np.where(acc.any(1))[0][[0, -1]]
    c0, c1 = np.where(acc.any(0))[0][[0, -1]]
    pad = 4
    r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
    r1, c1 = min(r1 + pad, hw[0] - 1), min(c1 + pad, hw[1] - 1)
    cut = lambda a: a[r0:r1 + 1, c0:c1 + 1]
    # vis_tent_pipeline.sag() applies a fixed voxel-grid flip, which for these specimens puts
    # ANTERIOR ON THE LEFT. Panel c derives its axes from the anatomy and puts anterior on the
    # right, as do the surface renders. Mirror panel a so one figure has one convention. Measured,
    # not assumed: a marker ball at the frontal pole lands in columns 31-100 of 337 before this.
    cut = lambda a, _c=cut: _c(a)[:, ::-1]
    rows = [(sid, [cut(im) for im in ims], tuple(cut(m) for m in seg))
            for sid, ims, seg in rows]
    return rows, (r1 - r0 + 1, c1 - c0 + 1)


def hist_data(sid):
    s = f"Subject{sid}"
    seg = np.asarray(nib.load(f"{vh.ROOT}/{s}/{s}_Mask/T2w_GM_WM_mask.nii.gz").dataobj)
    raw = np.asarray(nib.load(f"{vh.CSF}/S{sid}.t2w_n4.nii.gz").dataobj).astype(np.float32)
    A = dict(all=raw[seg > 0], wm=raw[seg == 2], gm=raw[seg == 1])
    b50 = np.asarray(nib.load(f"{vh.FS}/{s}/mri/brain.image.b50.mgz").dataobj).astype(np.float32)
    wm = np.asarray(nib.load(f"{vh.FS}/{s}/mri/wm.mgz").dataobj) > 0
    rib = np.asarray(nib.load(f"{vh.FS}/{s}/mri/ribbon.mgz").dataobj) > 0
    tis = np.asarray(nib.load(f"{vh.FS}/{s}/mri/tissue.mgz").dataobj) > 0
    B = dict(all=b50[tis], wm=b50[wm], gm=b50[rib & ~wm])
    return A, B


def draw_pipeline(fig, x0, ytop, Wmm, FW, FH, rows, hw, gap=0.7):
    """Four columns x three specimens, top edge at `ytop`. Returns its height in mm."""
    fx, fy = lambda v: v / FW, lambda v: v / FH
    GX, HEAD = 1.2, 5.0
    w = (Wmm - 3 * GX) / 4
    h = w * hw[0] / hw[1]
    for c, name in enumerate(COLS):
        fig.text(fx(x0 + c * (w + GX) + w / 2), fy(ytop - HEAD + 1.0), name,
                 fontsize=FS_HEAD, ha="center", va="bottom", color=ns.INK)
    for r, (sid, ims, seg) in enumerate(rows):
        yr = ytop - HEAD - (r + 1) * h - r * gap
        for c in range(4):
            ax = fig.add_axes([fx(x0 + c * (w + GX)), fy(yr), fx(w), fy(h)])
            vt.panel(ax, ims[c], None, seg=seg if c == 2 else None)
    return HEAD + 3 * h + 2 * gap


XLAB_H = 5.2          # room under the last histogram for its x-axis label and ticks


def draw_hists(fig, x0, ytop, Wmm, FW, FH, A, B, stacked, thumbs=None,
               hpanel=30.0, gap=5.0):
    """Both intensity histograms. `stacked` puts them one above the other, which is what the
    wide layout needs to keep them readable in a narrow column."""
    fx, fy = lambda v: v / FW, lambda v: v / FH
    hiA = float(np.percentile(A["all"], 99.5))
    # Stacked, the two share one x-axis label. Both axes carry the same quantity in arbitrary
    # units, so repeating "intensity" twice adds nothing - but the panels then need identifying.
    # A thumbnail of the image each distribution came from does that without a caption word, and
    # it ties the histogram to the row of panel a it belongs to.
    spec = [(A, 0, hiA, 160, "T2w intensity", (),
             (("wm", vh.WM, -hiA * .03, "right"), ("gm", vh.GM, hiA * .03, "left"))),
            # no target markers: the two modes already sit at 60 and 110, so the dashed
            # lines only restate what the curves show
            (B, 0, 120, 120, "B50 intensity", (),
             (("gm", vh.GM, -3, "right"), ("wm", vh.WM, -3, "right")))]
    YL = 9.0                                   # room for the rotated y-axis label
    made = []
    for i, (data, lo, hi, bins, xlab, marks, labs) in enumerate(spec):
        if stacked:
            ax = fig.add_axes([fx(x0 + YL), fy(ytop - (i + 1) * hpanel - i * gap),
                               fx(Wmm - YL), fy(hpanel)])
        else:
            w = (Wmm - 2 * YL - gap) / 2
            ax = fig.add_axes([fx(x0 + YL + i * (w + YL + gap)), fy(ytop - hpanel),
                               fx(w), fy(hpanel)])
        made.append(ax)
        vh.panel(ax, data, lo, hi, bins, "" if (stacked and i == 0) else xlab,
                 "", "", marks=marks, labels=labs)
        ax.set_ylabel("Density  (×10$^{-3}$)", fontsize=FS_AX, color=ns.INK)
        # tick labels are 1-3 digits wide in the two panels, so the default label placement
        # lands at two different x positions; pin both to the same axes fraction
        ax.yaxis.set_label_coords(-0.085, 0.5)
        if stacked:
            if thumbs is not None:
                # sized in axes fractions from the millimetres, so the thumbnail keeps the
                # image's own aspect instead of being stretched to the fraction box
                th = thumbs[i]
                wf = 0.215
                hf = wf * ((Wmm - YL) / hpanel) * (th.shape[0] / th.shape[1])
                inset = ax.inset_axes([0.022, 0.965 - hf, wf, hf])
                vt.panel(inset, th, None)
            if i == 1:
                ax.set_xlabel("Image intensity", fontsize=FS_AX, color=ns.INK)
        shrink_text(ax, FS_AX / 9.5)
    n = 2 if stacked else 1
    # inside the upper axes, not on the panel's top edge - that is where the title now sits
    axk = made[0]
    axk.add_patch(plt.Rectangle((0.545, 0.885), 0.030, 0.055, color=vh.ALL,
                                transform=axk.transAxes, clip_on=False, zorder=4))
    axk.text(0.588, 0.912, "Whole brain", transform=axk.transAxes, fontsize=FS_TICK,
             color=vh.MUTED, va="center", ha="left")
    return n * hpanel + (n - 1) * gap + XLAB_H


def build(wide):
    ns.nature()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONTS})
    fc.SUBJ_FS = FS_SUBJ
    # panel c carries the gyral skeleton as well as the landmarks: a 3HG is a degree->=3 vertex
    # of that network, so showing the network makes the definition legible rather than asserted
    fc.SHOW_SKEL = os.environ.get("SKEL", "1") == "1"
    if wide:                      # the plate is dense already; let the cohort block run tight
        fc.SP.update(GX=3.0, TITLE=3.6, VGAP=0.9, ROWGAP=3.2)
    rows, hw = pipeline_rows()
    A, B = hist_data(5)

    MARG, LET = 2.0, 4.8 if not wide else 4.0
    FW = ns.W2 / ns.MM
    Wtot = FW - 2 * MARG
    hc = fc.block_geom(Wtot)["total"]

    if wide:
        SPLIT = 4.0
        Wa = Wtot * 0.615
        Wb = Wtot - Wa - SPLIT
        ha = 5.0 + 3 * (((Wa - 3 * 1.2) / 4) * hw[0] / hw[1]) + 2 * 0.7
        hb = 2 * 30.0 + 5.0 + XLAB_H
        htop = max(ha, hb)
        FH = LET + htop + 3.4 + LET + hc + 1.4
    else:
        FH = LET + (5.0 + 3 * (((Wtot - 3 * 1.2) / 4) * hw[0] / hw[1]) + 2 * 0.7) \
             + 3.6 + LET + 30.0 + 4.6 + LET + hc + 2.0

    fig = plt.figure(figsize=(FW * ns.MM, FH * ns.MM), facecolor="white")
    fx, fy = lambda v: v / FW, lambda v: v / FH

    def letter(ch, x, y, title=None, w=None, tx0=0.0):
        """panel letter, and its descriptive title on the same baseline

        The other plates caption every panel with a sentence saying what it shows; this one had
        only column headers, so a reader met three blocks of images with nothing naming them."""
        fig.text(fx(x), fy(y), ch, fontsize=FS_LET, fontweight="bold", va="top", color=ns.INK)
        if title:
            fig.text(fx(x + tx0 + (w - tx0) / 2), fy(y), title, fontsize=FS_HEAD + 0.6,
                     va="top", ha="center", color=ns.INK)

    y = FH - 2.0
    if wide:
        letter("a", MARG, y, "From fixed tissue to a surface reconstruction", Wa)
        letter("b", MARG + Wa + SPLIT, y, "Tent mapping restores the T1-like ordering", Wb,
               tx0=6.0)
        draw_pipeline(fig, MARG, y - LET, Wa, FW, FH, rows, hw)
        # the thumbnails are the T2w and b50 columns of the Subject5 row of panel a
        th = next(ims for sid, ims, _ in rows if sid == 5)
        draw_hists(fig, MARG + Wa + SPLIT, y - LET, Wb, FW, FH, A, B, stacked=True,
                   thumbs=[th[0], th[3]])
        y = y - LET - htop - 3.4
    else:
        letter("a", MARG, y)
        ha = draw_pipeline(fig, MARG, y - LET, Wtot, FW, FH, rows, hw)
        y = y - LET - ha - 3.6
        letter("b", MARG, y)
        draw_hists(fig, MARG, y - LET, Wtot, FW, FH, A, B, stacked=False)
        y = y - LET - 30.0 - 4.6
    letter("c", MARG, y, "The six reconstructed hemispheres", Wtot)
    fc.draw_block(fig, MARG, y - LET, Wtot, FW, FH)

    name = "FIG_data_overview_wide" if wide else "FIG_data_overview"
    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    w, h = Image.open(f"{OUT}/{name}.png").size
    print(f"wrote {name}.png  {w / DPI * 25.4:.0f} x {h / DPI * 25.4:.0f} mm")


if __name__ == "__main__":
    build("--wide" in sys.argv)

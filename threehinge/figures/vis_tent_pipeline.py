#!/usr/bin/env python3
"""The tent-mapping pipeline figure, one row per subject.

Four columns, left to right, are exactly the volumes mris_place_surface is handed:

    original T2w (N4)    csf/S{i}.t2w_n4.nii.gz, resampled onto the FreeSurfer grid
    tent mapping         fs/Subject{i}/mri/brain.image.mgz      (wm_target.py --blend 0)
    b50                  fs/Subject{i}/mri/brain.image.b50.mgz  (tent x0.5 + label x0.5)
    segmentation         ribbon (GM) / white (WM) over b50

Sagittal slices; --preview renders all six subjects so a row selection can be made.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, argparse, numpy as np, nibabel as nib
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
from scipy import ndimage as ndi

ROOT = (_env.ROOT + "/28211843")
CSF = (_env.ROOT + "/surf/csf")
FS = (_env.ROOT + "/surf/fs")
OUT = (_env.ROOT + "/surf/results")


def load(sid):
    """All four panels on the FreeSurfer 0.5 mm grid of Subject{sid}."""
    s = f"Subject{sid}"
    wmi = nib.load(f"{FS}/{s}/mri/wm.mgz")
    aff, shape = wmi.affine, wmi.shape

    raw = nib.load(f"{CSF}/S{sid}.t2w_n4.nii.gz")
    ijk = np.indices(shape).reshape(3, -1)
    src = nib.affines.apply_affine(np.linalg.inv(raw.affine),
                                   nib.affines.apply_affine(aff, ijk.T)).T
    t2 = ndi.map_coordinates(np.asarray(raw.dataobj).astype(np.float32), src,
                             order=1, mode="constant", cval=0).reshape(shape)

    tent = np.asarray(nib.load(f"{FS}/{s}/mri/brain.image.mgz").dataobj).astype(np.float32)
    b50 = np.asarray(nib.load(f"{FS}/{s}/mri/brain.image.b50.mgz").dataobj).astype(np.float32)
    wm = np.asarray(wmi.dataobj) > 0
    gm = (np.asarray(nib.load(f"{FS}/{s}/mri/ribbon.mgz").dataobj) > 0) & ~wm
    return t2, tent, b50, wm, gm


def sag(vol, x):
    """Sagittal slice: mirrored left-right, then turned 90 deg clockwise."""
    return np.rot90(np.fliplr(np.rot90(vol[x])), -1)


def best_x(wm, gm):
    """Sagittal slice with the most cortex -- the one that shows the folding."""
    return int(np.argmax((wm | gm).sum((1, 2))[10:-10]) + 10)


def panel(ax, im, title, seg=None, ov=None):
    # lanczos on the greyscale (the 0.5 mm grid is upsampled well past 1:1 at print dpi);
    # nearest on the labels, so the overlay does not invent intermediate label colours
    ax.imshow(im, cmap="gray", interpolation="lanczos",
              vmin=0, vmax=np.percentile(im[im > 0], 99.5) if (im > 0).any() else 1)
    if seg is not None:
        gm, wm = seg
        rgba = np.zeros(im.shape + (4,))
        # same two tissue hues the histogram panel uses, at the plate's chroma. These were a
        # near-primary green and blue, which read as a different figure next to the coral/slate.
        rgba[gm] = [0.25, 0.79, 0.64, 0.68]
        rgba[wm] = [0.50, 0.65, 0.91, 0.68]
        ax.imshow(rgba, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=25, fontweight="bold")


WIDTH, LEFT, RIGHT, WSPACE = 14.0, 0.03, 0.995, 0.008


def pad_to(im, hw):
    """Centre `im` on a common canvas.  Padding, not resizing -- every panel keeps the
    same 0.5 mm/pixel scale, so brain sizes stay comparable across rows."""
    h, w = hw
    t, l = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    out = np.zeros(hw, im.dtype)
    out[t:t + im.shape[0], l:l + im.shape[1]] = im
    return out


def figure(subjects, out, dx=0, dpi=400):
    n = len(subjects)
    rows = []
    for sid in subjects:
        t2, tent, b50, wm, gm = load(sid)
        x = np.clip(best_x(wm, gm) + dx, 0, wm.shape[0] - 1)
        rows.append((sid, [sag(t2, x), sag(tent, x), sag(b50, x), sag(b50, x)],
                     (sag(gm, x).astype(bool), sag(wm, x).astype(bool))))

    # One canvas for every panel: pad each slice to the largest extent, then size the axes
    # box to that aspect.  imshow keeps square pixels and centres what it is given, so a
    # box/image mismatch would show up as letterboxing *inside* the axes -- a column gap
    # wspace cannot close.  With a common canvas every panel fills its box identically.
    hw = (max(im.shape[0] for _, ims, _ in rows for im in ims),
          max(im.shape[1] for _, ims, _ in rows for im in ims))
    rows = [(sid, [pad_to(im, hw) for im in ims], tuple(pad_to(m, hw) for m in seg))
            for sid, ims, seg in rows]

    axw = WIDTH * (RIGHT - LEFT) / (4 + 3 * WSPACE)
    rowh = axw / (hw[1] / hw[0])

    fig, axes = plt.subplots(n, 4, figsize=(WIDTH, rowh * n + 0.68), squeeze=False)
    cols = ["original  T2w", "tent mapping", "b50", "segmentation"]
    for r, (sid, ims, seg) in enumerate(rows):
        for c in range(4):
            panel(axes[r][c], ims[c], cols[c] if r == 0 else None,
                  seg=seg if c == 3 else None)
        axes[r][0].text(-0.045, 0.5, f"Subject{sid}", transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=25, fontweight="bold")
    top = 1 - 0.52 / (rowh * n + 0.68)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=top, bottom=0.005,
                        wspace=WSPACE, hspace=0.03)
    fig.savefig(f"{out}.png", dpi=dpi)
    fig.savefig(f"{out}.pdf")          # vector text, images embedded at `dpi`
    plt.close(fig)
    print(f"wrote {out}.png ({dpi} dpi) / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subjects", nargs="*", type=int, default=[1, 2, 4])
    p.add_argument("--out", default=f"{OUT}/tent_pipeline_newFLASH")
    p.add_argument("--dx", type=int, default=0, help="offset from the max-cortex sagittal slice")
    p.add_argument("--dpi", type=int, default=400)
    a = p.parse_args()
    figure(a.subjects, a.out, a.dx, a.dpi)

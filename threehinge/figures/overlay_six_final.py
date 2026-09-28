#!/usr/bin/env python3
"""overlay_six.png - white/pial contours over brain.image sections, six hemispheres x three
orthogonal planes (overlay_mixed.png style: grayscale on black, cyan white, yellow pial).
Slices hand-picked from overlay_six_sheet.png: mid-volume cuts where the section is large,
the anatomy legible, and the contours free of tangential grazing."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np, nibabel as nib, pyvista as pv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

FS = (_env.ROOT + "/surf/fs")
OUT = (_env.ROOT + "/surf/results")
C_WHITE, C_PIAL = "#00e5ff", "#ffd400"
pv.global_theme.allow_empty_mesh = True

PICK = {1: [(0, 79), (1, 110), (2, 161)],
        2: [(0, 79), (1, 110), (2, 163)],
        3: [(0, 74), (1, 106), (2, 177)],
        4: [(0, 80), (1, 118), (2, 168)],
        5: [(0, 69), (1, 125), (2, 164)],
        6: [(0, 69), (1, 107), (2, 159)]}


def load(sid):
    """Background = the ORIGINAL N4 T2w, resampled onto the FreeSurfer grid the surfaces
    live in (brain.image.mgz is the tent-mapped placement image - WM remapped bright -
    and reads as b50, not anatomy). Masked to the brain so the b50-era background stays black."""
    from scipy import ndimage as ndi
    img = nib.load(f"{FS}/Subject{sid}/mri/brain.image.mgz")
    aff, shape = img.affine, img.shape
    raw = nib.load(f"{_env.ROOT}/surf/csf/S{sid}.t2w_n4.nii.gz")
    ijk = np.indices(shape).reshape(3, -1)
    src = nib.affines.apply_affine(np.linalg.inv(raw.affine),
                                   nib.affines.apply_affine(aff, ijk.T)).T
    vol = ndi.map_coordinates(np.asarray(raw.dataobj).astype(np.float32), src,
                              order=1, mode="constant", cval=0).reshape(shape)
    vol[img.get_fdata() <= 0] = 0
    inv = np.linalg.inv(img.header.get_vox2ras_tkr())
    surf = {}
    for name in ("white", "pial"):
        V, F = nib.freesurfer.read_geometry(f"{FS}/Subject{sid}/surf/rh.{name}")
        Vv = nib.affines.apply_affine(inv, V)
        surf[name] = pv.PolyData(Vv, np.c_[np.full(len(F), 3), F].ravel())
    return vol, surf


def segments(mesh, axis, idx):
    n = np.eye(3)[axis]
    sl = mesh.slice(normal=n, origin=n * idx)
    if sl.n_points == 0:
        return np.empty((0, 2, 2))
    P = np.asarray(sl.points)[:, [a for a in range(3) if a != axis]]
    L = np.asarray(sl.lines).reshape(-1, 3)[:, 1:]
    return P[L][:, :, ::-1]          # (x, y) = (second, first in-plane axis): landscape


def draw(ax, vol, surf, axis, idx):
    sl = np.take(vol, idx, axis=axis)
    lo, hi = np.percentile(sl[sl > 0], [0.5, 99.9]); hi *= 1.06
    ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower",
              interpolation="bilinear")
    for name, col in (("pial", C_PIAL), ("white", C_WHITE)):
        seg = segments(surf[name], axis, idx)
        if len(seg):
            ax.add_collection(LineCollection(seg, colors=col, linewidths=0.5))
    r, c = np.where(sl > 0)
    ax.set_xlim(c.min() - 4, c.max() + 4); ax.set_ylim(r.min() - 4, r.max() + 4)
    ax.set_axis_off()


def main():
    fig, axes = plt.subplots(6, 3, figsize=(180 / 25.4, 292 / 25.4), facecolor="black",
                             gridspec_kw=dict(left=0.045, right=0.995, top=0.985,
                                              bottom=0.030, wspace=0.03, hspace=0.05))
    for r, sid in enumerate(sorted(PICK)):
        vol, surf = load(sid)
        for c, (axis, idx) in enumerate(PICK[sid]):
            draw(axes[r, c], vol, surf, axis, idx)
        p = axes[r, 0].get_position()
        fig.text(0.022, (p.y0 + p.y1) / 2, f"Specimen {sid}", color="white", fontsize=9,
                 rotation=90, ha="center", va="center")
        print(f"S{sid} done", flush=True)
    fig.legend(handles=[Line2D([], [], color=C_WHITE, lw=2, label="white surface"),
                        Line2D([], [], color=C_PIAL, lw=2, label="pial surface")],
               loc="lower center", ncol=2, frameon=False, fontsize=10,
               labelcolor="white", bbox_to_anchor=(0.5, 0.000))
    fig.savefig(f"{OUT}/overlay_six.png", dpi=1600, facecolor="black")
    fig.savefig(f"{OUT}/overlay_six.pdf", facecolor="black")
    w, h = Image.open(f"{OUT}/overlay_six.png").size
    print(f"wrote overlay_six.png  {w}x{h}px = {w/1600*25.4:.0f} x {h/1600*25.4:.0f} mm")


if __name__ == "__main__":
    main()

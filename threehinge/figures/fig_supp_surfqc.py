#!/usr/bin/env python3
"""Supplementary: surface-reconstruction QC.

 a  one representative section per hemisphere (N4 T2w background), white + pial overlaid;
    the zoom boxes of b/c are drawn on their parent sections.
 b  three difficult regions at full zoom: a run of deep sulci, a three-hinge junction, and
    the margin of the medial cut.
 c  RecoverSulci before/after on a typical stretch of collapsed sulci: the placement volume
    with the carved sulcal-CSF class highlighted, and the pial that follows it.

Background is the original N4 T2w resampled onto each specimen's FreeSurfer grid (the tkr
frame the surfaces live in); contours are exact plane-mesh intersections, not projections."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, nibabel as nib, pyvista as pv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy import ndimage as ndi
from PIL import Image
import natstyle as ns

FS = (_env.ROOT + "/surf/fs")
CSF = (_env.ROOT + "/surf/csf")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
IB = (_env.ROOT + "/surf/ibeat")
RES = (_env.ROOT + "/surf/three_hinge/results")
DPI = 1600
C_WHITE, C_PIAL = "#00e5ff", "#ffd400"
pv.global_theme.allow_empty_mesh = True

SLICE = {1: 79, 2: 79, 3: 74, 4: 80, 5: 69, 6: 69}         # axis-0 cut per specimen
# zoom windows: (specimen, (row_center, col_center), half_size, label)
ZOOMS = [(3, (130, 157), 26, "deep sulci"),
         (5, (206, 197), 24, "3HG junction"),
         (2, (63, 157), 26, "medial cut margin")]
HINGE_V = 72270                                             # the marked junction, S5
REC = (5, (160, 240), 19)                                   # RecoverSulci window, S5


def load(sid):
    img = nib.load(f"{FS}/Subject{sid}/mri/brain.image.mgz")
    raw = nib.load(f"{CSF}/S{sid}.t2w_n4.nii.gz")
    ijk = np.indices(img.shape).reshape(3, -1)
    src = nib.affines.apply_affine(np.linalg.inv(raw.affine),
                                   nib.affines.apply_affine(img.affine, ijk.T)).T
    vol = ndi.map_coordinates(np.asarray(raw.dataobj).astype(np.float32), src,
                              order=1, mode="constant", cval=0).reshape(img.shape)
    vol[img.get_fdata() <= 0] = 0
    inv = np.linalg.inv(img.header.get_vox2ras_tkr())
    surf = {}
    for name in ("white", "pial"):
        V, F = nib.freesurfer.read_geometry(f"{FS}/Subject{sid}/surf/rh.{name}")
        Vv = nib.affines.apply_affine(inv, V)
        surf[name] = pv.PolyData(Vv, np.c_[np.full(len(F), 3), F].ravel())
    return vol, surf, inv


def segments(mesh, idx):
    sl = mesh.slice(normal=[1, 0, 0], origin=[idx, 0, 0])
    if sl.n_points == 0:
        return np.empty((0, 2, 2))
    P = np.asarray(sl.points)[:, [1, 2]]
    L = np.asarray(sl.lines).reshape(-1, 3)[:, 1:]
    return P[L][:, :, ::-1]                                 # (x=col, y=row)


def paint(ax, sl, surf, idx, lw=0.5, window=None):
    ax.set_facecolor("black")
    if window is not None:
        (rc, cc), h = window
        crop = sl[max(rc - h, 0):rc + h, max(cc - h, 0):cc + h]
        m = crop[crop > 0]
    else:
        m = sl[sl > 0]
    lo, hi = np.percentile(m, [0.5, 99.9]); hi *= 1.06
    ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="bilinear")
    for name, col in (("pial", C_PIAL), ("white", C_WHITE)):
        seg = segments(surf[name], idx)
        if len(seg):
            ax.add_collection(LineCollection(seg, colors=col, linewidths=lw))
    if window is None:
        # bbox from the eroded largest component: S5's medial cut face grazes the slice as a
        # ~8-voxel-wide bright strip CONNECTED to the main section, so plain labelling keeps
        # it; eroding first detaches and removes it, and the margin restores the true border
        big = ndi.binary_erosion(sl > 0, iterations=4)
        L, n = ndi.label(big)
        if n > 1:
            big = L == (np.bincount(L.ravel())[1:].argmax() + 1)
        r, c = np.where(big)
        ax.set_xlim(c.min() - 8, c.max() + 8); ax.set_ylim(r.min() - 8, r.max() + 8)
    else:
        (rc, cc), h = window
        ax.set_xlim(cc - h, cc + h); ax.set_ylim(rc - h, rc + h)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("black")


def main():
    ns.nature()
    FH_MM = 150
    fig = plt.figure(figsize=(ns.W2, FH_MM * ns.MM), facecolor="white")
    data = {sid: load(sid) for sid in SLICE}

    # ---- a: the six hemispheres, 2 x 3 ----
    AW, AH, GX, GY = 0.293, 0.265, 0.028, 0.030
    X0, Y0 = 0.028, 0.960
    axs = {}
    for k, sid in enumerate([1, 2, 3, 4, 5, 6]):
        r, c = divmod(k, 3)
        ax = fig.add_axes([X0 + c * (AW + GX), Y0 - AH - r * (AH + GY), AW, AH])
        vol, surf, _ = data[sid]
        paint(ax, vol[SLICE[sid]], surf, SLICE[sid], lw=0.45)
        ax.text(0.018, 0.965, f"Specimen {sid}", transform=ax.transAxes, fontsize=6.2,
                ha="left", va="top", color="white")
        axs[sid] = ax
    for sid, (rc, cc), h, _ in [(z[0], z[1], z[2], z[3]) for z in ZOOMS] + [(REC[0], REC[1], REC[2], "")]:
        axs[sid].add_patch(Rectangle((cc - h, rc - h), 2 * h, 2 * h, facecolor="none",
                                     edgecolor="white", linewidth=0.7))
    fig.text(0.010, Y0 + 0.012, "a", fontsize=8, fontweight="bold", va="bottom")

    # ---- b: zooms ----
    ZW = 0.178; ZY = Y0 - 2 * AH - GY - 0.075 - ZW * (ns.W2 / (FH_MM * ns.MM))
    ZH = ZW * ns.W2 / (FH_MM * ns.MM)
    for j, (sid, ctr, h, lab) in enumerate(ZOOMS):
        ax = fig.add_axes([0.028 + j * (ZW + 0.018), ZY, ZW, ZH])
        vol, surf, inv = data[sid]
        paint(ax, vol[SLICE[sid]], surf, SLICE[sid], lw=0.9, window=(ctr, h))
        if lab == "3HG junction":
            V, _ = nib.freesurfer.read_geometry(f"{FS}/Subject{sid}/surf/rh.white")
            p = nib.affines.apply_affine(inv, V[HINGE_V])
            ax.plot([p[2]], [p[1]], marker="o", ms=8, mfc="none", mec="#ff5030", mew=1.4)
        ax.set_xlabel(f"{lab}  (S{sid})", fontsize=6.2, labelpad=2)
    fig.text(0.010, ZY + ZH + 0.012, "b", fontsize=8, fontweight="bold", va="bottom")

    # ---- c: RecoverSulci before / after ----
    sid, ctr, h = REC
    vol, surf, inv = data[sid]
    tis = np.asarray(nib.load(f"{IB}/Subject{sid}_tissue.nii.gz").dataobj)[10:-10, 10:-10, 10:-10]
    rec = np.asarray(nib.load(f"{IB}/Subject{sid}_rec.nii.gz").dataobj)[10:-10, 10:-10, 10:-10]
    def labimg(A):
        L = A[SLICE[sid]].astype(float)
        out = np.zeros_like(L)
        out[L == 10] = 0.10          # immersion medium
        out[L == 150] = 0.55         # grey
        out[L == 250] = 0.95         # white
        out[(L == 50) | (L == 100)] = 0.55   # carved: drawn as the red overlay instead
        return out
    for j, (A, lab, show_carved) in enumerate([(tis, "before RecoverSulci", False),
                                               (rec, "after, carved sulci", True)]):
        ax = fig.add_axes([0.028 + 3 * (ZW + 0.018) + j * (ZW + 0.018), ZY, ZW, ZH])
        ax.imshow(labimg(A), cmap="gray", vmin=0, vmax=1, origin="lower",
                  interpolation="nearest")
        if show_carved:
            carved = np.isin(A[SLICE[sid]], [50, 100]).astype(float)
            ov = np.zeros(carved.shape + (4,)); ov[..., 0] = 1.0; ov[..., 1] = 0.31
            ov[..., 2] = 0.19; ov[..., 3] = carved * 0.9
            ax.imshow(ov, origin="lower", interpolation="nearest")
            seg = segments(surf["pial"], SLICE[sid])
            ax.add_collection(LineCollection(seg, colors=C_PIAL, linewidths=0.9))
        (rc, cc) = ctr
        ax.set_xlim(cc - h, cc + h); ax.set_ylim(rc - h, rc + h)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("black")
        ax.set_xlabel(lab + f"  (S{sid})", fontsize=6.2, labelpad=2)
    fig.text(0.028 + 3 * (ZW + 0.018) - 0.018, ZY + ZH + 0.012, "c", fontsize=8,
             fontweight="bold", va="bottom")

    fig.legend(handles=[Line2D([], [], color=C_WHITE, lw=1.6, label="white surface"),
                        Line2D([], [], color=C_PIAL, lw=1.6, label="pial surface"),
                        Line2D([], [], color="#ff5030", lw=1.6, label="carved sulcal CSF")],
               loc="lower center", ncol=3, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, 0.005))
    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{RES}/FIGS_surfqc.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    Image.MAX_IMAGE_PIXELS = None
    w, hh = Image.open(f"{RES}/FIGS_surfqc.png").size
    print(f"wrote FIGS_surfqc.png  {w/DPI*25.4:.0f} x {hh/DPI*25.4:.0f} mm")


if __name__ == "__main__":
    main()

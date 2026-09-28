#!/usr/bin/env python3
"""Stage 1: build FreeSurfer-style volumes from the GM/WM segmentation.

Ex-vivo hemispheres have no usable T1 contrast, so synthesise a pseudo-T1
(white=110, cortex=60, background=0) from the nnU-Net labels, letting downstream
FreeSurfer steps run unchanged.

White-matter interior = WM label + cavities enclosed by the WM shell.  Cavities
(ventricle, residual deep grey) are found with a 6-ray test rather than
binary_fill_holes, because the midline cut leaves them open to the exterior.
"""
import sys, numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure

import os
# defaults are relative to the directory that holds this repository; override with the
# environment variables
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.environ.get("EXVIVO_DATASET", os.path.join(_PROJECT, "28211843"))
OUT = os.environ.get("EXVIVO_FS", os.path.join(_PROJECT, "surf/fs"))
# SEG / SUFFIX let a different segmentation drive the same pipeline (see labelfree.py)
SEG = os.environ.get("SEG", "{ROOT}/{s}/{s}_Mask/T2w_GM_WM_mask.nii.gz")
SUFFIX = os.environ.get("SUFFIX", "")
ISO = 0.5                       # resample to isotropic; native is 0.45-0.6 anisotropic
ST = np.ones((3, 3, 3), bool)


def lcc(m):
    lab, n = ndi.label(m, structure=ST)
    return m if n <= 1 else (lab == (np.bincount(lab.ravel())[1:].argmax() + 1))


def enclosed(mask):
    """Voxels blocked by `mask` along all 6 axis rays -> inside the shell."""
    hits = np.zeros(mask.shape, np.uint8)
    for ax in range(3):
        for rev in (False, True):
            m = np.flip(mask, ax) if rev else mask
            c = np.cumsum(m, axis=ax) > 0
            hits += (np.flip(c, ax) if rev else c).astype(np.uint8)
    return hits >= 6


def sdf_resample(binary, aff_in, zooms_in, aff_out, shape_out):
    """Resample a binary volume through its signed distance field.

    Nearest-neighbour would stair-step the 0.6 mm through-plane direction and
    inject spurious handles; interpolating the SDF keeps the boundary smooth.
    """
    d_out = ndi.distance_transform_edt(~binary, sampling=zooms_in)
    d_in = ndi.distance_transform_edt(binary, sampling=zooms_in)
    sdf = d_out - d_in                                   # <0 inside
    ijk = np.indices(shape_out).reshape(3, -1)
    xyz = nib.affines.apply_affine(aff_out, ijk.T)
    src = nib.affines.apply_affine(np.linalg.inv(aff_in), xyz).T
    out = ndi.map_coordinates(sdf, src, order=1, mode="constant", cval=10.0)
    return out.reshape(shape_out) < 0


def genus(m):
    return (2 - measure.euler_number(m, connectivity=3)) // 2


def build(sid):
    s = f"Subject{sid}"
    im = nib.load(SEG.format(ROOT=ROOT, s=s))
    s = s + SUFFIX
    d = np.asarray(im.dataobj).astype(np.uint8)
    zin = np.array(im.header.get_zooms()[:3], float)

    wm = lcc(d == 2)
    cav = ndi.binary_opening(enclosed(wm) & (d == 0), structure=ST)
    white = lcc(ndi.binary_fill_holes(wm | cav))
    ribbon = (d == 1) & ~white
    ribbon = ribbon & ndi.binary_dilation(white, structure=ST, iterations=6)  # drop stray GM specks
    brain = lcc(white | ribbon)
    ribbon = brain & ~white

    # isotropic target grid, same world coords, 4 mm margin
    ijk = np.array(np.nonzero(brain))
    P = nib.affines.apply_affine(im.affine, ijk.T)
    lo, hi = P.min(0) - 4.0, P.max(0) + 4.0
    shape = tuple(int(np.ceil((hi[i] - lo[i]) / ISO)) + 1 for i in range(3))
    aff = np.eye(4); aff[:3, :3] = np.diag([ISO] * 3); aff[:3, 3] = lo

    W = sdf_resample(white, im.affine, zin, aff, shape)
    B = sdf_resample(brain, im.affine, zin, aff, shape)
    W = lcc(ndi.binary_fill_holes(W)); B = lcc(ndi.binary_fill_holes(B | W))
    R = B & ~W

    # Topology from a smoothed copy, geometry from the sharp one.  mris_fix_topology
    # retessellates each defect from the intensities, which for the pseudo-T1 only
    # restate the labels, so fewer defects means less invented tessellation.  Smoothing
    # the signed distance field and re-thresholding to preserve volume removes
    # speckle-induced handles without moving the boundary more than a voxel.  The surface
    # is later deformed back onto the sharp boundary, and deformation preserves genus.
    WT = W
    sig = float(os.environ.get("SMOOTH", "0"))
    if sig > 0:
        sdf = (ndi.distance_transform_edt(~W) - ndi.distance_transform_edt(W)) * ISO
        sm = ndi.gaussian_filter(sdf, sig / ISO)
        WT = sm < np.partition(sm.ravel(), W.sum())[W.sum()]      # volume-preserving cut
        WT = lcc(ndi.binary_fill_holes(WT))

    # pseudo-T1: FreeSurfer's expected intensities, lightly blurred for sub-voxel gradients
    t1 = np.zeros(shape, np.float32)
    t1[R] = 60.0; t1[W] = 110.0
    t1 = ndi.gaussian_filter(t1, 0.6)
    t1 = np.clip(np.rint(t1), 0, 255).astype(np.uint8)

    os.makedirs(f"{OUT}/{s}/mri", exist_ok=True)
    os.makedirs(f"{OUT}/{s}/surf", exist_ok=True)
    os.makedirs(f"{OUT}/{s}/label", exist_ok=True)
    os.makedirs(f"{OUT}/{s}/scripts", exist_ok=True)

    def save(arr, name):
        nib.MGHImage(np.asarray(arr), aff).to_filename(f"{OUT}/{s}/mri/{name}")

    save(np.where(W, 110, 0).astype(np.uint8), "wm.mgz")        # sharp: the geometry target
    save(np.where(WT, 255, 0).astype(np.uint8), "filled.mgz")   # smoothed: tessellate this
    save(t1, "brain.finalsurfs.mgz")
    save(t1, "brain.mgz")
    save(t1, "norm.mgz")
    save(t1, "T1.mgz")
    save(t1, "orig.mgz")
    # Minimal aseg; must carry right-hemisphere labels: 41 = Right-Cerebral-WM,
    # 42 = Right-Cerebral-Cortex.  mris_place_surface runs with --rh, and MRISripMidline()
    # reads the seg to freeze vertices belonging to the other hemisphere.  Left-hemisphere
    # labels (2/3) would rip every vertex, leaving the surface unmoved.
    save(np.where(W, 41, np.where(R, 42, 0)).astype(np.uint8), "aseg.presurf.mgz")

    vv = ISO ** 3 / 1000.0
    print(f"{s}: white {W.sum()*vv:6.1f} cm3 (cav +{cav.sum()*np.prod(zin)/1000:5.2f})  "
          f"cortex {R.sum()*vv:6.1f} cm3   grid {shape}  "
          f"genus {genus(W)}" + (f" -> {genus(WT)} (tessellated)" if sig > 0 else ""))
    return genus(WT)


if __name__ == "__main__":
    for i in (range(1, 7) if len(sys.argv) < 2 else [int(sys.argv[1])]):
        build(i)

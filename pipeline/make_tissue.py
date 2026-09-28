#!/usr/bin/env python3
"""Define the cortical ribbon from the *image* rather than the provided labels.

The provided GM label distorts the pial two ways: it bleeds into the immersion medium
(ribbon too fat), and it leaves an unlabelled band of real tissue between GM and WM
(ribbon too thin).

After N4 the image separates tissue from medium by a simple intensity threshold: the medium
forms a dark peak near 0 while tissue starts well above it, so any threshold in a wide range
gives the same result.

Ex-vivo specimen: no CSF.  The dark region is the immersion medium (proton-free, reads ~0),
which only fills sulci that stayed open; collapsed sulci hold nothing to detect, so many
white vertices never reach medium along their normal.
"""
import sys, numpy as np, nibabel as nib
from scipy import ndimage as ndi

import os
# defaults are relative to the directory that holds this repository; override with the
# environment variables
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.environ.get("EXVIVO_DATASET", os.path.join(_PROJECT, "28211843"))
CSF = os.environ.get("EXVIVO_CSF", os.path.join(_PROJECT, "surf/csf"))
FS = os.environ.get("EXVIVO_FS", os.path.join(_PROJECT, "surf/fs"))
SEG = os.environ.get("SEG", "{ROOT}/{s}/{s}_Mask/T2w_GM_WM_mask.nii.gz")
SUFFIX = os.environ.get("SUFFIX", "")
ST = np.ones((3, 3, 3), bool)
TISSUE, MEDIUM = 0.40, 0.30          # x the *local* tissue reference, not a global one
REF_SIGMA = 8.0                      # mm, scale of the local reference field


def lcc(m):
    lab, n = ndi.label(m, structure=ST)
    return m if n <= 1 else (lab == (np.bincount(lab.ravel())[1:].argmax() + 1))


def resample(binary, aff_in, zoom_in, aff_out, shape_out):
    """Resample a binary mask via its signed distance field (avoids NN stair-stepping)."""
    sdf = (ndi.distance_transform_edt(~binary, sampling=zoom_in)
           - ndi.distance_transform_edt(binary, sampling=zoom_in))
    ijk = np.indices(shape_out).reshape(3, -1)
    xyz = nib.affines.apply_affine(aff_out, ijk.T)
    src = nib.affines.apply_affine(np.linalg.inv(aff_in), xyz).T
    return ndi.map_coordinates(sdf, src, order=1, mode="constant",
                               cval=10.0).reshape(shape_out) < 0


def run(sid):
    s = f"Subject{sid}"
    seg = nib.load(SEG.format(ROOT=ROOT, s=s))
    S = np.asarray(seg.dataobj)
    img = np.asarray(nib.load(f"{CSF}/S{sid}.t2w_n4.nii.gz").dataobj).astype(np.float32)
    g = img / np.median(img[S == 2])                    # WM as the intensity reference
    zin = np.array(seg.header.get_zooms()[:3], float)

    # close the sulci so their contents count as "inside", then split tissue from medium
    env = ndi.binary_fill_holes(ndi.binary_closing(S > 0, structure=ST, iterations=7))

    # The threshold must be local: B1 inhomogeneity darkens the specimen periphery, where a
    # global WM cut would delete cortex.  The medium still reads ~0 there, so tissue stays
    # separable against a local reference -- normalised-convolution smoothing of the intensity
    # over the labelled brain, which follows the intensity dropout.
    brain = (S > 0).astype(np.float32)
    sig = REF_SIGMA / zin
    ref = (ndi.gaussian_filter(g * brain, sig) /
           np.maximum(ndi.gaussian_filter(brain, sig), 1e-3))
    ref = np.maximum(ref, 0.3)

    tissue = lcc(ndi.binary_fill_holes(env & (g >= TISSUE * ref)))
    medium = env & (g < MEDIUM * ref) & ~tissue

    s = s + SUFFIX
    wmi = nib.load(f"{FS}/{s}/mri/wm.mgz")               # the 0.5 mm grid the surfaces live on
    aff, shape = wmi.header.get_affine(), wmi.shape
    W = np.asarray(wmi.dataobj) > 0

    T = resample(tissue, seg.affine, zin, aff, shape)
    M = resample(medium, seg.affine, zin, aff, shape)
    T = lcc(ndi.binary_fill_holes(T | W))                # white is tissue by construction
    R = T & ~W                                           # the ribbon, image-defined
    M = M & ~T

    nib.MGHImage(T.astype(np.uint8), aff).to_filename(f"{FS}/{s}/mri/tissue.mgz")
    nib.MGHImage(R.astype(np.uint8), aff).to_filename(f"{FS}/{s}/mri/ribbon.mgz")
    nib.MGHImage(M.astype(np.uint8), aff).to_filename(f"{FS}/{s}/mri/medium.mgz")

    vv = 0.125 / 1000
    lab_ribbon = (S == 1).sum() * np.prod(zin) / 1000
    print(f"{s}: ribbon {R.sum()*vv:6.1f} cm3 (label said {lab_ribbon:6.1f}, "
          f"{100*(R.sum()*vv/lab_ribbon - 1):+3.0f}%) | medium {M.sum()*vv:5.1f} cm3")


if __name__ == "__main__":
    for i in (range(1, 7) if len(sys.argv) < 2 else [int(sys.argv[1])]):
        run(i)

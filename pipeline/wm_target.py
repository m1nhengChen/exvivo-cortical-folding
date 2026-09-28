#!/usr/bin/env python3
"""Map the real N4 FLASH image to T1-like contrast for mris_place_surface.

A pseudo-T1 synthesised from the labels carries no information the labels lack, so surface
placement only re-finds the label boundary and reproduces its noise.  The image itself
separates cortex from WM locally; only a global threshold fails, because B1 dropout shifts the
intensity scale across the specimen.  mris_autodet_gwstats samples along each vertex normal and
works locally, so it just needs a volume where brighter-than-WM means cortex.

The mapping is a tent centred on white matter (both WM neighbours are darker in T1 terms):

    medium (0)  ->    0        immersion medium is proton-free, reads 0
    WM  (gw)    ->  110        FreeSurfer's white target
    GM  (gg)    ->   60        FreeSurfer's grey target

Intensities are taken against a local reference field (sigma 8 mm) to divide out B1 dropout
before mapping.  --blend mixes in the label-derived volume as a prior while the image supplies
the geometry.
"""
import sys, argparse, numpy as np, nibabel as nib
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
REF_SIGMA = 8.0


def run(sid, blend, dn):
    s = f"Subject{sid}"
    seg = nib.load(SEG.format(ROOT=ROOT, s=s))
    S = np.asarray(seg.dataobj)
    suf = "_dn" if dn else ""
    img = np.asarray(nib.load(f"{CSF}/S{sid}.t2w_n4{suf}.nii.gz").dataobj).astype(np.float32)
    z = np.array(seg.header.get_zooms()[:3], float)

    # local reference: divides out the B1 dropout the way N4's smooth model cannot
    brain = (S > 0).astype(np.float32)
    ref = np.maximum(ndi.gaussian_filter(img * brain, REF_SIGMA / z) /
                     np.maximum(ndi.gaussian_filter(brain, REF_SIGMA / z), 1e-3), 1e-3)
    g = img / ref

    # calibrate the tent on label interiors -- eroded, so partial volume stays out
    gw = np.median(g[ndi.binary_erosion(S == 2, ST, 2)])
    gg = np.median(g[ndi.binary_erosion(S == 1, ST, 1)])

    t1 = np.where(g <= gw, 110.0 * g / gw, 110.0 - 50.0 * (g - gw) / (gg - gw))
    t1 = np.clip(t1, 0, 140)

    # the surface grid the FreeSurfer volumes live on
    s = s + SUFFIX
    wmi = nib.load(f"{FS}/{s}/mri/wm.mgz")
    aff, shape = wmi.header.get_affine(), wmi.shape
    ijk = np.indices(shape).reshape(3, -1)
    src = nib.affines.apply_affine(np.linalg.inv(seg.affine),
                                   nib.affines.apply_affine(aff, ijk.T)).T
    T1 = ndi.map_coordinates(t1, src, order=1, mode="constant", cval=0).reshape(shape)

    # medium reads exactly 0: force it to 0, and floor the tissue to keep bright cortex off
    # the medium branch of the tent
    T = np.asarray(nib.load(f"{FS}/{s}/mri/tissue.mgz").dataobj) > 0
    T1 = np.where(T, np.maximum(T1, 35.0), np.minimum(T1, 10.0))

    if blend > 0:                      # keep the label as a prior, image as the geometry
        lab = np.asarray(nib.load(f"{FS}/{s}/mri/brain.finalsurfs.mgz").dataobj).astype(np.float32)
        # Blur the two halves separately: speckle lives in the label half while the image half
        # carries the edge.  A symmetric Gaussian does not move a step edge, so the label half
        # tolerates far more blur than blurring the two together (TSIGMA) would allow.
        ls = float(os.environ.get("LSIGMA", "0"))
        if ls > 0:
            lab = ndi.gaussian_filter(lab, ls)
        T1 = (1 - blend) * T1 + blend * lab

    # Blur the target.  A symmetric Gaussian does not move a step edge (max-gradient location
    # preserved), so it cannot bias the white surface in or out; it only gives up detail finer
    # than the blur and regularises against a noisy target.
    T1 = ndi.gaussian_filter(T1, float(os.environ.get("TSIGMA", "0.6")))
    out = np.clip(np.rint(T1), 0, 255).astype(np.uint8)
    ts = os.environ.get("TSIGMA", "0.6"); ls = os.environ.get("LSIGMA", "0")
    tag = ("dn" if dn else "") + ("" if blend == 0 else f".b{int(blend*100)}")
    tag += "" if ts == "0.6" else f".s{ts.replace('.','')}"
    tag += "" if ls == "0" else f".L{ls.replace('.','')}"
    tag = f".{tag}" if tag and not tag.startswith(".") else tag
    nib.MGHImage(out, aff).to_filename(f"{FS}/{s}/mri/brain.image{tag}.mgz")

    W = np.asarray(nib.load(f"{FS}/{s}/mri/wm.mgz").dataobj) > 0
    R = np.asarray(nib.load(f"{FS}/{s}/mri/ribbon.mgz").dataobj) > 0
    print(f"{s}: gw={gw:.2f} gg={gg:.2f} (对比度 {gg/gw:.2f})  ->  "
          f"目标体内 WM={np.median(out[W]):3.0f} GM={np.median(out[R]):3.0f}  "
          f"重叠={100*np.mean(out[R] > np.median(out[W])):.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subjects", nargs="*", type=int, default=list(range(1, 7)))
    p.add_argument("--blend", type=float, default=0.0)
    p.add_argument("--dn", action="store_true")
    a = p.parse_args()
    for i in a.subjects:
        run(i, a.blend, a.dn)

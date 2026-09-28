#!/usr/bin/env python3
"""Build brain.pialvol.mgz, the volume mris_place_surface uses to place the pial.

The deformable model is driven by the segmentation, not the raw T2w: intensity is flat
inside the grey ribbon so the gradient term can only stop the surface at the outer
cortical edge, not partway through the cortex.

Collapsed sulci (banks in contact, no CSF between them) leave most white vertices with
nothing to stop at. iBEAT's GeodesicDistanceTransform locates the buried sulcal sheets and
RecoverSulci cuts them open in the volume, producing the CSF separation the method needs.
"""
import os, sys, subprocess, numpy as np, nibabel as nib
from scipy import ndimage as ndi

# defaults are relative to the directory that holds this repository; override with the
# environment variables
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FS = os.environ.get("EXVIVO_FS", os.path.join(_PROJECT, "surf/fs"))
IB = os.environ.get("EXVIVO_IB", os.path.join(_PROJECT, "surf/ibeat"))
DOCK = "ibeatgroup/ibeat_v2:release210"
ST = np.ones((3, 3, 3), bool)
PAD = 10
HULL = 6

WM_I, GM_I, CSF_I = 110, 60, 5      # intensities FreeSurfer expects: bright WM, mid GM, dark CSF


def build_tissue(sid, suf=""):
    """Write the tissue label map in iBEAT's convention on the 0.5 mm grid: 250 WM, 150 GM, 10 CSF."""
    s = f"Subject{sid}{suf}"
    wmi = nib.load(f"{FS}/{s}/mri/wm.mgz")
    W = np.asarray(wmi.dataobj) > 0
    R = np.asarray(nib.load(f"{FS}/{s}/mri/ribbon.mgz").dataobj) > 0
    lab = np.zeros(W.shape, np.uint8)
    lab[ndi.binary_dilation(W | R, ST, HULL)] = 10
    lab[R] = 150
    lab[W] = 250
    aff = wmi.header.get_affine().copy()
    lab = np.pad(lab, PAD)
    aff[:3, 3] -= aff[:3, :3] @ np.full(3, PAD)
    nib.Nifti1Image(lab, aff).to_filename(f"{IB}/{s}_tissue.nii.gz")


def open_sulci(sid, suf=""):
    s = f"Subject{sid}{suf}"
    subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/bin/bash", "-v", f"{IB}:/D", DOCK, "-c",
         f"B=/usr/local/InfantPipeline/bin; cd /D; "
         f"$B/GeodesicDistanceTransform -i {s}_tissue.nii.gz -o {s}_sulci.nii.gz && "
         f"$B/RecoverSulci --in {s}_tissue.nii.gz --sulci {s}_sulci.nii.gz --out {s}_rec.nii.gz"],
        check=True, capture_output=True)


def pial_volume(sid, suf=""):
    """Convert the recovered label map to the intensities the pial step is calibrated for."""
    s = f"Subject{sid}{suf}"
    rec = nib.load(f"{IB}/{s}_rec.nii.gz")
    A = np.asarray(rec.dataobj)[PAD:-PAD, PAD:-PAD, PAD:-PAD]

    wmi = nib.load(f"{FS}/{s}/mri/wm.mgz")
    assert A.shape == wmi.shape, (A.shape, wmi.shape)

    t1 = np.zeros(A.shape, np.float32)
    t1[A == 150] = GM_I                       # flat grey so the gradient term cannot fire inside the ribbon
    t1[A == 250] = WM_I
    t1[(A == 100) | (A == 50)] = CSF_I        # sulci opened by RecoverSulci
    t1[A == 10] = CSF_I                       # background outside the brain
    t1 = ndi.gaussian_filter(t1, 0.6)         # sub-voxel gradient at edges only
    out = np.clip(np.rint(t1), 0, 255).astype(np.uint8)
    nib.MGHImage(out, wmi.header.get_affine()).to_filename(f"{FS}/{s}/mri/brain.pialvol.mgz")

    v = 0.125 / 1000
    print(f"{s}: 挖开的沟 {((A==100)|(A==50)).sum()*v:5.1f} cm3 | GM {(A==150).sum()*v:6.1f} | "
          f"WM {(A==250).sum()*v:6.1f}  -> mri/brain.pialvol.mgz")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("subjects", nargs="*", type=int, default=list(range(1, 7)))
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    for i in a.subjects:
        build_tissue(i, a.suffix)
        open_sulci(i, a.suffix)
        pial_volume(i, a.suffix)

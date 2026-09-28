#!/usr/bin/env python3
"""nnU-Net FLASH GM/WM output -> 0.5 mm GM/WM seg + N4-corrected image on a world grid.

The seg is on the spoofed grid (content 0.20 mm, header 0.15 mm, cropped).  nnU-Net writes on
its input voxel grid, so the true world affine is
  A_orig @ [[diag(content/orig_zoom), crop0], [0,1]]
Resample the seg and the original image onto a 0.5 mm world-axis-aligned grid, N4 the image,
and remap the nnU-Net labels to the GM=1 / WM=2 convention the surface pipeline expects.

    ./seg_post.py <workdir> <nnunet_out.nii.gz> <out_seg.nii.gz> <out_n4.nii.gz>
"""
import sys, os, shutil, subprocess
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

ISO = 0.5


def true_affine(meta):
    A = np.array(meta["orig_affine"]); z = np.array(meta["orig_zoom"])
    c0 = np.array(meta["crop0"]); ct = meta["content"]
    M = np.eye(4); M[:3, :3] = np.diag(ct / z); M[:3, 3] = c0
    return A @ M


def resample(vol, aff_in, shape_out, aff_out, order):
    ijk = np.indices(shape_out).reshape(3, -1)
    xyz = nib.affines.apply_affine(aff_out, ijk.T)
    src = nib.affines.apply_affine(np.linalg.inv(aff_in), xyz).T
    return ndi.map_coordinates(vol, src, order=order, mode="constant", cval=0).reshape(shape_out)


def n4_binary():
    for c in ("/opt/ants/bin/N4BiasFieldCorrection", "N4BiasFieldCorrection"):
        if os.path.isabs(c) and os.path.exists(c):
            return c
        if not os.path.isabs(c) and shutil.which(c):
            return c
    raise SystemExit("seg_post: N4BiasFieldCorrection (ANTs) not found on /opt/ants/bin or PATH")


def run(workdir, nnout, out_seg, out_n4):
    meta = np.load(os.path.join(workdir, "case_meta.npy"), allow_pickle=True).item()
    seg = np.asarray(nib.load(nnout).dataobj).astype(np.uint8)
    segaff = true_affine(meta)

    im = nib.load(meta["raw"])
    img = np.asarray(im.dataobj).astype(np.float32); imgaff = im.affine

    corners = np.array([[0, 0, 0], [*seg.shape], [seg.shape[0], 0, 0], [0, seg.shape[1], 0],
                        [0, 0, seg.shape[2]], [seg.shape[0], seg.shape[1], 0],
                        [seg.shape[0], 0, seg.shape[2]], [0, seg.shape[1], seg.shape[2]]], float)
    wc = nib.affines.apply_affine(segaff, corners)
    lo, hi = wc.min(0) - 3, wc.max(0) + 3
    shape = np.ceil((hi - lo) / ISO).astype(int)
    aff_out = np.eye(4); aff_out[:3, :3] = np.diag([ISO] * 3); aff_out[:3, 3] = lo

    segR = resample(seg, segaff, shape, aff_out, order=0).astype(np.uint8)
    imgR = resample(img, imgaff, shape, aff_out, order=1).astype(np.float32)

    # remap nnU-Net FLASH labels: 3=outer ribbon->GM, 1=sulcus patch->GM, 2=WM.
    # wm_target calibrates contrast from seg medians, so polarity is handled downstream.
    lab = np.zeros_like(segR)
    lab[segR == 3] = 1; lab[segR == 1] = 1
    lab[segR == 2] = 2
    if (lab == 2).sum() == 0 or (lab == 1).sum() == 0:
        raise SystemExit(f"seg_post: segmentation has no {'WM' if (lab==2).sum()==0 else 'GM'} "
                         f"-- the model likely failed on this input")

    os.makedirs(os.path.dirname(os.path.abspath(out_seg)), exist_ok=True)
    nib.Nifti1Image(lab, aff_out).to_filename(out_seg)

    os.makedirs(os.path.dirname(os.path.abspath(out_n4)), exist_ok=True)
    raw05 = out_n4.replace(".nii.gz", "") + ".raw.nii.gz"
    mask = out_n4.replace(".nii.gz", "") + ".n4mask.nii.gz"
    nib.Nifti1Image(imgR, aff_out).to_filename(raw05)
    brain = ndi.binary_dilation(lab > 0, iterations=3)
    nib.Nifti1Image(brain.astype(np.uint8), aff_out).to_filename(mask)
    subprocess.run([n4_binary(), "-d", "3", "-i", raw05, "-x", mask, "-o", out_n4,
                    "-s", "2", "-c", "[100x100x100x100,1e-7]", "-b", "[120]"],
                   check=True, capture_output=True)

    gm_i, wm_i = np.median(imgR[lab == 1]), np.median(imgR[lab == 2])
    print(f"seg_post: 0.5mm {tuple(shape)}  GM {np.mean(lab==1)*100:.1f}% WM {np.mean(lab==2)*100:.1f}%  "
          f"raw GM={gm_i:.0f} WM={wm_i:.0f} -> {out_seg}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: seg_post.py <workdir> <nnunet_out.nii.gz> <out_seg.nii.gz> <out_n4.nii.gz>")
    run(*sys.argv[1:])

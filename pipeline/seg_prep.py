#!/usr/bin/env python3
"""Prepare an ex-vivo FLASH hemisphere for the FLASH GM/WM nnU-Net (Task278).

The model resamples every input to 0.15 mm internally; at that spacing a whole hemisphere is
350-610 M voxels and the softmax + resample-back exhausts memory.  Feed the model a volume
already at its target spacing: resample the content to CONTENT mm and relabel the header
spacing to 0.15 mm, so nnU-Net does no internal resampling.  The crop offset and true content
spacing are saved so seg_post.py can rebuild the real world affine.

    ./seg_prep.py <input_flash.nii.gz> <workdir>
      -> <workdir>/data_for_inference/case_0000.nii.gz   (nnU-Net input)
      -> <workdir>/case_meta.npy                          (for seg_post.py)
"""
import sys, os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

CONTENT = 0.20          # resample content to this; header set to 0.15 to skip nnU-Net resampling
LABEL_SP = 0.15
THR = 20                # crop away air / immersion medium below this FLASH intensity
MARGIN = 6


def prep(inp, workdir):
    im = nib.load(inp)
    a = np.asarray(im.dataobj).astype(np.float32)
    z = np.array(im.header.get_zooms()[:3], float)
    A = im.affine.copy()

    mask = a > THR
    if not mask.any():
        raise SystemExit(f"seg_prep: nothing above intensity {THR} in {inp} -- wrong THR or empty image")
    xs = np.where(mask.any((1, 2)))[0]; ys = np.where(mask.any((0, 2)))[0]; zs = np.where(mask.any((0, 1)))[0]
    c0 = [max(0, xs[0] - MARGIN), max(0, ys[0] - MARGIN), max(0, zs[0] - MARGIN)]
    c1 = [min(a.shape[k], (xs, ys, zs)[k][-1] + MARGIN) for k in range(3)]
    blk = a[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]

    b = ndi.zoom(blk, z / CONTENT, order=1).astype(np.int16)

    # centred affine at the spoofed 0.15 mm spacing; real placement rebuilt in seg_post
    aff = np.eye(4)
    for j in range(3):
        aff[j, j] = LABEL_SP
    aff[:3, 3] = [-b.shape[k] * LABEL_SP / 2 for k in range(3)]
    out = nib.Nifti1Image(b, aff); out.header.set_zooms((LABEL_SP,) * 3)

    di = os.path.join(workdir, "data_for_inference")
    os.makedirs(di, exist_ok=True)
    out.to_filename(os.path.join(di, "case_0000.nii.gz"))

    meta = dict(orig_affine=A.tolist(), orig_zoom=z.tolist(), crop0=[int(x) for x in c0],
                content=CONTENT, seg_shape=list(b.shape), raw=os.path.abspath(inp))
    np.save(os.path.join(workdir, "case_meta.npy"), meta, allow_pickle=True)
    print(f"seg_prep: crop {blk.shape} -> content {CONTENT} mm {b.shape} = {b.size/1e6:.0f}M vox")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: seg_prep.py <input_flash.nii.gz> <workdir>")
    prep(sys.argv[1], sys.argv[2])

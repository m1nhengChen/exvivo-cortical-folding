#!/usr/bin/env python3
"""Assemble the raw SSFP-DWI into a single 4D series + gradient table, fit DTI, and VALIDATE
FA against the dataset's own FA map. If they agree, the gradient table is correct and we can
trust downstream CSD/tractography. SSFP nominal b=6000 treated as a standard shell -- the
same approximation the dataset authors used (they published DTI + Subject4 tensor tracts)."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, glob, numpy as np, nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from scipy import ndimage as ndi

ROOT = (_env.ROOT + "/28211843")
OUT = (_env.ROOT + "/surf/three_hinge/tracto")


def assemble(sid):
    s = f"Subject{sid}"
    raw = f"{ROOT}/{s}/{s}_Raw_data"
    q = sorted(glob.glob(f"{raw}/SSFP_DWI_Q300_FA29_B6K_SET*"))
    b0 = sorted(glob.glob(f"{raw}/SSFP_DWI_FMRIB_Q20_*"))
    print(f"{s}: {len(q)} DWI sets, {len(b0)} b0 sets")

    def load(d):
        return nib.load(glob.glob(f"{d}/*.nii.gz")[0])
    ref = load(q[0]); aff = ref.affine
    b0v = [np.asarray(load(d).dataobj, np.float32) for d in b0]
    dwv = [np.asarray(load(d).dataobj, np.float32) for d in q]
    bvecs = np.concatenate([np.loadtxt(f"{d}/bvecs.txt") for d in q], axis=1)   # 3 x 60
    nb0 = sum(x.shape[3] for x in b0v); ndw = sum(x.shape[3] for x in dwv)
    data = np.concatenate(b0v + dwv, axis=3)
    bvals = np.array([0] * nb0 + [6000] * ndw, float)
    gbvecs = np.concatenate([np.zeros((3, nb0)), bvecs], axis=1).T              # (80,3)
    print(f"  assembled 4D {data.shape}  |  {nb0} b0 + {ndw} dwi @ b6000")

    d = f"{OUT}/{s}"; os.makedirs(d, exist_ok=True)
    nib.save(nib.Nifti1Image(data, aff), f"{d}/dwi.nii.gz")
    np.savetxt(f"{d}/bvals.txt", bvals); np.savetxt(f"{d}/bvecs.txt", gbvecs)

    # brain mask from mean b0
    mb0 = data[..., :nb0].mean(-1)
    mask = mb0 > (0.15 * np.percentile(mb0, 99))
    mask = ndi.binary_closing(ndi.binary_fill_holes(mask), iterations=2)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), f"{d}/mask.nii.gz")

    # DTI fit + FA validation
    gtab = gradient_table(bvals, bvecs=gbvecs)
    ten = TensorModel(gtab).fit(data, mask=mask)
    fa = np.clip(ten.fa, 0, 1)
    nib.save(nib.Nifti1Image(fa.astype(np.float32), aff), f"{d}/fa_mine.nii.gz")

    ref_fa = np.asarray(nib.load(f"{ROOT}/{s}/{s}_DTI/FA.nii.gz").dataobj, np.float32)
    m = mask & (ref_fa > 0) & np.isfinite(fa)
    r = np.corrcoef(fa[m], ref_fa[m])[0, 1]
    print(f"  FA validation: corr(mine, dataset) = {r:.3f}  "
          f"(mine med {np.median(fa[m]):.3f}, dataset med {np.median(ref_fa[m]):.3f})")
    return r


if __name__ == "__main__":
    for i in [int(x) for x in sys.argv[1:]] or [5]:
        assemble(i)

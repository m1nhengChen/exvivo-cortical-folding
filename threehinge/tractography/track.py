#!/usr/bin/env python3
"""CSD -> FOD peaks -> whole-brain deterministic tractography (single-shell b6000, ex-vivo).
Seeds from the WM mask; stops on low FA. Saves peaks + tractogram in DWI space."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response_ssst, ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.reconst.dti import TensorModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram

ROOT = (_env.ROOT + "/28211843"); OUT = (_env.ROOT + "/surf/three_hinge/tracto")


def run(sid):
    s = f"Subject{sid}"; d = f"{OUT}/{s}"
    img = nib.load(f"{d}/dwi.nii.gz"); data = np.asarray(img.dataobj, np.float32); aff = img.affine
    bvals = np.loadtxt(f"{d}/bvals.txt"); bvecs = np.loadtxt(f"{d}/bvecs.txt")
    gtab = gradient_table(bvals, bvecs=bvecs)
    mask = np.asarray(nib.load(f"{d}/mask.nii.gz").dataobj) > 0
    gmwm = np.asarray(nib.load(f"{ROOT}/{s}/{s}_Mask/DWI_GM_WM_mask.nii.gz").dataobj)
    wm = gmwm == 2

    print(f"{s}: FA for stopping ...", flush=True)
    fa = np.clip(TensorModel(gtab).fit(data, mask=mask).fa, 0, 1)

    print("CSD response + peaks (this is the slow part) ...", flush=True)
    resp, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.5)
    csd = ConstrainedSphericalDeconvModel(gtab, resp, sh_order_max=8)
    peaks = peaks_from_model(csd, data, default_sphere, relative_peak_threshold=0.5,
                             min_separation_angle=25, mask=mask, npeaks=3,
                             return_sh=False, num_processes=4)
    np.save(f"{d}/peak_dirs.npy", peaks.peak_dirs.astype(np.float32))
    np.save(f"{d}/peak_vals.npy", peaks.peak_values.astype(np.float32))
    nib.save(nib.Nifti1Image(fa.astype(np.float32), aff), f"{d}/fa.nii.gz")

    print("tracking ...", flush=True)
    scc = ThresholdStoppingCriterion(fa, 0.12)
    seeds = utils.seeds_from_mask(wm, aff, density=1)
    sl = LocalTracking(peaks, scc, seeds, aff, step_size=0.4, max_cross=1)
    sl = Streamlines(sl)
    sl = sl[[len(x) > 5 for x in sl]]                    # drop tiny fragments (>2mm)
    print(f"  {len(sl)} streamlines", flush=True)
    sft = StatefulTractogram(sl, img, Space.RASMM)
    save_tractogram(sft, f"{d}/whole_brain.trk", bbox_valid_check=False)
    print(f"  saved {d}/whole_brain.trk", flush=True)


if __name__ == "__main__":
    for i in [int(x) for x in sys.argv[1:]] or [5]:
        run(i)

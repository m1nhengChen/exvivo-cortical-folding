#!/usr/bin/env python3
"""Two ablations that change ONE thing each, inside one pipeline.

The earlier comparison (DIPY peaks vs MRtrix SD_Stream vs MRtrix iFOD2) moved software, response
function, ODF representation, propagator and stopping rule all at once, so it could rank the
pipelines but not attribute the difference to anything.

  A. max_cross = 1, 2, 3, all      stored CSD peaks, everything else identical
     max_cross sets how many initial directions are tracked FROM EACH SEED in a crossing voxel; it
     does not change what happens at each subsequent step, which is the direction getter's job. So
     this isolates how heavily crossing voxels are represented in the tractogram - and three-hinge
     gyri are where the crossings are.

  B. direction getter                same CSD fODF, same seeds, stopping, angle, step size
     peaks (published)   deterministic, three discrete peaks
     det. maximum        deterministic, full fODF
     probabilistic       samples the fODF
     This is the one that isolates the propagation rule.

B needs the fODF spherical harmonics, which track.py did not save; they are refit once per
specimen and cached (~3 min each)."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, time, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response_ssst, ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.direction import (ProbabilisticDirectionGetter, DeterministicMaximumDirectionGetter,
                            peaks as _p)
from dipy.direction.peaks import PeaksAndMetrics
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from tracto_sensitivity import geometry, enrich, TR, ROOT, SUBJ
OUT = (_env.ROOT + "/surf/three_hinge/validate")
STEP, FA_STOP, ANG = 0.4, 0.12, 60.0
QA, TW = 0.0239, 0.5                              # the published PeaksAndMetrics thresholds
MAXCROSS = [1, 2, 3, None]                        # None = every initial peak
GETTERS = ["peaks (published)", "deterministic maximum", "probabilistic"]


def common(sid):
    d = f"{TR}/Subject{sid}"
    img = nib.load(f"{d}/dwi.nii.gz")
    fa = np.asarray(nib.load(f"{d}/fa.nii.gz").dataobj)
    gmwm = np.asarray(nib.load(f"{ROOT}/Subject{sid}/Subject{sid}_Mask/DWI_GM_WM_mask.nii.gz").dataobj)
    seeds = utils.seeds_from_mask(gmwm == 2, img.affine, density=1)
    return img, fa, seeds


def peaks_obj(sid):
    d = f"{TR}/Subject{sid}"
    pd = np.load(f"{d}/peak_dirs.npy").astype(np.float64)
    pv = np.load(f"{d}/peak_vals.npy").astype(np.float64)
    V = default_sphere.vertices
    idx = np.full(pv.shape, -1, dtype=np.int64)
    nz = pv > 0
    idx[nz] = np.argmax(np.abs(pd[nz] @ V.T), axis=1)
    p = PeaksAndMetrics()
    p.sphere = default_sphere
    p.peak_dirs, p.peak_values, p.peak_indices = pd, pv, idx
    p.ang_thr, p.qa_thr, p.total_weight = ANG, QA, TW
    p._initialize()
    return p


def shcoeff(sid):
    d = f"{TR}/Subject{sid}"; cache = f"{d}/csd_sh.npy"
    if os.path.exists(cache):
        return np.load(cache)
    img = nib.load(f"{d}/dwi.nii.gz"); data = np.asarray(img.dataobj, np.float32)
    gtab = gradient_table(np.loadtxt(f"{d}/bvals.txt"), bvecs=np.loadtxt(f"{d}/bvecs.txt"))
    mask = np.asarray(nib.load(f"{d}/mask.nii.gz").dataobj) > 0
    resp, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.5)
    t0 = time.time()
    sh = ConstrainedSphericalDeconvModel(gtab, resp, sh_order_max=8).fit(
        data, mask=mask).shm_coeff.astype(np.float32)
    np.save(cache, sh)
    print(f"    (CSD refit {time.time()-t0:.0f}s)", flush=True)
    return sh


def run_track(dg, fa, seeds, aff, mx):
    sl = Streamlines(LocalTracking(dg, ThresholdStoppingCriterion(fa, FA_STOP), seeds, aff,
                                   step_size=STEP, max_cross=mx))
    return [np.asarray(x) for x in sl if len(x) > 5]


def main():
    A = np.full((len(MAXCROSS), len(SUBJ), 2), np.nan); AN = np.zeros((len(MAXCROSS), len(SUBJ)), int)
    B = np.full((len(GETTERS), len(SUBJ), 2), np.nan); BN = np.zeros((len(GETTERS), len(SUBJ)), int)
    for j, sid in enumerate(SUBJ):
        pw, hinge, crown = geometry(sid)
        img, fa, seeds = common(sid)
        p = peaks_obj(sid)
        for i, mx in enumerate(MAXCROSS):
            t0 = time.time(); s = run_track(p, fa, seeds, img.affine, mx)
            A[i, j] = enrich(s, img, pw, hinge, crown); AN[i, j] = len(s)
            print(f"S{sid} max_cross={str(mx):4s} {len(s):8d} sl  target {A[i,j,0]:5.2f}  "
                  f"seed {A[i,j,1]:5.2f}  ({time.time()-t0:.0f}s)", flush=True)
        sh = shcoeff(sid)
        for i, name in enumerate(GETTERS):
            t0 = time.time()
            if name.startswith("peaks"):
                B[i, j] = A[0, j]; BN[i, j] = AN[0, j]      # identical run, no need to repeat
                print(f"S{sid} {name:22s} (= max_cross 1 above)", flush=True); continue
            cls = (DeterministicMaximumDirectionGetter if "maximum" in name
                   else ProbabilisticDirectionGetter)
            dg = cls.from_shcoeff(sh, max_angle=ANG, sphere=default_sphere)
            s = run_track(dg, fa, seeds, img.affine, 1)
            B[i, j] = enrich(s, img, pw, hinge, crown); BN[i, j] = len(s)
            print(f"S{sid} {name:22s} {len(s):8d} sl  target {B[i,j,0]:5.2f}  "
                  f"seed {B[i,j,1]:5.2f}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{OUT}/ablate_tracking.npz", maxcross=np.array([str(m) for m in MAXCROSS]),
             getters=np.array(GETTERS), A=A, B=B, AN=AN, BN=BN)
    for tag, lab, M, N in (("A. max_cross", [str(m) for m in MAXCROSS], A, AN),
                           ("B. direction getter", GETTERS, B, BN)):
        print(f"\n{tag:>24s}{'streamlines':>13s}{'target contrast':>22s}{'seed contrast':>22s}")
        for i, nm in enumerate(lab):
            o = []
            for c in range(2):
                v = M[i, :, c]; v = v[np.isfinite(v)]
                o.append(f"{np.exp(np.mean(np.log(v))):8.2f}x {int((v>1).sum())}/{len(v)} "
                         f"P={stats.ttest_1samp(np.log(v),0).pvalue:.4f}")
            print(f"{nm:>24s}{int(N[i].mean()):13d}{o[0]:>22s}{o[1]:>22s}")
    print(f"\nwrote {OUT}/ablate_tracking.npz")


if __name__ == "__main__":
    main()

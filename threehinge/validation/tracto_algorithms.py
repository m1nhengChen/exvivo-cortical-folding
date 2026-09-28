#!/usr/bin/env python3
"""Does the inter-hinge excess depend on the tracking ALGORITHM, not just its parameters?

validate/tracto_sensitivity.py varies ten parameters of one algorithm - deterministic tracking on
discrete CSD peaks. Every one of those runs shares the same direction getter, so none of them
answers the criticism that matters here: a three-hinge gyrus is by construction the place where
three gyral branches meet, i.e. where fibre populations fan and cross, and deterministic tracking
on three discrete peaks with max_cross = 1 is exactly the family that handles crossings worst.

This script changes the algorithm class instead, over the direction getters DIPY ships:

  peaks (published)        deterministic, discrete CSD peaks - the baseline
  deterministic maximum    the full fODF rather than three peaks; tests the peak extraction step
  closest peak             follows the fODF lobe nearest the incoming direction
  probabilistic            samples the fODF; spreads INTO a fan rather than committing to one
                           branch, so if the excess were manufactured by deterministic commitment
                           at junctions this is where it should collapse
  parallel transport (PTT) a geometrically different propagator, not a lobe-selection rule

The first needs the stored peaks; the other four need the fODF spherical-harmonic coefficients,
which track.py did not save (return_sh=False). Refitting CSD is the slow step and it is done once
per specimen; the four SH trackers are then cheap. Seeds, step size, stopping criterion and the
scoring estimator are held identical to the published run throughout, so the only thing that
changes is how a direction is chosen at each step."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, time, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
from scipy import stats
from scipy.spatial import cKDTree
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response_ssst, ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.direction import (ProbabilisticDirectionGetter, DeterministicMaximumDirectionGetter,
                            ClosestPeakDirectionGetter, PTTDirectionGetter)
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from tracto_sensitivity import geometry, enrich, TR, ROOT, SUBJ

OUT = (_env.ROOT + "/surf/three_hinge/validate")
STEP, FA_STOP, MAXANG = 0.4, 0.12, 60.0          # identical to the published run
GETTERS = ["deterministic maximum", "closest peak", "probabilistic", "parallel transport"]


def shcoeff(sid):
    """CSD fODF spherical harmonics, cached - this is the slow step"""
    d = f"{TR}/Subject{sid}"; cache = f"{d}/csd_sh.npy"
    if os.path.exists(cache):
        return np.load(cache)
    img = nib.load(f"{d}/dwi.nii.gz"); data = np.asarray(img.dataobj, np.float32)
    gtab = gradient_table(np.loadtxt(f"{d}/bvals.txt"), bvecs=np.loadtxt(f"{d}/bvecs.txt"))
    mask = np.asarray(nib.load(f"{d}/mask.nii.gz").dataobj) > 0
    resp, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.5)   # same response as track.py
    t0 = time.time()
    fit = ConstrainedSphericalDeconvModel(gtab, resp, sh_order_max=8).fit(data, mask=mask)
    sh = fit.shm_coeff.astype(np.float32)
    np.save(cache, sh)
    print(f"  S{sid}: CSD refit {time.time()-t0:.0f}s -> {cache}", flush=True)
    return sh


def track(sid, name):
    d = f"{TR}/Subject{sid}"
    img = nib.load(f"{d}/dwi.nii.gz"); aff = img.affine
    sh = shcoeff(sid)
    kw = dict(max_angle=MAXANG, sphere=default_sphere)
    dg = {"deterministic maximum": DeterministicMaximumDirectionGetter,
          "closest peak": ClosestPeakDirectionGetter,
          "probabilistic": ProbabilisticDirectionGetter,
          "parallel transport": PTTDirectionGetter}[name].from_shcoeff(sh, **kw)
    fa = np.asarray(nib.load(f"{d}/fa.nii.gz").dataobj)
    gmwm = np.asarray(nib.load(f"{ROOT}/Subject{sid}/Subject{sid}_Mask/DWI_GM_WM_mask.nii.gz").dataobj)
    seeds = utils.seeds_from_mask(gmwm == 2, aff, density=1)
    sl = Streamlines(LocalTracking(dg, ThresholdStoppingCriterion(fa, FA_STOP), seeds, aff,
                                   step_size=STEP, max_cross=1))
    return [np.asarray(x) for x in sl if len(x) > 5], img


def main():
    T = np.full((len(GETTERS), 6), np.nan); S = np.full((len(GETTERS), 6), np.nan)
    N = np.zeros((len(GETTERS), 6), int)
    for j, sid in enumerate(SUBJ):
        patch_w, hinge, crown = geometry(sid)
        for i, name in enumerate(GETTERS):
            t0 = time.time()
            streams, img = track(sid, name)
            T[i, j], S[i, j] = enrich(streams, img, patch_w, hinge, crown)
            N[i, j] = len(streams)
            print(f"S{sid} {name:22s}{len(streams):9d} streamlines  target {T[i,j]:5.2f}  "
                  f"seed {S[i,j]:5.2f}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{OUT}/tracto_algorithms.npz", getters=np.array(GETTERS), target=T, seed=S, n=N)
    print(f"\n{'algorithm':24s}{'streamlines':>12s}{'target contrast':>22s}{'seed contrast':>22s}")
    for i, name in enumerate(GETTERS):
        t, s_ = T[i], S[i]
        pt = stats.ttest_1samp(np.log(t), 0).pvalue
        ps = stats.ttest_1samp(np.log(s_), 0).pvalue
        print(f"{name:24s}{int(np.nanmean(N[i])):12d}"
              f"{np.exp(np.nanmean(np.log(t))):9.2f}x {int((t>1).sum())}/6 P={pt:.4f}"
              f"{np.exp(np.nanmean(np.log(s_))):9.2f}x {int((s_>1).sum())}/6 P={ps:.4f}")
    print(f"wrote {OUT}/tracto_algorithms.npz")


if __name__ == "__main__":
    main()

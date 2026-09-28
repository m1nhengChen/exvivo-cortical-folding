#!/usr/bin/env python3
"""Score the MRtrix iFOD2 tractograms with the published estimator, with and without SIFT2 weights.

Same node set, same 3 mm collection sphere, same 4 mm endpoint tolerance and the same two contrasts
as validate/seed_control.py - only the tractogram changes. Two readings per specimen:

  unweighted   every streamline counts once, as in the DIPY run
  SIFT2        each streamline carries its SIFT2 weight in both numerator and denominator, so the
               reconstructed density matches the FOD lobe amplitudes

The second is the point of running MRtrix at all. FIGS_gyralbias could only argue about whether the
per-node streamline count is a confounder or a mediator; SIFT2 makes the count quantitative instead
of arguing about it, and the seed contrast is then computed on a tractogram whose density is
FOD-consistent by construction."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
from scipy import stats
from scipy.spatial import cKDTree
from tracto_sensitivity import geometry, TR, SUBJ, R, NEAR
OUT = (_env.ROOT + "/surf/three_hinge/validate")


def load(sid):
    w = f"{TR}/Subject{sid}/mrtrix"
    tck = nib.streamlines.load(f"{w}/tracks.tck")
    sl = [np.asarray(x) for x in tck.streamlines]
    # tcksift2 writes a commented MRtrix header (its own command history) and then one long line
    # of whitespace-separated weights; take the last non-comment line rather than filtering tokens
    lines = [ln for ln in open(f"{w}/sift2.txt").read().splitlines() if ln and not ln.startswith("#")]
    wt = np.fromstring(lines[-1], sep=" ")
    assert len(wt) == len(sl), f"S{sid}: {len(wt)} weights for {len(sl)} streamlines"
    img = nib.load(f"{TR}/Subject{sid}/dwi.nii.gz")
    return sl, wt, img


def contrasts(streams, wt, img, patch_w, hinge, crown):
    """(target, seed) contrasts; wt=None gives the unweighted reading"""
    dinv = np.linalg.inv(img.affine); shp = img.shape[:3]
    ep0 = np.array([s[0] for s in streams]); ep1 = np.array([s[-1] for s in streams])
    vox2sl = defaultdict(set)
    for i, s in enumerate(streams):
        for a, b, c in np.round(nib.affines.apply_affine(dinv, s)).astype(int):
            if 0 <= a < shp[0] and 0 <= b < shp[1] and 0 <= c < shp[2]:
                vox2sl[(a, b, c)].add(i)
    rr = int(np.ceil(R)); zoom = float(np.mean(img.header.get_zooms()[:3]))
    off = [(a, b, c) for a in range(-rr, rr + 1) for b in range(-rr, rr + 1)
           for c in range(-rr, rr + 1) if a * a + b * b + c * c <= (R / zoom) ** 2]
    th = cKDTree(patch_w[hinge]); tc = cKDTree(patch_w[crown])
    nh0 = th.query(ep0)[0] < NEAR; nh1 = th.query(ep1)[0] < NEAR
    nc0 = tc.query(ep0)[0] < NEAR; nc1 = tc.query(ep1)[0] < NEAR

    def frac(seeds_, n0, n1):
        vals = []
        for vid in seeds_:
            cij = np.round(nib.affines.apply_affine(dinv, patch_w[vid])).astype(int)
            S = set()
            for o in off:
                S |= vox2sl.get((cij[0] + o[0], cij[1] + o[1], cij[2] + o[2]), set())
            ids = list(S)
            if len(ids) < 3:
                continue
            cw = patch_w[vid]
            far0 = np.linalg.norm(ep0[ids] - cw, axis=1) > np.linalg.norm(ep1[ids] - cw, axis=1)
            hit = np.where(far0, n0[ids], n1[ids]).astype(float)
            if wt is None:
                vals.append(hit.mean())
            else:
                ww = wt[ids]
                vals.append(float(np.sum(hit * ww) / max(ww.sum(), 1e-12)))
        return float(np.nanmean(vals))

    f_hh = frac(hinge, nh0, nh1); f_hc = frac(hinge, nc0, nc1); f_ch = frac(crown, nh0, nh1)
    return f_hh / f_hc, f_hh / f_ch


def main():
    rows = {}
    for sid in SUBJ:
        w = f"{TR}/Subject{sid}/mrtrix"
        if not (os.path.exists(f"{w}/tracks.tck") and os.path.exists(f"{w}/sift2.txt")):
            print(f"  S{sid}: not ready, skipped", flush=True); continue
        patch_w, hinge, crown = geometry(sid)
        sl, wt, img = load(sid)
        u = contrasts(sl, None, img, patch_w, hinge, crown)
        s = contrasts(sl, wt, img, patch_w, hinge, crown)
        rows[sid] = (u, s)
        print(f"S{sid}  {len(sl):,} streamlines | unweighted target {u[0]:.2f} seed {u[1]:.2f}"
              f" | SIFT2 target {s[0]:.2f} seed {s[1]:.2f}", flush=True)
    if not rows:
        return
    ks = sorted(rows)
    U = np.array([rows[k][0] for k in ks]); S = np.array([rows[k][1] for k in ks])
    np.savez(f"{OUT}/mrtrix_check.npz", subj=np.array(ks), unweighted=U, sift2=S)
    print(f"\n{'':22s}{'target contrast':>24s}{'seed contrast':>24s}")
    for nm, A in (("iFOD2, unweighted", U), ("iFOD2 + SIFT2", S)):
        out = []
        for c in range(2):
            v = A[:, c]; p = stats.ttest_1samp(np.log(v), 0).pvalue
            out.append(f"{np.exp(np.mean(np.log(v))):9.2f}x {int((v>1).sum())}/{len(v)} P={p:.4f}")
        print(f"{nm:22s}{out[0]:>24s}{out[1]:>24s}")
    print(f"wrote {OUT}/mrtrix_check.npz")


if __name__ == "__main__":
    main()

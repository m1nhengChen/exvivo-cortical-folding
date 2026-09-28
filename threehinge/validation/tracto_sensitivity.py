#!/usr/bin/env python3
"""Reviewer-proof check: does the inter-hinge enrichment depend on the tracking parameters?

The hinge network is read off deterministic tractography, so the obvious objection is that the
excess is a property of one particular parameter set. Here the tractogram is rebuilt under
several settings and the enrichment recomputed each time with the published estimator.

The CSD fit is NOT redone - peak_dirs / peak_values are reused - so what varies is everything
downstream of the fiber model. peak_indices were not stored, but every peak direction IS a sphere
vertex, so the indices are recovered exactly (max deviation 4e-8) and the direction getter is
rebuilt and _initialize()d. With qa_thr = 0.0239, total_weight = 0.5, ang_thr = 60 this reproduces
the published tractogram to the streamline (321,806 for Subject5), which is the check that the
rebuild is faithful before anything is varied.

  peak cutoff   a peak is dropped unless it reaches CUT x the voxel's leading peak. The stored
                peaks were extracted at relative_peak_threshold = 0.5, so only cutoffs ABOVE 0.5
                can be explored this way; going lower would need the CSD refit.
  peak weight   total_weight, the relative peak threshold applied during tracking itself.
  angle         the maximum turn allowed between steps (ang_thr).
  step          integration step in mm.
  FA stop       the ThresholdStoppingCriterion on FA.
  seed density  seeds per white-matter voxel per axis, so density 2 is 8x the streamlines.
Reported per setting: the target contrast f_hh/f_hc and the coverage-free seed contrast f_hh/f_ch,
with specimen as the unit (n = 6). The claim to defend is that the enrichment stays above 1 in
every setting, not that its magnitude is invariant."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, time, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
from scipy import sparse, stats
from scipy.sparse.csgraph import dijkstra
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree
from dipy.direction.peaks import PeaksAndMetrics
from dipy.data import default_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking import utils
import dk
ROOT = (_env.ROOT + "/28211843"); FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
OUT = (_env.ROOT + "/surf/three_hinge/validate")
H = "rh"; R = 3.0; INWARD = 2.0; NEAR = 4.0
SUBJ = [int(x) for x in os.environ.get("EXVIVO_SUBJ", "1 2 3 4 5 6").split()]

BASE = dict(cut=0.0, ang=60.0, fa=0.12, step=0.4, dens=1, tw=0.5, qa=0.0239)
SETTINGS = [("published", {}),
            ("FOD cutoff 0.65", dict(cut=0.65)),
            ("FOD cutoff 0.80", dict(cut=0.80)),
            ("peak weight 0.30", dict(tw=0.30)),
            ("peak weight 0.70", dict(tw=0.70)),
            ("angle 45", dict(ang=45.0)),
            ("angle 75", dict(ang=75.0)),
            ("step 0.6 mm", dict(step=0.6)),
            ("FA stop 0.15", dict(fa=0.15)),
            ("seed density 2", dict(dens=2))]


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                             shape=(n, n))


def geometry(sid):
    """everything that does not depend on the tracking parameters, built once per specimen"""
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    import pyvista as pv
    m = pv.PolyData(v.astype(float), np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = (tk[:3, :3] @ m.point_data["Normals"].T).T
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    vw = nib.affines.apply_affine(tk, v)
    if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    patch_w = vw - INWARD * nrm
    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
    parc = dk.read(s)[0]; ctx = parc > 0
    hinge = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int)
    hinge = hinge[ctx[hinge]]                                   # drop medial-wall 3HG
    dmin = dijkstra(mesh_graph(v.astype(float), f), indices=hinge.tolist(), min_only=True)
    lo = np.percentile(sulc[ctx], 25)
    poolmask = (sulc <= lo) & (dmin > 6) & ctx
    # Same node construction as tracto/conn_matrix.py and validate/seed_control.py, or this checks
    # the stability of a pipeline nobody runs: 3HG collapsed to one node per gyral junction at
    # 6 mm, one same-parcel crown partner each, and the crown partners kept 6 mm apart too. The
    # earlier version used the RAW landmark list and drew crowns independently per hinge with
    # replacement, which left both classes sharing streamlines between neighbouring nodes.
    lab = fcluster(linkage(patch_w[hinge], "single"), 6.0, "distance")
    hn = np.array([mem[np.argmin(np.linalg.norm(patch_w[mem] - patch_w[mem].mean(0), axis=1))]
                   for mem in (hinge[lab == c] for c in np.unique(lab))])
    rng = np.random.RandomState(sid)
    hm, cm, taken = [], [], []
    for h in hn:
        cand = np.where(poolmask & (parc == parc[h]))[0]
        rng.shuffle(cand)
        for c in cand:
            if not taken or np.min(np.linalg.norm(patch_w[taken] - patch_w[c], axis=1)) >= 6.0:
                hm.append(h); cm.append(c); taken.append(c); break
    print(f"  S{sid}: {len(hinge)} landmarks -> {len(hn)} junctions -> {len(hm)} paired nodes",
          flush=True)
    return patch_w, np.array(hm), np.array(cm)


def track(sid, cfg):
    d = f"{TR}/Subject{sid}"
    img = nib.load(f"{d}/dwi.nii.gz"); aff = img.affine
    pd = np.load(f"{d}/peak_dirs.npy").astype(np.float64)
    pv_ = np.load(f"{d}/peak_vals.npy").astype(np.float64)
    V = default_sphere.vertices
    nz = pv_ > 0
    idx = np.full(pv_.shape, -1, dtype=np.int64)                 # -1 = no peak, as dipy expects
    idx[nz] = np.argmax(np.abs(pd[nz] @ V.T), axis=1)            # exact: peaks ARE sphere vertices
    if cfg["cut"] > 0:                                           # post-hoc FOD cutoff
        lead = pv_.max(-1, keepdims=True)
        kill = pv_ < cfg["cut"] * np.maximum(lead, 1e-9)
        pv_ = np.where(kill, 0.0, pv_); idx = np.where(kill, -1, idx)
        pd = np.where(kill[..., None], 0.0, pd)
    p = PeaksAndMetrics()
    p.sphere = default_sphere
    p.peak_dirs, p.peak_values, p.peak_indices = pd, pv_, idx
    p.ang_thr, p.qa_thr, p.total_weight = cfg["ang"], cfg["qa"], cfg["tw"]
    p._initialize()                                              # builds _qa / _ind / _odf_vertices
    fa = np.asarray(nib.load(f"{d}/fa.nii.gz").dataobj)
    gmwm = np.asarray(nib.load(f"{ROOT}/Subject{sid}/Subject{sid}_Mask/DWI_GM_WM_mask.nii.gz").dataobj)
    seeds = utils.seeds_from_mask(gmwm == 2, aff, density=cfg["dens"])
    sl = LocalTracking(p, ThresholdStoppingCriterion(fa, cfg["fa"]), seeds, aff,
                       step_size=cfg["step"], max_cross=1)
    sl = Streamlines(sl)
    return [np.asarray(x) for x in sl if len(x) > 5], img


def enrich(streams, img, patch_w, hinge, crown):
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
            vals.append(np.mean(np.where(far0, n0[ids], n1[ids])))
        return np.nanmean(vals)

    f_hh = frac(hinge, nh0, nh1); f_hc = frac(hinge, nc0, nc1); f_ch = frac(crown, nh0, nh1)
    return f_hh / f_hc, f_hh / f_ch


def main():
    T = np.full((len(SETTINGS), 6), np.nan); S = np.full((len(SETTINGS), 6), np.nan)
    NS = np.zeros((len(SETTINGS), 6), int)
    for j, sid in enumerate(SUBJ):
        patch_w, hinge, crown = geometry(sid)
        for i, (name, over) in enumerate(SETTINGS):
            cfg = dict(BASE, **over); t0 = time.time()
            streams, img = track(sid, cfg)
            T[i, j], S[i, j] = enrich(streams, img, patch_w, hinge, crown)
            NS[i, j] = len(streams)
            print(f"S{sid} {name:16s} {len(streams):8d} streamlines  "
                  f"target {T[i, j]:5.2f}  seed {S[i, j]:5.2f}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f"{OUT}/tracto_sensitivity.npz", target=T, seed=S, nstream=NS,
             names=np.array([n for n, _ in SETTINGS]), subj=np.array(SUBJ))
    print(f"\n{'setting':18s}{'streamlines':>13s}{'target contrast':>18s}{'seed contrast':>18s}")
    for i, (name, _) in enumerate(SETTINGS):
        t, s = T[i][np.isfinite(T[i])], S[i][np.isfinite(S[i])]
        pt = stats.ttest_1samp(np.log(t), 0).pvalue if len(t) > 1 else np.nan
        ps = stats.ttest_1samp(np.log(s), 0).pvalue if len(s) > 1 else np.nan
        print(f"{name:18s}{int(np.nanmean(NS[i])):13d}"
              f"{np.exp(np.log(t).mean()):9.2f}x {int((t > 1).sum())}/{len(t)} P={pt:.4f}"
              f"{np.exp(np.log(s).mean()):9.2f}x {int((s > 1).sum())}/{len(s)} P={ps:.4f}")


if __name__ == "__main__":
    main()

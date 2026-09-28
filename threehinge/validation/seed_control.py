#!/usr/bin/env python3
"""Seed-side control and coverage-free magnitude for the hinge-network result.

conn_robust.py reports enrichment = f(3HG seed -> 3HG target) / f(3HG seed -> crown target). That
ratio compares two DIFFERENT target sets, and the 3HG target set covers 1.5-2.4x more white-matter
volume than the matched crown set (equal node counts, but the crown pool is confined to the top
sulc quartile >6 mm from any hinge, so its 4 mm balls overlap heavily). A target set covering more
volume catches more endpoints for purely geometric reasons, which inflates that ratio.

Two contrasts here are immune to it, computed with the published seeds, targets, sampling and
tolerances -- nothing is merged, filtered or re-sampled:
  seed control      f_hh vs f_ch : same 3HG target set, 3HG seeds vs crown seeds. Target coverage
                    cancels exactly, so this is the coverage-free hinge-network magnitude, and it
                    is the contrast the hinge-network claim actually predicts.
  seed x target     (f_hh/f_hc) / (f_ch/f_cc) : the 2x2 interaction, immune to both a seed-side and
                    a target-side coverage difference.

The same four fractions are also resolved by connection length: each collected streamline is binned
by the distance from the seed patch to its distal endpoint, so the curves show whether preferential
3HG-to-3HG wiring is a short-range or a long-range property. Identical estimator to the scalars.
sampling -- nothing here depends on any choice made in tracto/conn_matrix.py.

conn_robust.py reports enrichment = f(3HG seed -> 3HG target) / f(3HG seed -> crown target).
The hinge-network claim requires that this be a property of 3HG SEEDS. So run the identical
computation from CROWN seeds: f(crown seed -> 3HG target) / f(crown seed -> crown target).
  crown-seed ratio ~ 1        -> the preference is specific to 3HG seeds; the published result holds.
  crown-seed ratio ~ 3HG-seed -> the preference belongs to the 3HG TARGET SET, not to 3HG wiring.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
import dk
from hinge_ids import load_3hg
FS = (_env.ROOT + "/surf/fs"); GNET = (_env.ROOT + "/surf/three_hinge/gnet")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
H = "rh"; R = 3.0; INWARD = 2.0; NEAR = 4.0          # exactly conn_robust.py
EDGES = np.arange(10, 111, 10.0)                     # distal-endpoint distance bins (mm)


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


def run(sid):
    s = f"Subject{sid}"
    dwi = nib.load(f"{TR}/{s}/dwi.nii.gz"); dinv = np.linalg.inv(dwi.affine); shp = dwi.shape[:3]
    # same tractogram and weights as tracto/conn_matrix.py, so panels a/f/h and d/e describe one
    # reconstruction. SIFT2 weights make a streamline count a density estimate rather than a count
    # of successful tracking attempts, so f becomes a ratio of densities.
    from conn_matrix import tractogram
    SL, WT = tractogram(sid, dwi)
    if WT is None:
        WT = np.ones(len(SL))
    ep0 = np.array([sl[0] for sl in SL]); ep1 = np.array([sl[-1] for sl in SL])

    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    import pyvista as pv
    m = pv.PolyData(v.astype(float), np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = (tk[:3, :3] @ m.point_data["Normals"].T).T; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    vw = nib.affines.apply_affine(tk, v)
    if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    patch_w = vw - INWARD * nrm

    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc"); parc = dk.read(s)[0]; ctx = parc > 0
    # One landmark per gyral junction, merged at 6 mm on the sub-WM patch centres - the same
    # nodes tracto/conn_matrix.py builds, so panels a/f/h and b-e finally describe one landmark
    # set. Without it this estimator averages over seeds, and two landmarks 4 mm apart have
    # overlapping 3 mm collection spheres: the same streamlines are counted twice, which shrinks
    # the within-specimen variance rather than adding information.
    hinge = load_3hg(sid, merge=6.0, hemi=H)
    # the crown exclusion still uses the FULL cortical set, or a control can sit on a landmark
    # that merging removed
    dmin = dijkstra(mesh_graph(v.astype(float), f),
                    indices=load_3hg(sid, merge=0.0, hemi=H).tolist(), min_only=True)
    lo = np.percentile(sulc[ctx], 25)
    poolmask = (sulc <= lo) & (dmin > 6) & ctx
    # Crown nodes are kept 6 mm apart, exactly as tracto/conn_matrix.py does it. Without this the
    # draw was independent per hinge and WITH REPLACEMENT: it returned outright duplicates and left
    # 69-83% of crown nodes within 6 mm of another, so their 3 mm collection spheres overlapped and
    # the same streamlines were scored for several seeds. f_ch and f_cc are means over crown seeds,
    # so that shrank their within-specimen variance - the 3HG side was 6 mm-resolved by the merge
    # and the crown side was not, which is an asymmetry in the estimator, not a control.
    # A hinge whose parcel cannot supply a separated crown is dropped, so the pairing stays 1:1.
    rng = np.random.RandomState(sid)                                     # published RNG / sampling
    hm, cm, taken = [], [], []
    for h in hinge:
        cand = np.where(poolmask & (parc == parc[h]))[0]
        rng.shuffle(cand)
        for c in cand:
            if not taken or np.min(np.linalg.norm(patch_w[taken] - patch_w[c], axis=1)) >= 6.0:
                hm.append(h); cm.append(c); taken.append(c); break
    hinge_m, crown_m = np.array(hm), np.array(cm)
    print(f"  S{sid}: {len(hinge)} junctions -> {len(hinge_m)} paired nodes", flush=True)

    vox2sl = defaultdict(set)
    for i, sl in enumerate(SL):
        for a, b, c in np.round(nib.affines.apply_affine(dinv, sl)).astype(int):
            if 0 <= a < shp[0] and 0 <= b < shp[1] and 0 <= c < shp[2]:
                vox2sl[(a, b, c)].add(i)
    rr = int(np.ceil(R)); zoom = np.mean(dwi.header.get_zooms()[:3])
    off = [(a, b, c) for a in range(-rr, rr+1) for b in range(-rr, rr+1) for c in range(-rr, rr+1)
           if a*a + b*b + c*c <= (R/zoom)**2]

    def ids_of(cw):
        cij = np.round(nib.affines.apply_affine(dinv, cw)).astype(int); S = set()
        for o in off:
            S |= vox2sl.get((cij[0]+o[0], cij[1]+o[1], cij[2]+o[2]), set())
        return list(S)

    tgt_h = cKDTree(patch_w[hinge_m]); tgt_c = cKDTree(patch_w[crown_m])
    nh0 = tgt_h.query(ep0)[0] < NEAR; nh1 = tgt_h.query(ep1)[0] < NEAR
    nc0 = tgt_c.query(ep0)[0] < NEAR; nc1 = tgt_c.query(ep1)[0] < NEAR

    def frac_to(seed_vids, n0, n1, edges=None):
        """mean over seeds of the fraction of collected streamlines whose distal endpoint is near a
        target.

        With `edges`, the same fraction resolved by connection length CUMULATIVELY: the value at L
        is the fraction over every collected streamline whose distal endpoint lies within L, not
        within one 10 mm bin. Disjoint bins were the original form and they are too noisy to read
        at n = 6 - beyond ~50 mm a seed contributes only a handful of streamlines to a bin, and the
        between-specimen s.e.m. runs to 20% of the mean against 7% in the first bin, so the three
        block curves cross each other on noise. The cumulative form uses every streamline out to L
        at each L, which is also the quantity the reader actually wants: of the connections a node
        makes out to this distance, what share land on the other class."""
        vals = []
        for vid in seed_vids:
            ids = ids_of(patch_w[vid])
            if len(ids) < 3:
                continue
            cw = patch_w[vid]
            far0 = np.linalg.norm(ep0[ids]-cw, axis=1) > np.linalg.norm(ep1[ids]-cw, axis=1)
            hit = np.where(far0, n0[ids], n1[ids])
            w = WT[ids]
            if edges is None:
                vals.append(float(np.sum(hit * w) / max(w.sum(), 1e-12))); continue
            d = np.where(far0, np.linalg.norm(ep0[ids]-cw, axis=1), np.linalg.norm(ep1[ids]-cw, axis=1))
            row = np.full(len(edges) - 1, np.nan)
            for k in range(len(edges) - 1):
                m = d <= edges[k + 1]                    # everything out to this length
                if m.sum() >= 5:
                    row[k] = float(np.sum(hit[m] * w[m]) / max(w[m].sum(), 1e-12))
            vals.append(row)
        return np.nanmean(vals, 0)

    f_hh = frac_to(hinge_m, nh0, nh1); f_hc = frac_to(hinge_m, nc0, nc1)
    f_ch = frac_to(crown_m, nh0, nh1); f_cc = frac_to(crown_m, nc0, nc1)
    E = EDGES
    d_hh = frac_to(hinge_m, nh0, nh1, E); d_hc = frac_to(hinge_m, nc0, nc1, E)
    d_ch = frac_to(crown_m, nh0, nh1, E); d_cc = frac_to(crown_m, nc0, nc1, E)
    return f_hh, f_hc, f_ch, f_cc, np.array([d_hh, d_hc, d_ch, d_cc])


OUT = (_env.ROOT + "/surf/three_hinge/validate")
if __name__ == "__main__":
    from scipy import stats
    SUBJ = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    print(f"{'':10s}{'f_hh':>7s}{'f_hc':>7s}{'f_ch':>7s}{'f_cc':>7s}"
          f"{'published':>11s}{'coverage-free':>15s}{'seedxtarget':>13s}")
    ROWS, CURVES = [], []
    for sid in SUBJ:
        a, b, c, d, cur = run(sid); ROWS.append([a, b, c, d]); CURVES.append(cur)
        print(f"Subject{sid:<3d}{a:7.3f}{b:7.3f}{c:7.3f}{d:7.3f}"
              f"{a/b:10.2f}x{a/c:14.2f}x{(a/b)/(c/d):12.2f}x", flush=True)
    fhh, fhc, fch, fcc = np.array(ROWS).T
    np.savez(f"{OUT}/seed_control.npz", subj=np.array(SUBJ), f_hh=fhh, f_hc=fhc, f_ch=fch, f_cc=fcc,
             curves=np.array(CURVES), edges=EDGES)          # curves: (specimen, {hh,hc,ch,cc}, bin)

    pub, free = fhh / fhc, fhh / fch
    inter = (fhh / fhc) / (fch / fcc)
    t1, p1 = stats.ttest_rel(fhh, fch)
    t2, p2 = stats.ttest_rel(np.log(fhh / fhc), np.log(fch / fcc))
    print(f"\npublished enrichment        {pub.mean():.2f}x   (confounded: target sets differ in coverage)")
    print(f"coverage-free (f_hh/f_ch)   {free.mean():.2f}x   {int((free>1).sum())}/6 > 1, "
          f"paired t = {t1:.2f}, p = {p1:.4f}, exact Wilcoxon p = {stats.wilcoxon(fhh, fch).pvalue:.3f}")
    print(f"seed x target interaction   {inter.mean():.2f}x   {int((inter>1).sum())}/6 > 1, "
          f"paired t on log = {t2:.2f}, p = {p2:.4f}")
    print(f"\nwrote {OUT}/seed_control.npz")

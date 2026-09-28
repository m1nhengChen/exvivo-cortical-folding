#!/usr/bin/env python3
"""Node x node structural connectivity matrix over 3HG and matched gyral-crown patches.

Nodes are sub-WM patches 2 mm beneath the white surface. 3HG landmark vertices closer than
MERGE mm are collapsed into one node (they are unresolvable at the 3 mm patch / 4 mm endpoint
tolerance, and leaving them split would inflate the 3HG-3HG block with trivially short
neighbour-to-neighbour streamlines). Every 3HG node is then matched by a gyral-crown node drawn
from the SAME DK parcel, with the same minimum node separation, giving equal node counts and
parcel composition in the two blocks. Edges reproduce the published hinge-network statistic, decomposed by target. A streamline is
collected by node i if it PASSES WITHIN R mm of i's patch centre (3HG patches sit in sub-WM, where
streamlines pass through rather than terminate). Its DISTAL endpoint - the end farther from i - is
then matched to any node j within NEAR mm, giving M[i,j]. Row-normalizing by the number of
streamlines through i turns each row into "fraction of node i's streamlines terminating at j", so
the 3HG-row block sums are exactly the f_hh / f_hc of validate/conn_robust.py. The matrix is
asymmetric by construction and is symmetrized as (M + M.T)/2 for display; both are saved. The node
order [3HG..., crown...] makes the three blocks read directly off it: 3HG-3HG (upper-left),
3HG-crown (off-diagonal), crown-crown (lower-right)."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
import dk
ROOT = (_env.ROOT + "/28211843"); FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
H = "rh"; INWARD = 2.0; R = 3.0; NEAR = 4.0
MERGE = float(os.environ.get("MERGE", 6.0))       # min node separation (patch spheres just touch)
MINLEN = float(os.environ.get("MINLEN", 0.0))     # streamline length floor; 0 = published behaviour
# Which tractogram. "mrtrix" is MRtrix3 iFOD2 with SIFT2 weights - probabilistic, second-order
# integration over the FOD, and each streamline carrying the weight that makes reconstructed
# streamline density match the FOD lobe amplitudes. That last part is why it is the primary
# pipeline: it is the only one of the two in which a streamline count is a quantitative estimate
# of fibre density rather than a count of tracking attempts that happened to succeed. "dipy" is the
# earlier deterministic run on discrete CSD peaks, kept for the supplementary comparison.
SRC = os.environ.get("SRC", "dipy")


def tractogram(sid, dwi):
    """(streamlines, per-streamline weight) for the selected source; weights are 1 for dipy"""
    s = f"Subject{sid}"
    if SRC == "dipy":
        trk = load_tractogram(f"{TR}/{s}/whole_brain.trk", dwi, bbox_valid_check=False)
        trk.to_space(Space.RASMM)
        return list(trk.streamlines), None
    w = f"{TR}/{s}/mrtrix"
    sl = list(nib.streamlines.load(f"{w}/tracks.tck").streamlines)
    lines = [ln for ln in open(f"{w}/sift2.txt").read().splitlines()
             if ln and not ln.startswith("#")]
    wt = np.fromstring(lines[-1], sep=" ")
    assert len(wt) == len(sl), f"{s}: {len(wt)} SIFT2 weights for {len(sl)} streamlines"
    return sl, wt


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


def run(sid):
    s = f"Subject{sid}"
    dwi = nib.load(f"{TR}/{s}/dwi.nii.gz")
    SL, WT = tractogram(sid, dwi)
    if WT is None:
        WT = np.ones(len(SL))
    ep0 = np.array([sl[0] for sl in SL]); ep1 = np.array([sl[-1] for sl in SL])
    length = np.array([np.linalg.norm(np.diff(sl, axis=0), axis=1).sum() for sl in SL])

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
    patch_w = vw - INWARD * nrm                                    # sub-WM patch centres

    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc"); parc = dk.read(s)[0]; ctx = parc > 0
    hinge = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int); hinge = hinge[parc[hinge] > 0]
    dmin = dijkstra(mesh_graph(v.astype(float), f), indices=hinge.tolist(), min_only=True)

    # ---- collapse 3HG landmarks that the patch geometry cannot resolve; keep the medoid vertex ----
    lab = fcluster(linkage(patch_w[hinge], "single"), MERGE, "distance")
    hnodes = []
    for c in np.unique(lab):
        mem = hinge[lab == c]
        hnodes.append(mem[np.argmin(np.linalg.norm(patch_w[mem] - patch_w[mem].mean(0), axis=1))])
    hnodes = np.array(hnodes)

    # ---- same-parcel crown match, one per 3HG node, same MERGE separation ----
    lo = np.percentile(sulc[ctx], 25)
    pool = (sulc <= lo) & (dmin > 6) & ctx
    rng = np.random.RandomState(sid)
    cnodes, taken, paired = [], [], []
    for i, h in enumerate(hnodes):
        cand = np.where(pool & (parc == parc[h]))[0]
        rng.shuffle(cand)
        for c in cand:                       # keep crown nodes MERGE-separated, like the 3HG nodes
            if not taken or np.min(np.linalg.norm(patch_w[taken] - patch_w[c], axis=1)) >= MERGE:
                cnodes.append(c); taken.append(c); paired.append(i); break
    # a 3HG node whose parcel cannot supply a crown is dropped, so node i and node i+n stay a
    # genuine same-parcel pair -- the paired label-swap null depends on this alignment
    hnodes = hnodes[paired]; cnodes = np.array(cnodes)
    nodes = np.r_[hnodes, cnodes]
    kind = np.r_[np.zeros(len(hnodes), int), np.ones(len(cnodes), int)]   # 0 = 3HG, 1 = crown
    P = patch_w[nodes]

    # ---- which streamlines pass through which patch (binary incidence) ----
    n = len(nodes); tree = cKDTree(P)
    keep_sl = np.where(length > MINLEN)[0]
    pts = np.concatenate([SL[i] for i in keep_sl])
    owner = np.repeat(keep_sl, [len(SL[i]) for i in keep_sl])
    hits = tree.query_ball_point(pts, R)
    rows = np.concatenate([np.full(len(h), o) for o, h in zip(owner, hits) if h])
    cols = np.concatenate([h for h in hits if h])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows.astype(int), cols.astype(int))),
                          shape=(len(SL), n))
    A.sum_duplicates(); A.data[:] = 1.0

    # ---- distal endpoint of each collected streamline -> target nodes within NEAR ----
    e0 = tree.query_ball_point(ep0, NEAR); e1 = tree.query_ball_point(ep1, NEAR)
    M = np.zeros((n, n)); nseed = np.zeros(n)
    A = A.tocsr()
    # Every streamline enters M and nseed with its SIFT2 weight, so F is a ratio of weighted
    # densities rather than of tracking-attempt counts. Binary connectivity is unaffected: a pair
    # is connected if any streamline joins it, whatever that streamline weighs.
    Mb = np.zeros((n, n))                                          # unweighted, for the binary A
    for sl_id in np.unique(rows).astype(int):
        seeds = A.indices[A.indptr[sl_id]:A.indptr[sl_id + 1]]
        wsl = WT[sl_id]
        nseed[seeds] += wsl
        d0 = np.linalg.norm(P[seeds] - ep0[sl_id], axis=1)
        d1 = np.linalg.norm(P[seeds] - ep1[sl_id], axis=1)
        for tg, sel in ((e0[sl_id], d0 > d1), (e1[sl_id], d1 >= d0)):
            if len(tg) and sel.any():
                M[np.ix_(seeds[sel], tg)] += wsl
                Mb[np.ix_(seeds[sel], tg)] += 1.0
    np.fill_diagonal(Mb, 0.0)
    np.fill_diagonal(M, 0.0)
    F = M / np.maximum(nseed, 1)[:, None]                          # row = fraction of node i's streamlines

    # ---- chance baseline per target class ----------------------------------------------------
    # The 3HG and crown target sets have equal node counts but need not cover equal WM volume, and
    # a target set covering more volume catches more endpoints for purely geometric reasons. The
    # chance level is the fraction of ALL long streamlines with an endpoint within NEAR of the set;
    # dividing each observed fraction by it makes the two classes comparable.
    hh, cc = kind == 0, kind == 1
    near = lambda cls: np.mean([len(set(e0[i]) & set(np.where(cls)[0])) > 0 or
                                len(set(e1[i]) & set(np.where(cls)[0])) > 0 for i in keep_sl])
    idx_h, idx_c = set(np.where(hh)[0]), set(np.where(cc)[0])
    ch = np.mean([bool(idx_h & set(e0[i])) or bool(idx_h & set(e1[i])) for i in keep_sl])
    cc_ = np.mean([bool(idx_c & set(e0[i])) or bool(idx_c & set(e1[i])) for i in keep_sl])
    chance = np.array([ch, cc_])

    # Mb is the unweighted count matrix; downstream code that wants BINARY connectivity must use
    # it, because a SIFT2 weight can be arbitrarily small and (M > 0) would then depend on floating
    # point rather than on whether a streamline exists.
    np.savez(f"{TR}/{s}/conn_matrix.npz", M=M, Mb=Mb, F=F, nseed=nseed, chance=chance, kind=kind,
             parc=parc[nodes], pos=P,
             vert=nodes, nodedist=np.linalg.norm(P[:, None] - P[None], axis=-1),
             params=np.array([INWARD, R, NEAR, MERGE, MINLEN]))
    B = lambda a, b: F[np.ix_(a, b)].sum(1).mean()          # per-seed fraction reaching that class
    raw = B(hh, hh) / B(hh, cc)
    print(f"{s}: {hh.sum()} 3HG + {cc.sum()} crown nodes | nnz {int((M>0).sum())} | "
          f"f_hh={B(hh,hh):.3f} f_hc={B(hh,cc):.3f} enrich_raw={raw:.2f}x | "
          f"chance 3HG={ch:.3f} crown={cc_:.3f} (ratio {ch/cc_:.2f}x) -> "
          f"enrich_corrected={raw/(ch/cc_):.2f}x", flush=True)


if __name__ == "__main__":
    for i in ([int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]):
        run(i)

#!/usr/bin/env python3
"""One place that decides which vertices are 3HG.

Two operations, both of which every analysis should agree on:

  cortex mask   GyralNet runs on the whole surface and reports junctions on the medial wall,
                where there is no folding to be a junction of.
  merge         landmarks closer than MERGE mm are one junction as far as any measurement here
                is concerned. The clustering runs on the sub-white-matter PATCH CENTRES, 2 mm
                inside the white surface - not on the surface vertices. Two points on opposite
                banks of a sulcus are far apart along the surface and close underneath it, and
                it is the tissue underneath that a 3 mm patch or a 2 mm disk actually samples.
                Single linkage, medoid vertex kept, exactly as tracto/conn_matrix.py does it.

    from hinge_ids import load_3hg, cluster_groups, group_mean, mirror_groups
    hv = load_3hg(5)              # cortex-masked and merged at 6 mm
    hv = load_3hg(5, merge=0)     # cortex-masked only

Why measurements are aggregated per cluster rather than per landmark. Half the landmarks are
isolated, but the rest sit in groups of up to 42 whose 2 mm measuring disks overlap almost
completely. Treating those as independent observations shrinks the within-specimen variance and
inflates Cohen's d by about a third: measured on the six specimens, MD gives d = -0.245 over all
451 landmarks, -0.166 over the ~180 medoids, and -0.181 when every landmark is used but averaged
within its cluster. The third number is the honest one - nothing is discarded, and no patch of
tissue is counted twice. Averaging only the 3HG side would shrink its variance and inflate d for
a different reason, so the control side is grouped to the same sizes with mirror_groups().
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np, nibabel as nib, sys
from scipy.cluster.hierarchy import fcluster, linkage
import dk

FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
INWARD = 2.0
MERGE = 6.0
# Two merge scales, and the split is NOT arbitrary for one of them and IS inherited for the other.
#
# DISK = 4.0 is the non-overlap distance of the 2 mm geodesic disk that hinge_profile.py and
# fig_fiberconv.py average over: two disks centred 4 mm apart just touch, so below that the same
# tissue enters twice. Derived, not tuned, and writable into the methods without having seen a
# result.
#
# MERGE = 6.0 is the same argument for the connectivity estimator's 3 mm sub-white-matter sphere.
#
# region_dk.py and depth/depth_profile.py sample PER VERTEX, so neither number follows from their
# support - the pseudo-replication there is "one junction contributes several vertices", whose
# natural scale is the junction's own extent, which is what the merge is trying to define in the
# first place. They use MERGE by convention. That is a real gap in the justification, not a
# decision: see the ledger entry on the merge column.
DISK = 4.0


_PC = {}


def patch_centres(sid, hemi="rh"):
    """world coordinates 2 mm inside the white surface, along the outward normal"""
    if (sid, hemi) in _PC:                 # called once per parcel per depth otherwise
        return _PC[(sid, hemi)]
    import pyvista as pv
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{hemi}.white")
    m = pv.PolyData(v.astype(float), np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = (tk[:3, :3] @ m.point_data["Normals"].T).T
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    vw = nib.affines.apply_affine(tk, v)
    if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    _PC[(sid, hemi)] = vw - INWARD * nrm
    return _PC[(sid, hemi)]


def load_3hg(sid, merge=MERGE, hemi="rh", verbose=False):
    s = f"Subject{sid}"
    hv = np.atleast_1d(np.loadtxt(f"{GNET}/{s}/out/{hemi}_3hinge_ids.txt", dtype=int))
    raw = len(hv)
    hv = hv[dk.read(s)[0][hv] > 0]
    cortical = len(hv)
    if merge and merge > 0:
        P = patch_centres(sid, hemi)
        lab = fcluster(linkage(P[hv], "single"), merge, "distance")
        hv = np.array([mem[np.argmin(np.linalg.norm(P[mem] - P[mem].mean(0), axis=1))]
                       for mem in (hv[lab == c] for c in np.unique(lab))])
    if verbose:
        print(f"  {s}: {raw} raw -> {cortical} cortical -> {len(hv)} at merge={merge}")
    return np.sort(hv)


def cluster_groups(sid, ids, merge=MERGE, hemi="rh"):
    """single-linkage groups of `ids` on the patch centres; index arrays INTO `ids`"""
    ids = np.asarray(ids)
    if not merge or merge <= 0 or len(ids) < 2:
        return [np.array([i]) for i in range(len(ids))]
    P = patch_centres(sid, hemi)
    lab = fcluster(linkage(P[ids], "single"), merge, "distance")
    return [np.where(lab == c)[0] for c in np.unique(lab)]


def group_mean(values, groups):
    """one value per group; NaN-safe, so a landmark with no data does not kill its cluster"""
    v = np.asarray(values, float)
    return np.array([np.nanmean(v[g]) if np.isfinite(v[g]).any() else np.nan for g in groups])


def spread_sample(coords, pool, k, min_sep=MERGE, seed=0):
    """Draw up to k points from `pool` with no two closer than `min_sep`, greedily.

    The obvious subsample - pool[np.linspace(0, len(pool)-1, k)] - takes points at even intervals
    of VERTEX INDEX, and FreeSurfer numbering has strong spatial locality, so it returns runs of
    neighbouring vertices. Measured on the gyral-crown pool that gives a median nearest-neighbour
    distance of 0.8-1.4 mm with 81-95% of points within 2 mm of another, against 4.4-5.1 mm and
    0-2% for the 3HG set. The control was therefore far more pseudo-replicated than the landmarks
    it is compared with, which shrinks its variance and inflates Cohen's d through the pooled SD.
    This enforces on the control the same spacing the 6 mm merge enforces on the landmarks."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(np.asarray(pool))
    kept, pts = [], np.empty((0, 3))
    for v in order:
        if len(kept) >= k:
            break
        p = coords[v]
        if not len(kept) or np.min(np.linalg.norm(pts - p, axis=1)) >= min_sep:
            kept.append(int(v)); pts = np.vstack([pts, p])
    return np.sort(np.array(kept, int))


def matched_spread(coords, pool, pool_val, target_val, min_sep=MERGE, nbin=10, seed=0):
    """Minimum-separation draw from `pool` whose `pool_val` histogram matches `target_val`.

    Spacing alone is not enough. Enforcing min_sep on the gyral-crown pool thins dense regions
    hardest, and the densest regions are the most gyral, so the surviving control drifts to
    sulc -3.5 against -5.3 for the landmarks - reopening the fold-depth confound that restricting
    the pool to the most-gyral quartile was there to close. Sampling per stratum keeps both
    properties: the control is spaced like the landmarks AND sits at their depth."""
    edges = np.quantile(target_val, np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pool = np.asarray(pool)
    out = []
    for b in range(nbin):
        want = int(((target_val > edges[b]) & (target_val <= edges[b + 1])).sum())
        cand = pool[(pool_val > edges[b]) & (pool_val <= edges[b + 1])]
        if want and len(cand):
            out.append(spread_sample(coords, cand, want, min_sep, seed=seed + b))
    return np.sort(np.concatenate(out)) if out else np.array([], int)


def mirror_groups(n, sizes):
    """Partition range(n) into blocks with the given sizes, so a control set is averaged exactly
    as much as the landmark set it is compared with.

    The size pattern CYCLES until all n are covered. Stopping after one pass silently truncates
    the control to sum(sizes) entries, which is fine when the control is count-matched to the
    landmarks and catastrophic when it is a pool: in the within-parcel depth contrast the crown
    pool holds hundreds of vertices against a few dozen landmarks, and one pass reduced the crown
    mean to the lowest-index corner of the parcel."""
    if not len(sizes):
        return [np.arange(n)]
    out, i, j = [], 0, 0
    while i < n:
        k = max(int(sizes[j % len(sizes)]), 1)
        out.append(np.arange(i, min(i + k, n)))
        i += k; j += 1
    return out


if __name__ == "__main__":
    for i in range(1, 7):
        load_3hg(i, verbose=True)

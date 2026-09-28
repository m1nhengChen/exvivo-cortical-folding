#!/usr/bin/env python3
"""In-vivo validation, experiment 2: the folding-degree ladder (crest -> 3HG -> 4HG+) in AABC.

hinge_order.py transplanted to the 163842 mesh and the exp-1 sampler:
  rungs      3HG = skeleton degree 3;  4HG+ = degree >= 4;
             crest = degree < 3, most-gyral sulc quartile, > FAR_MM geodesic from any junction.
  matching   every rung resampled to a common sulc histogram (deciles of the 3HG distribution,
             per-stratum quota = min availability), NREP draws averaged. |curvature| is NOT
             matched - local curvature is what makes a hinge a hinge (ex-vivo rationale kept).
  measure    FA/MD/AD/RD geodesic-disk means at mid-thickness, disk computed once per candidate.
  summary    per subject: rung means z-scored against the subject's pooled landmark spread
             (never across rung means - that manufactures P values), then the end-to-end
             contrast 4HG+ - crest (the (-1,0,+1) contrast; tests endpoint separation,
             not linearity).

Usage: invivo_ladder.py SESSION [...]        writes DERIV/<ses>/ladder.npz
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invivo_pipeline import (ROOT, DERIV, METS, DISK_MM, mesh_graph, trilinear)

FAR_MM = 6.0                     # ex-vivo hinge_order value (crest vs junction clearance)
NBIN, NREP, POOL = 10, 50, 1200  # ex-vivo hinge_order values
RUNGS = ["crest", "3HG", "4HG+"]


def skeleton_degree(path, nv):
    import pyvista as pv
    m = pv.read(path)
    assert m.n_points == nv, f"skeleton {m.n_points} != surface {nv}"
    L = np.asarray(m.lines); adj = [set() for _ in range(nv)]
    i = 0
    while i < len(L):
        k = int(L[i]); seg = L[i + 1:i + 1 + k]
        for j in range(k - 1):
            a, b = int(seg[j]), int(seg[j + 1])
            adj[a].add(b); adj[b].add(a)
        i += k + 1
    return np.array([len(s) for s in adj])


def strata(sulc, tgt):
    e = np.quantile(sulc[tgt], np.linspace(0, 1, NBIN + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def hemi_rungs(S, ses, h):
    """rung vertex sets + geometry + sampler inputs on the 163842 mesh, one hemisphere"""
    import pyvista as pv
    from scipy.spatial import cKDTree
    G = f"{S}/postprocessing_tong/gii_native"
    HL = {"lh": "L", "rh": "R"}[h]
    white = pv.read(f"{G}/surf/{h}.white.resample.163842.vtk")
    feat = pv.read(f"{G}/gyralnet_island_resample_163842_scanner/{h}_surf_feature_updated.vtk")
    sulc = np.asarray(feat.point_data["sulc"])
    lab, _, names = nib.freesurfer.read_annot(f"{G}/label/{h}.aparc.resample.163842.annot")
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    bad = {i for i, n in enumerate(names) if n in ("unknown", "corpuscallosum")}
    ctx = ~np.isin(lab, list(bad)) & (lab >= 0)
    deg = skeleton_degree(
        f"{G}/gyralnet_island_resample_163842_scanner/{h}_surf_skelenton_allpoints_final_refine.vtk",
        white.n_points)

    # gyral direction self-calibrated from where the junctions sit (see invivo_pipeline)
    jn = (deg >= 3) & ctx
    sgn = sulc * np.sign(np.median(sulc[jn]))          # larger = more gyral
    g = mesh_graph(white)
    dmin = dijkstra(g, indices=np.where(jn)[0], min_only=True)
    hi = np.quantile(sgn[ctx], 0.75)
    sets = {"3HG": np.where((deg == 3) & ctx)[0],
            "4HG+": np.where((deg >= 4) & ctx)[0],
            "crest": np.where((deg < 3) & ctx & (sgn >= hi) & (dmin > FAR_MM))[0]}

    nat_w = nib.load(f"{S}/sMRI/T1w/Native/{ses}.{HL}.white.native.surf.gii").darrays[0].data
    nat_m = nib.load(f"{S}/sMRI/T1w/Native/{ses}.{HL}.midthickness.native.surf.gii").darrays[0].data
    _, nn = cKDTree(nat_w).query(np.asarray(white.points), workers=4)
    return sets, sulc, ctx, g, nat_m[nn]


def run(ses):
    S = f"{ROOT}/{ses}"; out = f"{DERIV}/{ses}"
    ref = nib.load(f"{out}/FA.nii.gz")
    ainv = np.linalg.inv(ref.affine)
    maps = {m: nib.load(f"{out}/{m}.nii.gz").get_fdata() for m in METS}
    rng = np.random.default_rng(abs(hash(ses)) % 2**32)

    # candidates + their disk means, both hemispheres pooled (vertex ids offset by hemi)
    cand = {r: [] for r in RUNGS}; dmv = {r: [] for r in RUNGS}
    sulc_all = {r: [] for r in RUNGS}; rawn = {r: 0 for r in RUNGS}
    for h in ("lh", "rh"):
        sets, sulc, ctx, g, mid = hemi_rungs(S, ses, h)
        E = strata(sulc, sets["3HG"])
        per = max(POOL // NBIN, 1)
        for r in RUNGS:
            rawn[r] += len(sets[r])
            take = []
            for b in range(NBIN):
                idx = sets[r][(sulc[sets[r]] > E[b]) & (sulc[sets[r]] <= E[b + 1])]
                if len(idx):
                    take.append(idx if len(idx) <= per else rng.choice(idx, per, replace=False))
            cv = np.concatenate(take) if take else sets[r]
            vals = np.empty((len(cv), len(METS)))
            for i, v in enumerate(cv):
                dd = dijkstra(g, indices=int(v), limit=DISK_MM)
                disk = np.where(np.isfinite(dd))[0]
                disk = disk[ctx[disk]]
                xyz = mid[disk]
                for j, m in enumerate(METS):
                    vals[i, j] = float(np.mean(trilinear(maps[m], ainv, xyz)))
            cand[r].append(cv); dmv[r].append(vals); sulc_all[r].append(sulc[cv])
    # pool hemis into flat arrays; sulc kept per candidate for matching
    val = {r: np.vstack(dmv[r]) for r in RUNGS}
    sv = {r: np.concatenate(sulc_all[r]) for r in RUNGS}
    idx = {r: np.arange(len(sv[r])) for r in RUNGS}

    # z reference: pooled landmark spread within this subject
    allv = np.vstack([val[r] for r in RUNGS])
    mu, sd = np.nanmean(allv, 0), np.nanstd(allv, 0)

    acc = {r: [] for r in RUNGS}
    for rep in range(NREP):
        # resample every rung onto the 3HG candidates' sulc deciles, quota = min availability
        edges = np.quantile(sv["3HG"], np.linspace(0, 1, NBIN + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        binned = {r: [idx[r][(sv[r] > edges[b]) & (sv[r] <= edges[b + 1])]
                      for b in range(NBIN)] for r in RUNGS}
        pick = {r: [] for r in RUNGS}
        for b in range(NBIN):
            q = min(len(binned[r][b]) for r in RUNGS)
            if q == 0:
                continue
            for r in RUNGS:
                p = binned[r][b]
                pick[r].append(p if len(p) == q else rng.choice(p, q, replace=False))
        for r in RUNGS:
            sel = np.concatenate(pick[r]) if pick[r] else idx[r]
            acc[r].append(np.nanmean(val[r][sel], 0))
    z = {r: (np.mean(acc[r], 0) - mu) / sd for r in RUNGS}
    trend = z["4HG+"] - z["crest"]                       # end-to-end contrast, per metric
    np.savez(f"{out}/ladder.npz", mets=METS, rungs=RUNGS, rawn=[rawn[r] for r in RUNGS],
             z=np.array([z[r] for r in RUNGS]), trend=trend)
    print(f"{ses}  n crest/3HG/4HG+ = {rawn['crest']}/{rawn['3HG']}/{rawn['4HG+']}  " +
          "  ".join(f"{m} {z['crest'][j]:+.2f}->{z['3HG'][j]:+.2f}->{z['4HG+'][j]:+.2f}"
                    for j, m in enumerate(METS)), flush=True)


if __name__ == "__main__":
    for ses in sys.argv[1:]:
        run(ses)

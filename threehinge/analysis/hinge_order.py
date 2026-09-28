#!/usr/bin/env python3
"""Hinge-order analysis: is the 3HG signature a step change, or the top of a gradient?

GyralNet defines a 3HG as a vertex of degree >= 3 in the gyral skeleton. The same skeleton gives
the ladder of folding convergence, so the 3HG-vs-crown contrast can be opened out into

    gyral crest (no junction)  ->  3HG (three-way)  ->  4HG+ (four-way or higher)

An earlier version of this analysis split the first rung into an off-skeleton gyral crown and a
degree-2 skeleton point ("2HG"). They are not two things: a crown vertex lies a median 1.0 mm from
the nearest skeleton vertex, 95-98% of the most-gyral quartile is within 3 mm of the skeleton, and
the 2 mm disks used to read the microstructure therefore overlap almost completely. Nor is a
degree-2 vertex a hinge of order two - it is an ordinary point on the crest line, sitting at
sulc percentile 15-18 with ~70% of such points inside the most-gyral quartile, while the junctions
sit higher still at percentile 8-10. The two are pooled here into one non-junction crest rung.

Each rung is a set of surface vertices; the microstructure is read with exactly the estimator used
for the main figure (2 mm geodesic disk means of the mid-cortical maps) so the numbers are directly
comparable. Specimen is the unit (n = 6). Two things are tested per metric:
  monotonic trend - a linear contrast over the four ordered rungs, per specimen, tested against 0;
  step vs gradient - whether 2HG already differs from crown, and whether 3HG differs from 2HG.
A gradient upgrades the claim from '3HG carry a special label' to 'microstructure tracks folding
convergence'; a 2HG ~ 3HG result would instead say the phenotype is a general gyral-hinge one.

The rungs do NOT sit at the same depth in the fold: raw medians are at host-sulc percentile 15-17
(crown), 20-29 (2HG), 8-10 (3HG) and 2-9 (4HG+), so an unmatched ladder measures position in the
fold as much as folding convergence. Every rung is therefore resampled to the 3HG sulc histogram
before anything is measured. Curvature is deliberately NOT matched - local curvature is what makes
a hinge a hinge, so matching it would control away the effect itself."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import sparse, stats
from scipy.sparse.csgraph import dijkstra
import pyvista as pv
import dk
from hinge_profile import mesh_graph, clean_mid, disk_means, MODS, FS, GNET, H, DISK_MM

OUT = (_env.ROOT + "/surf/three_hinge/results")
SUBJ = [1, 2, 3, 4, 5, 6]
FAR = 6.0                      # crest points must be this far (geodesic mm) from any junction
NBIN = 10                      # sulc-matching strata
NREP = 50                      # matched draws averaged, so one lucky draw cannot carry a rung
POOL = 1200                    # cap on how many candidates per rung get a disk mean computed
RUNGS = ["crest", "3HG", "4HG+"]
SHOW = ["thickness", "ADC", "hindered_ratio", "water_ratio", "fiber_ratio", "restricted_ratio", "FA"]


def skeleton_degree(subj, nv):
    """degree of every surface vertex in the gyral skeleton (0 = not on the skeleton).

    The skeleton VTK carries one point per white-surface vertex in the same order (verified), so
    its line segments index surface vertices directly."""
    m = pv.read(f"{GNET}/{subj}/out/{H}_surf_skelenton_allpoints_final_refine.vtk")
    assert m.n_points == nv, f"{subj}: skeleton has {m.n_points} points, surface has {nv}"
    L = np.asarray(m.lines); deg = {}
    i = 0
    adj = [set() for _ in range(nv)]
    while i < len(L):
        k = int(L[i]); seg = L[i + 1:i + 1 + k]
        for j in range(k - 1):
            a, b = int(seg[j]), int(seg[j + 1])
            adj[a].add(b); adj[b].add(a)
        i += k + 1
    return np.array([len(s) for s in adj])


def strata(sulc, tgt):
    """decile edges of the 3HG sulc distribution -- the matching target"""
    e = np.quantile(sulc[tgt], np.linspace(0, 1, NBIN + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def match_sulc(sets, sulc, rng):
    """Resample the rungs onto a common sulc histogram, so they differ in folding order and not in
    where they sit along the fold.

    The quota per stratum is the smallest availability across rungs. That gives every rung the
    IDENTICAL sulc distribution, which is what matching requires - the distributions must equal
    each other, not the 3HG one in particular - and it keeps the four-way junctions, of which there
    are only a few dozen, essentially intact. The abundant crest rung is subsampled to the same
    size, but NREP repeated draws sweep through its candidates, so little is lost.
    Trying instead to give each rung the largest sample matching the 3HG histogram costs the
    scarcest rung two thirds of its points, which is the opposite of what is wanted."""
    edges = strata(sulc, sets["3HG"])
    binned = {r: [idx[(sulc[idx] > edges[b]) & (sulc[idx] <= edges[b + 1])] for b in range(NBIN)]
              for r, idx in sets.items()}
    out = {r: [] for r in sets}
    for b in range(NBIN):
        q = min(len(binned[r][b]) for r in sets)
        if q == 0:
            continue
        for r in sets:
            pool = binned[r][b]
            out[r].append(pool if len(pool) == q else rng.choice(pool, q, replace=False))
    return {r: np.sort(np.concatenate(v)) if v else np.array([], int) for r, v in out.items()}


def rungs_of(sid):
    s = f"Subject{sid}"
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white"); v = v.astype(float)
    parc = dk.read(s)[0]; ctx = parc > 0
    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
    deg = skeleton_degree(s, len(v))
    G = mesh_graph(v, f)
    hinge_any = np.where((deg >= 3) & ctx)[0]                 # every real hinge, any order
    dmin = dijkstra(G, indices=hinge_any.tolist(), min_only=True)
    lo = np.percentile(sulc[ctx], 25)                         # the most-gyral quartile, as in Fig. 1

    # NOT merged at 6 mm, deliberately, unlike FIG1 and the connectivity nodes. Single linkage
    # collapses a junction complex correctly - the largest 3HG cluster holds 13 landmarks - but
    # it CHAINS along a continuum: crest points run continuously down a gyral ridge, every
    # neighbour is within 6 mm, and one cluster swallows 916 of 10,644 points. Its medoid is then
    # an arbitrary point on a whole gyrus, not a collapsed junction. Merging made every metric
    # 6/6 with P < 0.05, which is an artefact of replacing a 10,000-point mean with 95 noisy ones
    # while the three-rung z-score rescales around them.
    #
    # Precision here is handled differently anyway: the rungs are matched on a common sulc
    # histogram and averaged over 50 redraws, and the estimator is a 2 mm geodesic disk whose
    # overlap is far smaller than the 3 mm collection spheres of the connectivity analysis.
    sets = {
        "3HG":   np.where((deg == 3) & ctx)[0],
        "4HG+":  np.where((deg >= 4) & ctx)[0],
        # non-junction gyral crest: on or beside the crest line, in the most-gyral quartile as in
        # Fig. 1, and kept away from junctions. Degree 0 and degree 2 vertices are pooled - they
        # are the same tissue at this scale.
        "crest": np.where((deg < 3) & ctx & (sulc <= lo) & (dmin > FAR))[0],
    }
    return v, f, G, parc, sets, deg, ctx, sulc


def run(sid):
    s = f"Subject{sid}"
    v, f, G, parc, raw, deg, ctx, sulc = rungs_of(sid)
    sets = match_sulc(raw, sulc, np.random.RandomState(sid))
    maps = {m: clean_mid(s, m) for m in MODS
            if os.path.exists(f"{FS}/{s}/surf_maps/{H}.{m}.profile.mgh")}
    maps["thickness"] = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.thickness")
    for m in maps:                                            # never sample outside cortex
        maps[m] = np.where(ctx, maps[m], np.nan)

    # disk means are computed ONCE per candidate vertex; the matched draws then just index into
    # them, so 50 repeats cost nothing and the result no longer hinges on one lucky subsample
    rng = np.random.RandomState(sid)
    E = strata(sulc, raw["3HG"])
    per = max(POOL // NBIN, 1)
    cand = {}
    for r in RUNGS:                       # stratified, so every sulc stratum stays represented
        take = []
        for b in range(NBIN):
            idx = raw[r][(sulc[raw[r]] > E[b]) & (sulc[raw[r]] <= E[b + 1])]
            if len(idx):
                take.append(idx if len(idx) <= per else rng.choice(idx, per, replace=False))
        cand[r] = np.sort(np.concatenate(take)) if take else raw[r]
    dm = {r: disk_means(G, cand[r], maps, DISK_MM) for r in RUNGS}
    pos = {r: {int(v): i for i, v in enumerate(cand[r])} for r in RUNGS}

    # within-specimen scale reference: the spread of individual landmark values across all rungs.
    # z-scoring across the four rung MEANS instead would divide out exactly the between-specimen
    # variance the group test needs, and manufacture tiny P values.
    ref = {m: (np.nanmean(np.concatenate([dm[r][m] for r in RUNGS])),
               np.nanstd(np.concatenate([dm[r][m] for r in RUNGS]))) for m in maps}
    acc = {r: {m: [] for m in maps} for r in RUNGS}; msulc = None
    msize = {r: [] for r in RUNGS}
    for rep in range(NREP):
        sets = match_sulc(cand, sulc, np.random.RandomState(1000 * sid + rep))
        for r in RUNGS:
            msize[r].append(len(sets[r]))
        if rep == 0:
            msulc = {r: np.median(sulc[sets[r]]) for r in RUNGS}
        for r in RUNGS:
            sel = np.array([pos[r][int(v)] for v in sets[r]], int)
            for m in maps:
                acc[r][m].append(np.nanmean(dm[r][m][sel]))
    out = {r: {m: np.array(acc[r][m]) for m in maps} for r in RUNGS}
    nmatched = {r: int(np.mean(msize[r])) for r in RUNGS}
    print(f"{s}: raw " + " ".join(f"{r}={len(raw[r])}" for r in RUNGS) +
          f"  -> {NREP} sulc-matched draws, n=" + "/".join(str(nmatched[r]) for r in RUNGS) +
          "  | matched median sulc " + " ".join(f"{msulc[r]:+.2f}" for r in RUNGS))
    return out, ref, {r: len(raw[r]) for r in RUNGS}, list(maps.keys())


def main():
    Z = {}; N = {}
    for sid in SUBJ:
        out, ref, cnt, mods = run(sid)
        N[sid] = cnt
        for m in mods:
            # z-score within specimen across the pooled rungs, so specimens can be averaged
            mu, sd = ref[m]                       # per-landmark spread within this specimen
            for r in RUNGS:
                Z.setdefault(m, {}).setdefault(r, []).append((np.nanmean(out[r][m]) - mu) / (sd or 1))
    np.savez(f"{OUT}/hinge_order.npz",
             **{f"{m}|{r}": np.array(Z[m][r]) for m in Z for r in RUNGS},
             rungs=np.array(RUNGS), subj=np.array(SUBJ),
             counts=np.array([[N[s][r] for r in RUNGS] for s in SUBJ]))

    con = np.array([-1.0, 0.0, 1.0])                          # linear contrast over the three rungs
    print(f"\n{'metric':18s}" + "".join(f"{r:>9s}" for r in RUNGS) +
          f"{'trend P':>10s}{'6/6':>5s}{'3HG-crest':>11s}{'4HG-3HG':>10s}")
    for m in SHOW:
        if m not in Z:
            continue
        M = np.array([Z[m][r] for r in RUNGS])                # (rung, specimen)
        t = con @ M                                           # per-specimen trend score
        p = stats.ttest_1samp(t, 0).pvalue
        d1 = stats.ttest_rel(M[1], M[0]).pvalue               # does a three-way junction differ?
        d2 = stats.ttest_rel(M[2], M[1]).pvalue               # does going beyond three add more?
        print(f"{m:18s}" + "".join(f"{M[i].mean():+9.3f}" for i in range(len(RUNGS))) +
              f"{p:10.4f}{int(np.all(np.sign(t) == np.sign(t[0]))):5d}"
              f"{d1:11.4f}{d2:10.4f}")
    print("\nvalues are within-specimen z (pooled over the four rungs); trend = linear contrast,")
    print("two-sided t over 6 specimens; 6/6 = all specimens agree in sign.")


if __name__ == "__main__":
    main()

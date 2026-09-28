#!/usr/bin/env python3
"""Matching / confound diagnostics behind the folding-convergence ladder (data pass).

Per specimen, on the gyral skeleton machinery of hinge_order.py:
  sulc     host-sulc percentile distributions of the three rungs, BEFORE and AFTER the
           decile matching (the matching is what the ladder rests on - show it).
  curv     |mean curvature| distributions after matching (deliberately NOT matched:
           a junction is a point of high curvature by definition - show what remains).
  counts   raw vertex counts at each skeleton degree 2,3,4,5,6+, per specimen.
  beta     per-specimen OLS of the landmark z value on skeleton degree (2..6, capped),
           with host-sulc percentile and DK-parcel dummies as covariates, WITH and
           WITHOUT |curvature| - the slope the folding-convergence claim rests on,
           shown to survive the curvature covariate.
Landmarks are the POOL-capped candidate sets of hinge_order (disk means computed once).
Writes results/matching_diag.npz.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy.sparse.csgraph import dijkstra
from hinge_profile import mesh_graph, clean_mid, disk_means, FS, H, DISK_MM
from hinge_order import rungs_of, match_sulc, strata, RUNGS, NBIN, POOL

OUT = (_env.ROOT + "/surf/three_hinge/results")
SUBJ = [1, 2, 3, 4, 5, 6]
METS = {"MD": "ADC", "AD": "AD", "RD": "RD", "FA": "FA", "restricted": "restricted_ratio",
        "fiber": "fiber_ratio", "hindered": "hindered_ratio", "water": "water_ratio"}


def pctile_of(sulc, ctx):
    order = np.argsort(np.argsort(sulc[ctx]))
    p = np.full(len(sulc), np.nan)
    p[ctx] = order / (ctx.sum() - 1) * 100
    return p


def run(sid):
    s = f"Subject{sid}"
    v, f, G, parc, raw, deg, ctx, sulc = rungs_of(sid)
    curv = np.abs(nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.curv"))
    spct = pctile_of(sulc, ctx)
    rng = np.random.RandomState(sid)

    # distributions: raw rung sulc percentiles; matched (pooled over 20 draws)
    dist_raw = {r: spct[raw[r]] for r in RUNGS}
    m_s, m_c = {r: [] for r in RUNGS}, {r: [] for r in RUNGS}
    for rep in range(20):
        m = match_sulc(raw, sulc, np.random.RandomState(7000 * sid + rep))
        for r in RUNGS:
            m_s[r].append(spct[m[r]]); m_c[r].append(curv[m[r]])
    dist_mat = {r: np.concatenate(m_s[r]) for r in RUNGS}
    dist_cur = {r: np.concatenate(m_c[r]) for r in RUNGS}

    # counts per skeleton degree
    cnt = [int(((deg == d) if d < 6 else (deg >= 6))[ctx[np.arange(len(deg))] if False else ctx].sum())
           for d in (2, 3, 4, 5, 6)]
    cnt = [int((ctx & ((deg == d) if d < 6 else (deg >= 6))).sum()) for d in (2, 3, 4, 5, 6)]

    # regression candidates: POOL-capped degree-2 crest line + every junction, degree capped at 6
    E = strata(sulc, raw["3HG"]); per = max(POOL // NBIN, 1)
    c2 = np.where((deg == 2) & ctx)[0]
    take = []
    for b in range(NBIN):
        idx = c2[(sulc[c2] > E[b]) & (sulc[c2] <= E[b + 1])]
        if len(idx):
            take.append(idx if len(idx) <= per else rng.choice(idx, per, replace=False))
    cand = np.unique(np.r_[np.concatenate(take), np.where((deg >= 3) & ctx)[0]])
    maps = {m: np.where(ctx, clean_mid(s, k), np.nan) for m, k in METS.items()}
    dm = disk_means(G, cand, maps, DISK_MM)

    dcap = np.minimum(deg[cand], 6).astype(float)
    X0 = [dcap, spct[cand]]
    P = parc[cand]
    dums = [(P == p).astype(float) for p in np.unique(P)[1:]]
    beta = {}
    for m in METS:
        y = dm[m]
        ok = np.isfinite(y)
        z = (y - np.nanmean(y)) / (np.nanstd(y) or 1)
        for tag, cols in (("base", X0), ("curv", X0 + [curv[cand]])):
            A = np.c_[np.ones(ok.sum()), np.array(cols).T[ok], np.array(dums).T[ok]]
            b = np.linalg.lstsq(A, z[ok], rcond=None)[0]
            beta[f"{m}|{tag}"] = b[1]
    print(f"{s}: cand={len(cand)} counts(deg2..6+)={cnt}  " +
          " ".join(f"{m}:{beta[f'{m}|base']:+.3f}/{beta[f'{m}|curv']:+.3f}" for m in ("MD", "FA")),
          flush=True)
    return dist_raw, dist_mat, dist_cur, cnt, beta


def main():
    DR, DM, DC, CNT, BET = {}, {}, {}, [], {}
    for sid in SUBJ:
        dr, dm_, dc, cnt, bet = run(sid)
        CNT.append(cnt)
        for r in RUNGS:
            DR.setdefault(r, []).append(dr[r]); DM.setdefault(r, []).append(dm_[r])
            DC.setdefault(r, []).append(dc[r])
        for k, v in bet.items():
            BET.setdefault(k, []).append(v)
    np.savez(f"{OUT}/matching_diag.npz",
             **{f"raw|{r}": np.concatenate(DR[r]) for r in RUNGS},
             **{f"mat|{r}": np.concatenate(DM[r]) for r in RUNGS},
             **{f"cur|{r}": np.concatenate(DC[r]) for r in RUNGS},
             counts=np.array(CNT), degrees=np.array([2, 3, 4, 5, 6]),
             **{f"beta|{k}": np.array(v) for k, v in BET.items()},
             mets=np.array(list(METS)))
    print("wrote matching_diag.npz")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ex-vivo 3HG-vs-crown Cohen's d with SAME-PARCEL crown controls, for Fig. 4b only.

The primary ex-vivo analysis (hinge_profile.py -> Fig. 2) draws its count-matched crown
control from the whole hemisphere; the in-vivo pipeline matches the crown draw to the 3HG
DK-parcel histogram. Fig. 4b overlays the two experiments' d on one axis, so the ex-vivo
values shown THERE are recomputed with the in-vivo control design - per-parcel matched
draws - while every other choice (2 mm disk means at mid-depth, junction-level grouping,
20 draws averaged at the level of d, the same crown pool definition sulc<=Q25, >8 mm from
any 3HG, cortex only) is exactly hinge_profile's. Fig. 2 is untouched.

Writes results/exvivo_parcel_crown_d.npz: d (6 specimens x [FA MD AD RD]).
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np, nibabel as nib, sys
import dk
import hinge_profile as hp
from hinge_ids import cluster_groups, group_mean
from scipy.sparse.csgraph import dijkstra

METS = {"FA": "FA", "MD": "ADC", "AD": "AD", "RD": "RD"}
NDRAW = 20


def run(sid):
    s = f"Subject{sid}"
    v, f = nib.freesurfer.read_geometry(f"{hp.FS}/{s}/surf/{hp.H}.white")
    v = v.astype(np.float64)
    sulc = nib.freesurfer.read_morph_data(f"{hp.FS}/{s}/surf/{hp.H}.sulc")
    hinge = np.atleast_1d(np.loadtxt(f"{hp.GNET}/{s}/out/{hp.H}_3hinge_ids.txt", dtype=int))
    parc = dk.read(s)[0]
    ctx = parc > 0
    hinge = hinge[ctx[hinge]]
    maps = {m: np.where(ctx, hp.clean_mid(s, hp_m), np.nan) for m, hp_m in METS.items()}

    G = hp.mesh_graph(v, f)
    dmin = dijkstra(G, indices=hinge, min_only=True)
    lo = np.percentile(sulc[ctx], 25)
    pool = (sulc <= lo) & (dmin > hp.CROWN_FAR) & ctx

    # per-parcel matched draws: k_p crown vertices from parcel p, k_p = 3HG count there
    rng = np.random.RandomState(sid)
    draws, short = [], 0
    for _ in range(NDRAW):
        pick = []
        for p in np.unique(parc[hinge]):
            k = int((parc[hinge] == p).sum())
            cand = np.where(pool & (parc == p))[0]
            if len(cand):
                pick.append(rng.choice(cand, size=min(k, len(cand)), replace=False))
            short += max(k - len(cand), 0)
        draws.append(np.sort(np.concatenate(pick)))
    Hd = hp.disk_means(G, hinge, maps, hp.DISK_MM)
    grp = cluster_groups(sid, hinge, merge=hp.SEP)
    Hd = {m: group_mean(Hd[m], grp) for m in Hd}
    allc = np.unique(np.concatenate(draws))
    Cd_all = hp.disk_means(G, allc, maps, hp.DISK_MM)
    ci = {int(x): i for i, x in enumerate(allc)}

    d = {}
    for m in METS:
        h = Hd[m]; ds = []
        for c in draws:
            cv = Cd_all[m][[ci[int(x)] for x in c]]
            sp = np.sqrt((np.nanvar(h, ddof=1) + np.nanvar(cv, ddof=1)) / 2)
            ds.append((np.nanmean(h) - np.nanmean(cv)) / sp)
        d[m] = float(np.nanmean(ds))
    print(f"{s}: n3HG={len(hinge)} crown/draw={len(draws[0])} "
          f"(unfilled quota {short // NDRAW}/draw)  " +
          "  ".join(f"{m} d={d[m]:+.3f}" for m in METS), flush=True)
    return [d[m] for m in METS]


if __name__ == "__main__":
    D = np.array([run(sid) for sid in [1, 2, 3, 4, 5, 6]])
    np.savez(f"{hp.OUT}/exvivo_parcel_crown_d.npz", d=D, mets=list(METS))
    print("mean d:", dict(zip(METS, np.round(D.mean(0), 3))))

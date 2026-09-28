#!/usr/bin/env python3
"""Fold-depth distributions of the three sampled classes, one density per hemisphere.

Feeds the fold-depth axis of Fig. 2a (figures/fig_main1_combined.py). For each of the six
hemispheres the FreeSurfer `sulc` map is standardized to that hemisphere's own cortex, and a
density is taken on one fixed grid for: all cortical vertices, the 3HG landmarks, the gyral-crown
draw and the sulcal-fundus draw. The crown and fundus draws are exactly draw 0 of
hinge_profile.py - same pools, same RandomState(sid) - so the panel shows the matched control
that was measured rather than an illustration of one. The figure plots the mean over hemispheres
with its s.e.m.; the unit is the specimen.

    python3 match_sulc.py        # writes results/_classes/match_sulc.npz
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy.sparse.csgraph import dijkstra
import dk
from hinge_profile import mesh_graph, CROWN_FAR

FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
OUT = (_env.ROOT + "/surf/three_hinge/results/_classes")
H = "rh"; SUBJ = [1, 2, 3, 4, 5, 6]
BINS = np.linspace(-2.4, 2.4, 41)          # 0.12 s.d. bins: per-specimen curves, then the cohort


def main():
    os_ = _os; os_.makedirs(OUT, exist_ok=True)
    D = {k: [] for k in ("ctx", "hinge", "crown", "fundus")}; q = []; n = []
    for sid in SUBJ:
        s = f"Subject{sid}"
        v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white"); v = v.astype(float)
        sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
        ctx = dk.read(s)[0] > 0
        hv = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int); hv = hv[ctx[hv]]
        dmin = dijkstra(mesh_graph(v, f), indices=hv.tolist(), min_only=True)
        lo, hi = np.percentile(sulc[ctx], 25), np.percentile(sulc[ctx], 75)
        cp = np.where((sulc <= lo) & (dmin > CROWN_FAR) & ctx)[0]
        fp = np.where((sulc >= hi) & ctx)[0]
        rng = np.random.RandomState(sid)            # exactly hinge_profile.py's draw 0
        cr = rng.choice(cp, min(len(hv), len(cp)), replace=False)
        fu = rng.choice(fp, min(len(hv), len(fp)), replace=False)
        mu, sd = sulc[ctx].mean(), sulc[ctx].std()
        for k, idx in (("ctx", np.where(ctx)[0]), ("hinge", hv), ("crown", cr), ("fundus", fu)):
            D[k].append(np.histogram((sulc[idx] - mu) / sd, bins=BINS, density=True)[0])
        q.append(((lo - mu) / sd, (hi - mu) / sd)); n.append(len(hv))
    q = np.array(q)
    np.savez(f"{OUT}/match_sulc.npz", bins=BINS, q25=q[:, 0].mean(), q75=q[:, 1].mean(),
             nland=np.array(n), **{k: np.array(v) for k, v in D.items()})
    print(f"wrote {OUT}/match_sulc.npz  ({len(BINS) - 1} bins, landmarks per hemisphere {n})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does gyral bias explain the inter-hinge excess? Measured in this data, not imported.

Deterministic tractography terminates preferentially on gyral crowns rather than sulcal walls and
fundi. The published correction for it (e.g. Schilling et al. 2018 as applied by others) assumes a
crown-to-wall fibre-density ratio from histology and rescales node strength by a linear function of
`sulc`. Two reasons that is the wrong instrument here, and one reason it is unnecessary:

  - our contrast is 3HG against GYRAL CROWN, and the control is restricted to the most-gyral
    quartile, so both classes sit on the crown side of the fold - the crown-versus-fundus axis the
    correction acts on is already controlled by design;
  - the correction rescales node STRENGTH, and the reported effect is connection INCIDENCE, which a
    multiplicative strength correction leaves untouched;
  - and the residual fold-depth difference between the two classes works AGAINST the effect here,
    so correcting for it makes the result stronger, not weaker.

This script measures all three, per specimen:
  1. endpoint density on the white surface against `sulc` percentile - the canonical gyral-bias
     curve, so the reader can see the bias exists in this data and how steep it is;
  2. where the two classes actually sit on that curve;
  3. the class effect on pair-level connection incidence with `sulc` and inter-node distance as
     covariates, against the unadjusted effect.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from scipy.spatial import cKDTree
import dk
FS = (_env.ROOT + "/surf/fs")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
OUT = (_env.ROOT + "/surf/three_hinge/results")
H = "rh"; SUBJ = [1, 2, 3, 4, 5, 6]
NBIN = 20                                        # sulc percentile bins for the density curve
NEAR = 3.0                                       # mm; an endpoint is assigned to a vertex within this


def vertex_area(v, f):
    tri = v[f]
    a = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(v)); np.add.at(out, f.ravel(), np.repeat(a, 3) / 3)
    return out


def run(sid):
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    vw = nib.affines.apply_affine(tk, v.astype(float))
    parc = dk.read(s)[0]; ctx = parc > 0
    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
    pct = np.full(len(sulc), np.nan)
    pct[ctx] = stats.rankdata(sulc[ctx]) / ctx.sum() * 100
    area = vertex_area(vw, f)

    from dipy.io.streamline import load_tractogram
    from dipy.io.stateful_tractogram import Space
    dwi = nib.load(f"{TR}/{s}/dwi.nii.gz")
    tg = load_tractogram(f"{TR}/{s}/whole_brain.trk", dwi, to_space=Space.RASMM,
                         bbox_valid_check=False)
    ep = np.vstack([np.r_[[np.asarray(x)[0]], [np.asarray(x)[-1]]] for x in tg.streamlines])

    # every endpoint to its nearest white-surface vertex, then density per mm^2 of cortex in each
    # sulc bin. Area-normalised, or the curve would just report how much cortex each bin holds.
    d, idx = cKDTree(vw).query(ep)
    hit = idx[(d < NEAR) & ctx[idx]]
    cnt = np.bincount(hit, minlength=len(vw)).astype(float)
    edges = np.linspace(0, 100, NBIN + 1)
    b = np.digitize(pct, edges) - 1
    dens = np.array([cnt[(b == k) & ctx].sum() / max(area[(b == k) & ctx].sum(), 1e-9)
                     for k in range(NBIN)])

    z = np.load(f"{TR}/{s}/conn_matrix.npz")
    kind, vert, D = z["kind"], z["vert"], z["nodedist"]
    A = (((z["Mb"] if "Mb" in z.files else z["M"]) + (z["Mb"] if "Mb" in z.files else z["M"]).T) > 0)
    sp = pct[vert]
    hi = np.where(kind == 0)[0]; ci = np.where(kind == 1)[0]

    import statsmodels.api as sm
    ns = z["nseed"]; ln = np.log(np.maximum(ns, 1))
    rows = []
    for ids, cls in ((hi, 1), (ci, 0)):
        iu = np.triu_indices(len(ids), 1)
        i, j = ids[iu[0]], ids[iu[1]]
        rows.append(np.c_[A[i, j].astype(float), np.full(len(i), cls),
                          (sp[i] + sp[j]) / 2, D[i, j], ln[i] + ln[j]])
    M = np.vstack(rows); M = M[np.all(np.isfinite(M), 1)]
    fit = lambda cols: sm.Logit(M[:, 0], sm.add_constant(M[:, cols])).fit(disp=0).params[1]
    # the three covariate sets the supplementary panel contrasts. log seed count is a MEDIATOR if
    # the extra streamlines are real fibre convergence and a CONFOUNDER if they are sampling bias;
    # both fits are reported because the data alone cannot separate the two.
    ors = [fit([1]), fit([1, 2, 3]), fit([1, 4]), fit([1, 2, 3, 4])]
    deg = A.sum(1).astype(float)
    print(f"  S{sid}: {len(ep):,} endpoints, {100*len(hit)/len(ep):.0f}% assigned; "
          f"3HG sulc pct {np.median(sp[hi]):.1f}, crown {np.median(sp[ci]):.1f}; "
          f"logOR {ors[0]:+.3f} raw -> {ors[2]:+.3f} with seed count", flush=True)
    return (dens, sp[hi], sp[ci], np.array(ors),
            np.array([ns[hi].mean(), ns[ci].mean()]), np.array([deg[hi].mean(), deg[ci].mean()]))


if __name__ == "__main__":
    D, SH, SC, OR, NS, DG = [], [], [], [], [], []
    for sid in SUBJ:
        d, sh, sc, o, ns, dg = run(sid)
        D.append(d); SH.append(sh); SC.append(sc); OR.append(o); NS.append(ns); DG.append(dg)
    np.savez(f"{OUT}/gyral_bias.npz", dens=np.array(D), edges=np.linspace(0, 100, NBIN + 1),
             sulc_h=np.concatenate(SH), sulc_c=np.concatenate(SC),
             ors=np.array(OR), nseed=np.array(NS), degree=np.array(DG))
    d = np.array(D)
    q = np.nanmedian(d, 0)
    print(f"\nendpoint density, gyral end {q[0]:.3f} -> sulcal end {q[-1]:.3f} per mm2  "
          f"({q[0]/q[-1]:.2f}x)")
    print(f"wrote {OUT}/gyral_bias.npz")

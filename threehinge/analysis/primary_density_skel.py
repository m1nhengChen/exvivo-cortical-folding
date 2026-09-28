#!/usr/bin/env python3
"""Sensitivity analysis for the primary vs non-primary 3HG enrichment: normalize by GYRAL
SKELETON LENGTH instead of cortical area - junctions per 100 mm of gyral ridge, so the
comparison is relative to the extent of each region's ridge system, not its surface area.

Segment lengths come from the skeleton polylines themselves (Euclidean, on the same white
surface the skeleton indexes); a segment with endpoints in different classes contributes
half to each. Per hemisphere/subject: R = (N_non/L_non) / (N_prim/L_prim), tested as a
one-sample t on log R (the same independent-unit statistic as the area version).

Usage: primary_density_skel.py --exvivo | --invivo SESSION [...]
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
import pyvista as pv

PRIM = ("precentral", "postcentral", "pericalcarine", "transversetemporal")


def skel_stats(skel_path, names, lab, ctx, hids):
    m = pv.read(skel_path)
    P = np.asarray(m.points)
    isprim_name = np.array([any(n.endswith(p) for p in PRIM) for n in names])
    vprim = isprim_name[lab] & ctx
    L = np.asarray(m.lines)
    Lp = Ln = 0.0
    i = 0
    while i < len(L):
        k = int(L[i]); seg = L[i + 1:i + 1 + k]
        for j in range(k - 1):
            a, b = int(seg[j]), int(seg[j + 1])
            if not (ctx[a] or ctx[b]):
                i_ = None  # neither end analyzable: skip
            d = float(np.linalg.norm(P[a] - P[b]))
            for e in (a, b):                       # half to each endpoint's class
                if ctx[e]:
                    if vprim[e]:
                        Lp += d / 2
                    else:
                        Ln += d / 2
        i += k + 1
    npm = int(vprim[hids].sum()); nnm = int((~vprim & ctx)[hids].sum())
    return npm, Lp, nnm, Ln


def exvivo():
    FS = (_env.ROOT + "/surf/fs")
    GNET = (_env.ROOT + "/surf/three_hinge/gnet")
    out = []
    for sid in [1, 2, 3, 4, 5, 6]:
        s = f"Subject{sid}"
        lab, _, names = nib.freesurfer.read_annot(f"{FS}/{s}/label/rh.aparc.annot")
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        bad = {i for i, n in enumerate(names) if n in ("unknown", "corpuscallosum")}
        ctx = ~np.isin(lab, list(bad)) & (lab >= 0)
        hids = np.atleast_1d(np.loadtxt(f"{GNET}/{s}/out/rh_3hinge_ids.txt", dtype=int))
        hids = hids[ctx[hids]]
        r = skel_stats(f"{GNET}/{s}/out/rh_surf_skelenton_allpoints_final_refine.vtk",
                       names, lab, ctx, hids)
        out.append(r)
        print(f"{s}: primary {r[0]}/{r[1]:.0f}mm = {r[0]/r[1]*100:.2f}/100mm   "
              f"non-primary {r[2]}/{r[3]:.0f}mm = {r[2]/r[3]*100:.2f}/100mm   "
              f"ratio {(r[2]/r[3])/(r[0]/r[1]):.2f}", flush=True)
    np.savez("results/primary_skel_exvivo.npz", data=np.array(out))


def invivo(sessions):
    ROOT = (_env.AABC + "/organized")
    DERIV = (_env.AABC + "/derived")
    for ses in sessions:
        S = f"{ROOT}/{ses}"; row = np.zeros(4)
        try:
            for h in ("lh", "rh"):
                G = f"{S}/postprocessing_tong/gii_native"
                lab, _, names = nib.freesurfer.read_annot(
                    f"{G}/label/{h}.aparc.resample.163842.annot")
                names = [n.decode() if isinstance(n, bytes) else n for n in names]
                bad = {i for i, n in enumerate(names) if n in ("unknown", "corpuscallosum")}
                ctx = ~np.isin(lab, list(bad)) & (lab >= 0)
                hids = np.atleast_1d(np.loadtxt(
                    f"{G}/gyralnet_island_resample_163842_scanner/{h}_3hinge_ids.txt", dtype=int))
                hids = hids[ctx[hids]]
                row += np.array(skel_stats(
                    f"{G}/gyralnet_island_resample_163842_scanner/"
                    f"{h}_surf_skelenton_allpoints_final_refine.vtk",
                    names, lab, ctx, hids))
        except Exception as e:
            print(f"{ses} FAILED {type(e).__name__}: {e}", flush=True); continue
        np.save(f"{DERIV}/{ses}/primary_skel.npy", row)
        print(f"{ses} ratio {(row[2]/row[3])/(row[0]/row[1]):.2f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--exvivo":
        exvivo()
    else:
        invivo(sys.argv[2:])

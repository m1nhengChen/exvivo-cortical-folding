#!/usr/bin/env python3
"""Geometry / partial-volume validation of the intracortical depth result.

The 3HG sits where the cortex is most sharply curved, which is exactly where two artefacts live:
depth sampling that is not curvature-corrected assigns the wrong tissue to a given depth, and a
0.83 mm diffusion voxel at a tight gyral crest contains more white matter and less cortex than the
same voxel on a flat bank. Either could manufacture an MD / hindered / free-water difference.

Three controls, all on the same equivolumetric (Bok / Waehnert) sampling used for the main figure:

  1. tissue fraction   A three-class label volume is built at the 0.5 mm structural resolution -
                       WM inside the white surface (filled.mgz), cortical GM in the ribbon
                       (ribbon.mgz), everything else non-brain - and averaged into each diffusion
                       voxel. Every sampled point therefore carries the GM / WM / other fractions
                       of the voxel it was read from.
  2. GM-purity subset  the profile recomputed using only samples from voxels above PURE GM
                       fraction, so the comparison is between well-resolved cortical voxels.
  3. covariate model   per depth, a linear model of the metric on class (3HG vs crown) with the
                       WM and other fractions as covariates; the reported effect is the class
                       coefficient, i.e. the difference at matched partial volume.

All three are computed on the WITHIN-PARCEL, region-matched contrast that the published depth
statistics test (depth/depth_stats.py operates on dWP), so the validation and the claim are the
same quantity. The count-matched whole-hemisphere contrast is kept alongside as a cross-check.

The metric panel is widened from 6 to 16 so the answer does not rest on one contrast."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import sparse, stats, ndimage as ndi
from scipy.sparse.csgraph import dijkstra
import dk
from depth_profile import vertex_area, equivol_t, mesh_graph, sample, RHO, THMIN, ROOT, FS, GNET, H

OUT = (_env.ROOT + "/surf/three_hinge/depth")
SUBJ = [int(x) for x in os.environ.get("EXVIVO_SUBJ", "1 2 3 4 5 6").split()]
PURE = 0.70                                   # minimum GM fraction for the high-purity subset
MET = {
    "MD":         "{s}_DTI/ADC.nii.gz",       "FA":  "{s}_DTI/FA.nii.gz",
    "AD":         "{s}_DTI/AD.nii.gz",        "RD":  "{s}_DTI/RD.nii.gz",
    "FD":         "{s}_DTI/FD.nii.gz",
    "fiber":      "{s}_DBSI/fiber_ratio_map.nii.gz",
    "restricted": "{s}_DBSI/restricted_ratio_map.nii.gz",
    "hindered":   "{s}_DBSI/hindered_ratio_map.nii.gz",
    "water":      "{s}_DBSI/water_ratio_map.nii.gz",
    "DBSI_FA":    "{s}_DBSI/dti_fa_map.nii.gz",
    "DBSI_MD":    "{s}_DBSI/dti_adc_map.nii.gz",
    "DBSI_AD":    "{s}_DBSI/dti_axial_map.nii.gz",
    "DBSI_RD":    "{s}_DBSI/dti_radial_map.nii.gz",
    "fib1_FA":    "{s}_DBSI/fiber1_fa_map.nii.gz",
    "fib1_AD":    "{s}_DBSI/fiber1_axial_map.nii.gz",
    "fib1_RD":    "{s}_DBSI/fiber1_radial_map.nii.gz",
}


def tissue_fractions(sid, ref):
    """GM / WM / other fraction of every diffusion voxel, from the 0.5 mm structural labels.

    Each structural voxel is assigned to the diffusion voxel containing its centre and the class
    counts accumulated, so the fractions are exact volume fractions rather than an interpolation."""
    s = f"Subject{sid}"
    wm = nib.load(f"{FS}/{s}/mri/filled.mgz")
    gm = nib.load(f"{FS}/{s}/mri/ribbon.mgz")
    A = np.asarray(wm.dataobj) > 0
    B = np.asarray(gm.dataobj) > 0
    lab = np.where(B, 1, np.where(A, 2, 0)).astype(np.int8)      # 0 other, 1 GM, 2 WM
    ijk = np.indices(lab.shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(wm.affine, ijk)               # structural voxel centres, world
    d = np.round(nib.affines.apply_affine(np.linalg.inv(ref.affine), xyz)).astype(int)
    shp = ref.shape[:3]
    keep = np.all((d >= 0) & (d < np.array(shp)), 1)
    flat = np.ravel_multi_index(d[keep].T, shp)
    L = lab.ravel()[keep]
    tot = np.bincount(flat, minlength=int(np.prod(shp))).astype(np.float32)
    out = {}
    for k, name in ((1, "gm"), (2, "wm"), (0, "other")):
        c = np.bincount(flat[L == k], minlength=int(np.prod(shp))).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(tot > 0, c / np.maximum(tot, 1), np.nan).reshape(shp)
    return out


def run(sid):
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    vw, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    vp, _ = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.pial")
    W = nib.affines.apply_affine(tk, vw); P = nib.affines.apply_affine(tk, vp)
    Aw, Ap = vertex_area(W, f), vertex_area(P, f)
    thick = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.thickness")
    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc"); parc = dk.read(s)[0]
    valid = (parc > 0) & (thick >= THMIN)

    imgs = {m: nib.load(f"{ROOT}/{s}/{MET[m].format(s=s)}") for m in MET}
    ref = imgs["MD"]
    fr = tissue_fractions(sid, ref)
    frimg = {k: nib.Nifti1Image(v, ref.affine) for k, v in fr.items()}

    XYZ = [W + equivol_t(r, Aw, Ap)[:, None] * (P - W) for r in RHO]
    prof = {m: np.stack([sample(imgs[m], x) for x in XYZ]) for m in MET}
    pvf = {k: np.stack([sample(frimg[k], x) for x in XYZ]) for k in fr}   # (depth, vert)

    hinge = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int)
    hinge = hinge[parc[hinge] > 0]
    hinge = hinge[valid[hinge]]
    dmin = dijkstra(mesh_graph(W, f), indices=hinge.tolist(), min_only=True)
    lo = np.percentile(sulc[valid], 25)
    crown = np.where((sulc <= lo) & (dmin > 6) & valid)[0]
    # random, not crown[np.linspace(...)]: even intervals of VERTEX INDEX are not even intervals of
    # cortex, so the old draw returned runs of neighbouring vertices. Same fix as hinge_profile.py.
    crown = np.sort(np.random.RandomState(sid).choice(crown, min(len(hinge), len(crown)),
                                                      replace=False))

    nd = len(RHO)
    res = {k: {m: np.full(nd, np.nan) for m in MET} for k in ("raw", "pure", "adj")}
    wp = {k: {m: np.full(nd, np.nan) for m in MET} for k in ("raw", "pure", "adj")}
    pids = np.unique(parc[valid])
    npure = np.zeros(nd)
    for di in range(nd):
        gh, gc = pvf["gm"][di][hinge], pvf["gm"][di][crown]
        npure[di] = np.nanmean(np.r_[gh, gc] >= PURE)
        for m in MET:
            h, c = prof[m][di][hinge], prof[m][di][crown]
            sd = np.nanstd(np.r_[h, c]) or 1
            res["raw"][m][di] = (np.nanmean(h) - np.nanmean(c)) / sd
            mh, mc = gh >= PURE, gc >= PURE
            if mh.sum() >= 30 and mc.sum() >= 30:
                res["pure"][m][di] = (np.nanmean(h[mh]) - np.nanmean(c[mc])) / sd
            # class effect at matched partial volume: metric ~ class + wm_frac + other_frac
            y = np.r_[h, c]
            X = np.c_[np.ones(len(y)), np.r_[np.ones(len(h)), np.zeros(len(c))],
                      np.r_[pvf["wm"][di][hinge], pvf["wm"][di][crown]],
                      np.r_[pvf["other"][di][hinge], pvf["other"][di][crown]]]
            ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
            if ok.sum() > 50:
                beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                res["adj"][m][di] = beta[1] / sd

        # ---- the published contrast: matched within DK parcel, then averaged over parcels ----
        for m in MET:
            vals = prof[m][di]
            acc = {k: [] for k in wp}
            for pid in pids:
                reg = (parc == pid) & valid
                if reg.sum() <= 4:
                    continue
                loc = np.percentile(sulc[reg], 25)
                hin = hinge[parc[hinge] == pid]
                cro = np.where(reg & (sulc <= loc) & (dmin > 6))[0]
                if len(hin) < 5 or len(cro) < 5:
                    continue
                sd = np.nanstd(vals[reg]) or 1
                acc["raw"].append((np.nanmean(vals[hin]) - np.nanmean(vals[cro])) / sd)
                mh, mc = pvf["gm"][di][hin] >= PURE, pvf["gm"][di][cro] >= PURE
                if mh.sum() >= 5 and mc.sum() >= 5:
                    acc["pure"].append((np.nanmean(vals[hin][mh]) - np.nanmean(vals[cro][mc])) / sd)
                y = np.r_[vals[hin], vals[cro]]
                X = np.c_[np.ones(len(y)), np.r_[np.ones(len(hin)), np.zeros(len(cro))],
                          np.r_[pvf["wm"][di][hin], pvf["wm"][di][cro]],
                          np.r_[pvf["other"][di][hin], pvf["other"][di][cro]]]
                ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
                if ok.sum() > 20 and np.linalg.matrix_rank(X[ok]) == X.shape[1]:
                    acc["adj"].append(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0][1] / sd)
            for k in wp:
                if acc[k]:
                    wp[k][m][di] = float(np.mean(acc[k]))
    np.savez(f"{OUT}/{s}_depth_pv.npz", rho=RHO, mets=np.array(list(MET)), npure=npure,
             gm_h=np.nanmean(pvf["gm"][:, hinge], 1), gm_c=np.nanmean(pvf["gm"][:, crown], 1),
             wm_h=np.nanmean(pvf["wm"][:, hinge], 1), wm_c=np.nanmean(pvf["wm"][:, crown], 1),
             **{f"{k}_{m}": res[k][m] for k in res for m in MET},
             **{f"wp{k}_{m}": wp[k][m] for k in wp for m in MET})
    print(f"{s}: GM fraction 3HG {np.nanmean(pvf['gm'][:, hinge]):.3f} vs crown "
          f"{np.nanmean(pvf['gm'][:, crown]):.3f} | WM fraction {np.nanmean(pvf['wm'][:, hinge]):.3f} "
          f"vs {np.nanmean(pvf['wm'][:, crown]):.3f} | high-purity samples "
          f"{100 * npure.mean():.0f}%", flush=True)


if __name__ == "__main__":
    for i in SUBJ:
        run(i)

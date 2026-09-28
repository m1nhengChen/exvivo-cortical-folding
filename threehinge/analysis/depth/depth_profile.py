#!/usr/bin/env python3
"""Cortical-depth (intracortical) profiling of the 3HG microstructure signature.

For every vertex we build EQUIVOLUMETRIC intracortical surfaces between the white (d=0) and
pial (d=1) surfaces — the metric distance for a given volume fraction is curvature-corrected
so each depth bin encloses ~equal cortical volume (Bok / Waehnert), avoiding the crown-vs-
fundus layer-thickness bias of equidistant sampling. At 9 depths we sample MD, FA and the
four DBSI fractions and compare the whole profile Y_3HG(d) vs Y_crown(d).

NOTE: this is cortical-depth / intracortical profiling, NOT histological layer (I–VI) analysis;
no laminar identity is claimed."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import sparse, ndimage as ndi
from scipy.sparse.csgraph import dijkstra
import dk
from hinge_ids import cluster_groups, group_mean, mirror_groups, DISK
ROOT = (_env.ROOT + "/28211843"); FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet"); OUT = (_env.ROOT + "/surf/three_hinge/depth")
H = "rh"; SUBJ = [1, 2, 3, 4, 5, 6]; RHO = np.round(np.arange(0.1, 0.91, 0.1), 2)
THMIN = 1.76
MET = {"MD": "{s}_DTI/ADC.nii.gz", "FA": "{s}_DTI/FA.nii.gz",
       "fiber": "{s}_DBSI/fiber_ratio_map.nii.gz", "restricted": "{s}_DBSI/restricted_ratio_map.nii.gz",
       "hindered": "{s}_DBSI/hindered_ratio_map.nii.gz", "water": "{s}_DBSI/water_ratio_map.nii.gz"}


def vertex_area(v, f):
    tri = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    a = 0.5 * np.linalg.norm(tri, axis=1) / 3.0
    va = np.zeros(len(v)); np.add.at(va, f[:, 0], a); np.add.at(va, f[:, 1], a); np.add.at(va, f[:, 2], a)
    return va


def equivol_t(rho, Aw, Ap):
    """metric distance fraction from white for equal-volume fraction rho (linear area model)."""
    d = Ap - Aw
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-Aw + np.sqrt(np.clip(Aw**2 + d * rho * (Aw + Ap), 0, None))) / d
    t = np.where(np.abs(d) < 1e-6, rho, t)
    return np.clip(t, 0, 1)


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


def sample(img, xyz):
    ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz).T
    return ndi.map_coordinates(np.asarray(img.dataobj, np.float32), ijk, order=1, mode="constant", cval=np.nan)


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
    # profile[metric] : (ndepth, nvert)
    prof = {m: np.stack([sample(imgs[m], W + equivol_t(r, Aw, Ap)[:, None] * (P - W)) for r in RHO])
            for m in MET}

    hinge = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int); hinge = hinge[parc[hinge] > 0]
    dmin = dijkstra(mesh_graph(W, f), indices=hinge.tolist(), min_only=True)
    lo = np.percentile(sulc[valid], 25)
    poolmask = (sulc <= lo) & (dmin > 6) & valid
    crown = np.where(poolmask)[0]
    # random, not index-even: see hinge_profile.py
    crown = np.sort(np.random.RandomState(sid).choice(crown, min(len(hinge), len(crown)),
                                                      replace=False))

    # display profiles (cortex-only, count-matched crown) and within-parcel diff (region-matched)
    # per gyral junction, not per landmark: landmarks within 6 mm are one junction
    # sampled repeatedly and their measuring disks overlap. Controls grouped to match.
    hv = hinge[valid[hinge]]
    grp = cluster_groups(sid, hv); sz = [len(g) for g in grp]
    cg = mirror_groups(len(crown), sz)
    Yh = {m: np.array([np.nanmean(group_mean(prof[m][d][hv], grp))
                       for d in range(len(RHO))]) for m in MET}
    Yc = {m: np.array([np.nanmean(group_mean(prof[m][d][crown], cg))
                       for d in range(len(RHO))]) for m in MET}
    dWP = {}
    gcache = {}                       # clusters are a property of the landmarks, not of depth
    for m in MET:
        cols = []
        for di in range(len(RHO)):
            vals = prof[m][di]; diffs = []
            for pid in np.unique(parc[valid]):
                reg = (parc == pid) & valid
                loc = np.percentile(sulc[reg], 25) if reg.sum() > 4 else 0
                hin = hinge[(parc[hinge] == pid) & valid[hinge]]
                cro = np.where(reg & (sulc <= loc) & (dmin > 6))[0]
                if len(hin) >= 5 and len(cro) >= 5:
                    sd = np.nanstd(vals[reg]) or 1
                    key = int(pid)
                    if key not in gcache:
                        gg = cluster_groups(sid, hin)
                        gcache[key] = (gg, mirror_groups(len(cro), [len(x) for x in gg]))
                    g, cgg = gcache[key]
                    diffs.append((np.nanmean(group_mean(vals[hin], g)) -
                                  np.nanmean(group_mean(vals[cro], cgg))) / sd)
            cols.append(np.mean(diffs) if diffs else np.nan)
        dWP[m] = np.array(cols)

    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{s}_depth.npz", rho=RHO, mets=np.array(list(MET)),
             **{f"Yh_{m}": Yh[m] for m in MET}, **{f"Yc_{m}": Yc[m] for m in MET},
             **{f"dWP_{m}": dWP[m] for m in MET})
    print(f"{s}: equivol depth profiles done ({len(hinge)} 3HG, {len(crown)} crown)", flush=True)


if __name__ == "__main__":
    for i in ([int(x) for x in sys.argv[1:]] or SUBJ):
        run(i)

#!/usr/bin/env python3
"""Ex-vivo vs in-vivo regional concordance of the 3HG - crown contrast (analysis pass).

Per DK parcel: delta_p = mean z(3HG landmarks in p) - mean z(crown landmarks in p), z within
subject/specimen over its pooled landmarks, averaged over subjects. Correlated across parcels
per metric. Decides whether the concordance supplement is worth a figure - not a figure itself.

  --exvivo    six hemispheres, hinge_profile machinery, parcel-matched crown (as Fig. 4b)
  --invivo N  first N cohort sessions, invivo_pipeline machinery
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy.sparse.csgraph import dijkstra

OUT = (_env.ROOT + "/surf/three_hinge/results")


def exvivo():
    import dk
    import hinge_profile as hp
    METS = {"FA": "FA", "MD": "ADC", "AD": "AD", "RD": "RD"}
    import dk_names
    acc = {}
    for sid in [1, 2, 3, 4, 5, 6]:
        s = f"Subject{sid}"
        v, f = nib.freesurfer.read_geometry(f"{hp.FS}/{s}/surf/{hp.H}.white")
        sulc = nib.freesurfer.read_morph_data(f"{hp.FS}/{s}/surf/{hp.H}.sulc")
        hinge = np.atleast_1d(np.loadtxt(f"{hp.GNET}/{s}/out/{hp.H}_3hinge_ids.txt", dtype=int))
        parc = dk.read(s)[0]; ctx = parc > 0
        hinge = hinge[ctx[hinge]]
        maps = {m: np.where(ctx, hp.clean_mid(s, k), np.nan) for m, k in METS.items()}
        G = hp.mesh_graph(v.astype(float), f)
        dmin = dijkstra(G, indices=hinge, min_only=True)
        lo = np.percentile(sulc[ctx], 25)
        pool = (sulc <= lo) & (dmin > hp.CROWN_FAR) & ctx
        rng = np.random.RandomState(sid)
        pick = []
        for p in np.unique(parc[hinge]):
            k = int((parc[hinge] == p).sum())
            cand = np.where(pool & (parc == p))[0]
            if len(cand):
                pick.append(rng.choice(cand, size=min(3 * k, len(cand)), replace=False))
        crown = np.sort(np.concatenate(pick))
        Hd = hp.disk_means(G, hinge, maps, hp.DISK_MM)
        Cd = hp.disk_means(G, crown, maps, hp.DISK_MM)
        for m in METS:
            allv = np.r_[Hd[m], Cd[m]]
            mu, sd = np.nanmean(allv), np.nanstd(allv) or 1
            zh, zc = (Hd[m] - mu) / sd, (Cd[m] - mu) / sd
            for p in np.unique(parc[hinge]):
                hz = zh[parc[hinge] == p]; cz = zc[parc[crown] == p]
                if len(hz) >= 3 and len(cz) >= 3:
                    acc.setdefault(m, {}).setdefault(int(p), []).append(
                        (np.nanmean(hz) - np.nanmean(cz), len(hz)))
        print(f"{s} done", flush=True)
    np.savez(f"{OUT}/concord_exvivo.npz",
             **{f"{m}|{p}": np.array(acc[m][p]) for m in acc for p in acc[m]})
    print("wrote concord_exvivo.npz")


def invivo(n):
    import invivo_pipeline as ip
    import pyvista as pv
    from scipy.spatial import cKDTree
    ses_all = [l.strip() for l in open((_env.AABC + "/cohort_v1_48_69.txt"))][:n]
    acc = {}
    for si, ses in enumerate(ses_all):
        S = f"{ip.ROOT}/{ses}"; out = f"{ip.DERIV}/{ses}"
        try:
            ref = nib.load(f"{out}/FA.nii.gz")
        except Exception:
            continue
        ainv = np.linalg.inv(ref.affine)
        maps = {m: nib.load(f"{out}/{m}.nii.gz").get_fdata() for m in ip.METS}
        rng = np.random.default_rng(1)
        HZ, CZ, HP, CP = [], [], [], []
        for h in ("lh", "rh"):
            G = f"{S}/postprocessing_tong/gii_native"
            HL = {"lh": "L", "rh": "R"}[h]
            white = pv.read(f"{G}/surf/{h}.white.resample.163842.vtk")
            feat = pv.read(f"{G}/gyralnet_island_resample_163842_scanner/{h}_surf_feature_updated.vtk")
            sulc = np.asarray(feat.point_data["sulc"])
            lab, _, names = nib.freesurfer.read_annot(f"{G}/label/{h}.aparc.resample.163842.annot")
            names = [x.decode() if isinstance(x, bytes) else x for x in names]
            bad = {i for i, x in enumerate(names) if x in ("unknown", "corpuscallosum")}
            cortex = ~np.isin(lab, list(bad)) & (lab >= 0)
            ids = np.atleast_1d(np.loadtxt(
                f"{G}/gyralnet_island_resample_163842_scanner/{h}_3hinge_ids.txt", dtype=int))
            ids = ids[cortex[ids]]
            nat_w = nib.load(f"{S}/sMRI/T1w/Native/{ses}.{HL}.white.native.surf.gii").darrays[0].data
            nat_m = nib.load(f"{S}/sMRI/T1w/Native/{ses}.{HL}.midthickness.native.surf.gii").darrays[0].data
            _, nn = cKDTree(nat_w).query(np.asarray(white.points), workers=4)
            mid = nat_m[nn]
            sgn = sulc * np.sign(np.median(sulc[ids]))
            thr = np.quantile(sgn[cortex], 0.75)
            g = ip.mesh_graph(white)
            dist = dijkstra(g, indices=ids, min_only=True, limit=ip.FAR_MM + ip.DISK_MM)
            pool = cortex & (sgn >= thr) & (dist > ip.FAR_MM)
            pick = []
            for p in np.unique(lab[ids]):
                k = int((lab[ids] == p).sum())
                cand = np.where(pool & (lab == p))[0]
                if len(cand):
                    pick.append(rng.choice(cand, size=min(2 * k, len(cand)), replace=False))
            crown = np.sort(np.concatenate(pick))

            def disk_means(seeds):
                o = np.empty((len(seeds), len(ip.METS)))
                for i, vv in enumerate(seeds):
                    dd = dijkstra(g, indices=int(vv), limit=ip.DISK_MM)
                    disk = np.where(np.isfinite(dd))[0]; disk = disk[cortex[disk]]
                    xyz = mid[disk]
                    for j, m in enumerate(ip.METS):
                        o[i, j] = float(np.mean(ip.trilinear(maps[m], ainv, xyz)))
                return o
            HZ.append(disk_means(ids)); CZ.append(disk_means(crown))
            # DK label NAME so lh/rh pool; ids -> parcel name index table shared
            HP += [names[l] for l in lab[ids]]
            CP += [names[l] for l in lab[crown]]
        HV, CV = np.vstack(HZ), np.vstack(CZ)
        HP, CP = np.array(HP), np.array(CP)
        for j, m in enumerate(ip.METS):
            allv = np.r_[HV[:, j], CV[:, j]]
            mu, sd = np.nanmean(allv), np.nanstd(allv) or 1
            zh, zc = (HV[:, j] - mu) / sd, (CV[:, j] - mu) / sd
            for p in np.unique(HP):
                hz = zh[HP == p]; cz = zc[CP == p]
                if len(hz) >= 3 and len(cz) >= 3:
                    acc.setdefault(m, {}).setdefault(p, []).append(np.nanmean(hz) - np.nanmean(cz))
        if (si + 1) % 10 == 0:
            print(f"{si+1}/{len(ses_all)}", flush=True)
    np.savez(f"{OUT}/concord_invivo.npz",
             **{f"{m}|{p}": np.array(v) for m in acc for p, v in acc[m].items()})
    print("wrote concord_invivo.npz")


if __name__ == "__main__":
    if sys.argv[1] == "--exvivo":
        exvivo()
    else:
        invivo(int(sys.argv[2]))

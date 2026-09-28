#!/usr/bin/env python3
"""In-vivo validation, experiment 1: the 3HG-vs-crown tensor phenotype in AABC.

Per session: fit the tensor (b0 + b1500 shell, WLS) in the native T1w-acpc space the HCP
pipeline already registered the dwi to, then read FA/MD/AD/RD at every GyralNet 3HG and at
region-matched gyral-crown controls, with the ex-vivo estimator transplanted:

  landmark value  = geodesic-disk mean (DISK_MM) of the map, sampled at the MID-THICKNESS
                    coordinate of each disk vertex (trilinear). One depth, not nine - at
                    1.5 mm the laminar axis is the ex-vivo paper's territory, not this one's.
  crown control   = per-parcel draw matched to the 3HG parcel histogram, from the most-gyral
                    sulc quartile, >= FAR_MM geodesic from any 3HG, medial wall excluded,
                    NDRAW draws averaged.
  subject summary = Cohen's d over landmarks (both hemispheres pooled), one d per metric.

Geometry and selection live on the 163842 resampled mesh (what the 3hinge ids index); the
sampling coordinate comes from the native midthickness surface via nearest white vertex -
both sit in the same physical space (verified: bboxes agree to 0.3 mm).

Sulc sign is self-calibrated per hemisphere: whichever side of zero the 3HGs' median sulc
falls on is the gyral direction. (HCP and FreeSurfer disagree on the sign convention, and
a wrong guess silently inverts the crown pool - the ex-vivo medial-wall lesson, sign edition.)

Usage: invivo_pipeline.py SESSION [SESSION...]      writes DERIV/<ses>/landmarks.npz
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

ROOT = (_env.AABC + "/organized")
DERIV = (_env.AABC + "/derived")
DISK_MM = 3.0        # ex-vivo used 2 mm on 0.8 mm dwi; scaled to the 1.5 mm voxel
FAR_MM = 8.0         # crown pool must be this far from any 3HG (ex-vivo value)
NDRAW = 20           # crown draws averaged (ex-vivo value)
BMAX = 1600          # tensor fit on b0 + b1500 only; b3000 biases the tensor
METS = ["FA", "MD", "AD", "RD"]
RNG = np.random.default_rng(0)


def fit_dti(S, out):
    """WLS tensor on the low shell; writes FA/MD/AD/RD next to the session's derived dir."""
    if all(os.path.exists(f"{out}/{m}.nii.gz") for m in METS):
        return
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    D = f"{S}/dMRI/T1w/Diffusion"
    img = nib.load(f"{D}/data.nii.gz")
    bval = np.loadtxt(f"{D}/bvals"); bvec = np.loadtxt(f"{D}/bvecs")
    keep = bval < BMAX
    data = np.asarray(img.dataobj)[..., keep]
    mask = nib.load(f"{D}/nodif_brain_mask.nii.gz").get_fdata() > 0
    gtab = gradient_table(bval[keep], bvecs=bvec[:, keep].T)
    fit = TensorModel(gtab, fit_method="WLS").fit(data, mask=mask)
    os.makedirs(out, exist_ok=True)
    for m, vol in (("FA", fit.fa), ("MD", fit.md), ("AD", fit.ad), ("RD", fit.rd)):
        nib.save(nib.Nifti1Image(np.nan_to_num(vol).astype(np.float32), img.affine),
                 f"{out}/{m}.nii.gz")


def read_vtk_points_lines(path):
    """minimal ASCII vtk polydata reader: points + line connectivity (v5 offsets or legacy)"""
    import pyvista as pv
    m = pv.read(path)
    lines = m.lines.reshape(-1, 3)[:, 1:] if m.lines.size else np.empty((0, 2), int)
    return np.asarray(m.points), lines, m


def mesh_graph(m):
    f = m.faces.reshape(-1, 4)[:, 1:]
    e = np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]
    w = np.linalg.norm(m.points[e[:, 0]] - m.points[e[:, 1]], axis=1)
    n = m.n_points
    g = sparse.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(n, n)).tocsr()
    return g


def trilinear(vol, ainv, xyz):
    ijk = (ainv[:3, :3] @ xyz.T + ainv[:3, 3:4])
    from scipy.ndimage import map_coordinates
    return map_coordinates(vol, ijk, order=1, mode="nearest")


def hemi(S, ses, h, maps, ainv):
    """returns (hinge_vals, crown_vals) arrays: n_landmarks x n_metrics"""
    import pyvista as pv
    G = f"{S}/postprocessing_tong/gii_native"
    HL = {"lh": "L", "rh": "R"}[h]
    white = pv.read(f"{G}/surf/{h}.white.resample.163842.vtk")
    feat = pv.read(f"{G}/gyralnet_island_resample_163842_scanner/{h}_surf_feature_updated.vtk")
    sulc = np.asarray(feat.point_data["sulc"])
    lab, _, names = nib.freesurfer.read_annot(f"{G}/label/{h}.aparc.resample.163842.annot")
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    bad = {i for i, n in enumerate(names) if n in ("unknown", "corpuscallosum")}
    cortex = ~np.isin(lab, list(bad)) & (lab >= 0)
    ids = np.atleast_1d(np.loadtxt(
        f"{G}/gyralnet_island_resample_163842_scanner/{h}_3hinge_ids.txt", dtype=int))
    ids = ids[cortex[ids]]

    # mid-thickness coordinate per 163842 vertex, via nearest native white vertex
    nat_w = nib.load(f"{S}/sMRI/T1w/Native/{ses}.{HL}.white.native.surf.gii").darrays[0].data
    nat_m = nib.load(f"{S}/sMRI/T1w/Native/{ses}.{HL}.midthickness.native.surf.gii").darrays[0].data
    from scipy.spatial import cKDTree
    _, nn = cKDTree(nat_w).query(np.asarray(white.points), workers=4)
    mid = nat_m[nn]

    # gyral direction: the side the 3HGs sit on
    gyr_sign = np.sign(np.median(sulc[ids]))
    sgn = sulc * gyr_sign                              # larger = more gyral
    thr = np.quantile(sgn[cortex], 0.75)
    g = mesh_graph(white)
    dist = dijkstra(g, indices=ids, min_only=True, limit=FAR_MM + DISK_MM)
    pool = cortex & (sgn >= thr) & (dist > FAR_MM)

    # per-parcel crown draws matched to the 3HG parcel histogram
    draws = []
    for _ in range(NDRAW):
        pick = []
        for p in np.unique(lab[ids]):
            k = int((lab[ids] == p).sum())
            cand = np.where(pool & (lab == p))[0]
            if len(cand):
                pick.append(RNG.choice(cand, size=min(k, len(cand)), replace=False))
        draws.append(np.concatenate(pick) if pick else np.array([], int))

    def disk_means(seeds):
        out = np.empty((len(seeds), len(METS)))
        for i, v in enumerate(seeds):
            dd = dijkstra(g, indices=v, limit=DISK_MM)
            disk = np.where(np.isfinite(dd))[0]
            disk = disk[cortex[disk]]
            xyz = mid[disk]
            for j, m in enumerate(METS):
                out[i, j] = float(np.mean(trilinear(maps[m], ainv, xyz)))
        return out

    hv = disk_means(ids)
    # dedupe crown vertices across draws so each disk is computed once
    uniq = np.unique(np.concatenate([d for d in draws if len(d)]))
    uv = disk_means(uniq)
    lut = dict(zip(uniq.tolist(), range(len(uniq))))
    cv = np.mean([uv[[lut[v] for v in d]].mean(0) for d in draws if len(d)], axis=0)
    cs = np.concatenate([uv[[lut[v] for v in d]] for d in draws if len(d)])
    return hv, cv, cs


def run(ses):
    S = f"{ROOT}/{ses}"; out = f"{DERIV}/{ses}"
    fit_dti(S, out)
    ref = nib.load(f"{out}/FA.nii.gz")
    ainv = np.linalg.inv(ref.affine)
    maps = {m: nib.load(f"{out}/{m}.nii.gz").get_fdata() for m in METS}
    HV, CS = [], []
    for h in ("lh", "rh"):
        hv, cv, cs = hemi(S, ses, h, maps, ainv)
        HV.append(hv); CS.append(cs)
    HV, CS = np.vstack(HV), np.vstack(CS)
    sp = np.sqrt((HV.var(0, ddof=1) + CS.var(0, ddof=1)) / 2)
    d = (HV.mean(0) - CS.mean(0)) / sp
    np.savez(f"{out}/landmarks.npz", hinge=HV, crown_draws=CS, mets=METS, d=d,
             n_hinge=len(HV))
    print(f"{ses}  n3HG={len(HV)}  " +
          "  ".join(f"{m} d={v:+.3f}" for m, v in zip(METS, d)), flush=True)
    return d


if __name__ == "__main__":
    for ses in sys.argv[1:]:
        run(ses)

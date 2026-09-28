"""Superficial white-matter fiber density beneath 3HG vs beneath ordinary gyral crowns.

Whole-brain tractography does not exist in this dataset (only Subject4's IFOF/ILF), so the
'structural-hinge / fiber-convergence' hypothesis is tested with the closest available proxy:
sample the DTI fiber-density (FD) and the DBSI anisotropic-fiber fraction in the SUPERFICIAL
WM just below the white surface (0.5-3 mm inward along the surface normal), beneath each 3HG
vs beneath cortex-only, count-matched gyral crowns. Higher sub-WM FD beneath 3HG = more fiber
mass converging under the hinge. Per specimen; specimen is the unit."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import sparse, stats
from scipy.sparse.csgraph import dijkstra
from scipy import ndimage as ndi
import dk

ROOT = (_env.ROOT + "/28211843"); FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet"); H = "rh"
SUBJ = [1, 2, 3, 4, 5, 6]; DEPTHS = np.arange(0.5, 3.01, 0.5)   # mm below white, into WM
VOLS = {"FD": "{s}_DTI/FD.nii.gz", "DBSIfiber": "{s}_DBSI/fiber_ratio_map.nii.gz"}


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


def sample_vol(img, xyz):
    inv = np.linalg.inv(img.affine)
    ijk = nib.affines.apply_affine(inv, xyz).T
    return ndi.map_coordinates(np.asarray(img.dataobj, np.float32), ijk, order=1,
                               mode="constant", cval=np.nan)


def subwm(sid):
    import pyvista as pv
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk2ras = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    vw = nib.affines.apply_affine(tk2ras, v)                       # world coords
    m = pv.PolyData(v.astype(float), np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = (tk2ras[:3, :3] @ m.point_data["Normals"].T).T           # normals -> world
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    # ensure outward (away from surface centroid), then go inward = -normal
    if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    out = {}
    for name, tpl in VOLS.items():
        img = nib.load(f"{ROOT}/{s}/{tpl.format(s=s)}")
        prof = np.stack([sample_vol(img, vw - t * nrm) for t in DEPTHS])   # (depth, vert)
        out[name] = np.nanmean(prof, 0)
    return v, f, out


def run():
    print(f"Sub-WM fiber density beneath 3HG vs crown (0.5-3 mm below white), per specimen d:")
    print(f"{'metric':12s}  " + "  ".join(f"S{i}" for i in SUBJ) + f"   {'mean':>6s} {'paired p':>8s} {'6/6':>4s}")
    res = {k: [] for k in VOLS}
    for sid in SUBJ:
        v, f, vals = subwm(sid)
        v = v.astype(float)
        sulc = nib.freesurfer.read_morph_data(f"{FS}/Subject{sid}/surf/{H}.sulc")
        parc = dk.read(f"Subject{sid}")[0]; ctx = parc > 0
        hinge = np.atleast_1d(np.loadtxt(f"{GNET}/Subject{sid}/out/{H}_3hinge_ids.txt", dtype=int))
        hinge = hinge[ctx[hinge]]                   # drop medial-wall 3HG
        dmin = dijkstra(mesh_graph(v, f), indices=hinge.tolist(), min_only=True)
        lo = np.percentile(sulc[ctx], 25)
        pool = np.where((sulc <= lo) & (dmin > 8) & ctx)[0]
        crown = pool[np.linspace(0, len(pool) - 1, len(hinge)).astype(int)]
        for k in VOLS:
            a = vals[k][hinge]; b = vals[k][crown]
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            res[k].append((a.mean() - b.mean()) / sd if sd else np.nan)
    for k in VOLS:
        d = np.array(res[k]); p = stats.ttest_1samp(d, 0).pvalue
        cons = "yes" if (np.all(d > 0) or np.all(d < 0)) else "no"
        print(f"{k:12s}  " + "  ".join(f"{x:+.2f}" for x in d) + f"   {d.mean():+6.2f} {p:8.4f} {cons:>4s}")


if __name__ == "__main__":
    run()

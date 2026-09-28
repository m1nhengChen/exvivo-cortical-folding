#!/usr/bin/env python3
"""Export reconstructed surfaces to legacy VTK PolyData.

Anatomical surfaces (white, pial, mid) are written in world (scanner RAS) coordinates with
outward normals; mid is the midpoint of white and pial.  prep_volumes.py builds an RAS+ grid
while FreeSurfer writes vox2ras_tkr in LIA form, so tkr is left-handed and det(tk2ras) < 0;
the triangle winding is flipped when the determinant is negative so normals face outward.
tk2ras carries no scale, so metric quantities (vertex area) are computed in world coordinates.

Spherical surfaces (sphere, sphere.reg) are written in their native FreeSurfer coordinates;
sphere.reg shares the fsaverage frame, so its vertex correspondence is directly usable.

Every surface carries the same per-vertex morphometrics (all FreeSurfer-derived), indexed by
the shared white-surface vertex order: thickness, curv, sulc, area, and the DK parcellation.

    ./export_vtk.py             all twelve
    ./export_vtk.py 1 2 3
"""
import os, sys, argparse
import numpy as np
import nibabel as nib
import pyvista as pv

# defaults are relative to the directory that holds this repository; override with the
# environment variables
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FS = os.environ.get("EXVIVO_FS", os.path.join(_PROJECT, "surf/fs"))
OUT = os.environ.get("EXVIVO_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
H = os.environ.get("HEMI", "rh")           # ds006001 whole brains contribute left hemispheres too
ANATOMICAL = ["white", "pial", "mid"]      # world coordinates
SPHERICAL = ["sphere", "sphere.reg"]       # native FreeSurfer coordinates
MORPH = ["thickness", "curv", "sulc"]


def tk2ras(s):
    m = nib.load(f"{FS}/{s}/mri/wm.mgz")
    return m.header.get_affine() @ np.linalg.inv(m.header.get_vox2ras_tkr())


def vertex_area(V, F):
    """Barycentric vertex area: a third of each incident triangle's area."""
    tri = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a3 = 0.5 * np.linalg.norm(tri, axis=1) / 3.0
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a3)
    return out


def point_arrays(s, Vw, Fw):
    """FreeSurfer morphometrics on the white-surface vertex order.  area from world geometry."""
    out = {}
    for m in MORPH:
        p = f"{FS}/{s}/surf/{H}.{m}"
        if os.path.exists(p):
            v = nib.freesurfer.read_morph_data(p)
            if len(v) == len(Vw):
                out[m] = v
    out["area"] = vertex_area(Vw, Fw)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
    annot = f"{FS}/{s}/label/{H}.aparc.annot"
    if os.path.exists(annot):
        try:
            import dk
            lab, _, _ = dk.read(s, f"{H}.aparc.annot")   # DK ids: 0=wall, 2001-2035
            if len(lab) == len(Vw):
                # 0-based labels: 0 = medial wall / unknown, region k = DK id 2000+k
                out["parcel"] = np.where(lab >= 2001, lab - 2000, 0).astype(np.int32)
        except Exception as e:
            print(f"  {s}: aparc.annot skipped ({e})")
    return out


def polydata(V, F):
    return pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel())


def run(sid):
    s = f"Subject{sid}"
    if not os.path.exists(f"{FS}/{s}/mri/wm.mgz"):
        print(f"{s}: missing, skipped")
        return
    tk = tk2ras(s)
    flip = np.linalg.det(tk[:3, :3]) < 0
    d = f"{OUT}/{s}"
    os.makedirs(d, exist_ok=True)

    Vw, Fw = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    Wwhite = nib.affines.apply_affine(tk, Vw)
    Fout = Fw[:, [0, 2, 1]] if flip else Fw
    arrays = point_arrays(s, Wwhite, Fout)

    wrote = []
    # anatomical surfaces: world coordinates, mid = midpoint of white and pial.
    # pial may not exist yet when this is run mid-rollout as a preview; export white alone
    # rather than failing, since white is the surface the gyral QC actually looks at.
    tkr = {"white": Vw}
    if os.path.exists(f"{FS}/{s}/surf/{H}.pial"):
        Vp, _ = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.pial")
        tkr.update({"pial": Vp, "mid": 0.5 * (Vw + Vp)})
    for g in [g for g in ANATOMICAL if g in tkr]:
        mesh = polydata(nib.affines.apply_affine(tk, tkr[g]), Fout)
        for k, v in arrays.items():
            mesh.point_data[k] = np.array(v)      # copy: pyvista shares numpy buffers zero-copy,
                                                  # so reusing one array across meshes aliases it
        mesh.save(f"{d}/{H}.{g}.vtk", binary=True)
        wrote.append(g)

    # spherical surfaces: native FreeSurfer coordinates
    for g in SPHERICAL:
        p = f"{FS}/{s}/surf/{H}.{g}"
        if not os.path.exists(p):
            continue
        Vs, Fs = nib.freesurfer.read_geometry(p)
        mesh = polydata(Vs, Fs[:, [0, 2, 1]] if flip else Fs)
        if len(Vs) == len(Vw):
            for k, v in arrays.items():
                mesh.point_data[k] = np.array(v)      # copy, not the shared buffer
        mesh.save(f"{d}/{H}.{g}.vtk", binary=True)
        wrote.append(g)

    print(f"{s}: {' '.join(wrote)}  arrays: {sorted(arrays)}  det(tk2ras) {np.linalg.det(tk[:3,:3]):+.0f}"
          f"{'  [winding flipped]' if flip else ''}")


def write_parcel_labels():
    """Sidecar mapping the 0-based parcel index to its DK id and name."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
    import dk
    names = dk.id2name()
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/parcel_labels.tsv", "w") as f:
        f.write("index\tdk_id\tname\n")
        f.write("0\t0\tunknown\n")
        for k in range(1, 36):
            if 2000 + k in names:
                f.write(f"{k}\t{2000+k}\t{names[2000+k]}\n")
    print(f"wrote {OUT}/parcel_labels.tsv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subjects", nargs="*", type=int, default=list(range(1, 13)))
    a = p.parse_args()
    for i in a.subjects:
        run(i)
    write_parcel_labels()

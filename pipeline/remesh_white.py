#!/usr/bin/env python3
"""Remesh the finished white surface to a clean, uniform triangulation, then re-place
the pial from it.

The white surface is image-accurate but over-tessellated: mri_tessellate places
~one vertex per 0.5 mm voxel (420-520k), and the voxel-corner staircase reads as
roughness.  Rather than decimate before topology correction (which places the surface
on a coarse mesh and loses image accuracy), keep the image-accurate white and clean up
only the mesh afterwards with an isotropic remesh -- uniform edge length,
well-conditioned triangles -- then deform the pial from it.

Isotropic explicit remeshing (equal-edge target) is the pymeshlab stand-in for the
Geogram CVT in remesh.py (vorpalite is unavailable); the goal of uniform,
near-equilateral triangles is the same.

Report area and one-sided Hausdorff against the pre-remesh surface: a remesh that
rounds off sulcal fundi would look smooth while deleting the folds.
"""
import os, sys, numpy as np, nibabel as nib, pymeshlab as ml
from scipy.spatial import cKDTree

# defaults are relative to the directory that holds this repository; override with the
# environment variables
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FS = os.environ.get("EXVIVO_FS", os.path.join(_PROJECT, "surf/fs"))
H = "rh"


def area(V, F):
    return np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / 2


def remesh(V, F, edge_mm, iters=8, taubin=5):
    """Isotropic remesh to uniform triangles, then Taubin-smooth.

    The remesh alone does not reduce roughness, which is a property of the surface
    shape (voxel staircase in the vertex positions), not the triangulation.  It yields
    uniform triangles, on which Taubin smoothing is unbiased (on the raw
    ~one-vertex-per-voxel mesh it would be pulled toward dense regions).  Taubin's
    lambda/mu low-pass is volume-preserving, so it does not shrink the folds away.
    """
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=V.astype(np.float64), face_matrix=F.astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iters,
        targetlen=ml.PureValue(edge_mm),
        adaptive=False,          # uniform, not curvature-adaptive: even triangles
        checksurfdist=False)
    if taubin > 0:
        ms.apply_coord_taubin_smoothing(stepsmoothnum=int(taubin))
    m = ms.mesh(0)
    return m.vertex_matrix(), m.face_matrix()


def run(sid, edge_mm=0.7, surf="white", taubin=5, out="white.clean", subj=None):
    s = subj if subj else f"Subject{sid}"
    V, F = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.{surf}")
    V2, F2 = remesh(V, F, edge_mm, taubin=taubin)

    # topology + geometry check
    e = np.unique(np.sort(np.vstack([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]]), 1), axis=0)
    g = (2 - (len(V2) - len(e) + len(F2))) // 2
    d, _ = cKDTree(V2).query(V)                       # old vertices -> remeshed surface
    print(f"{s}: {len(V)} -> {len(V2)} 顶点 (edge {edge_mm}mm) | genus {g} | "
          f"面积 {area(V,F)/100:.0f}->{area(V2,F2)/100:.0f} cm² | "
          f"位移 中位 {np.median(d):.2f} 95% {np.percentile(d,95):.2f} >2mm {100*(d>2).mean():.1f}%")

    nib.freesurfer.write_geometry(f"{FS}/{s}/surf/{H}.{out}",
                                  V2.astype(np.float32), F2.astype(np.int32))
    return g


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("subjects", nargs="*", type=int, default=list(range(1, 7)))
    p.add_argument("--edge", type=float, default=0.7)
    p.add_argument("--surf", default="white")
    p.add_argument("--taubin", type=int, default=5)
    p.add_argument("--out", default="white.clean")
    a = p.parse_args()
    for i in a.subjects:
        run(i, a.edge, a.surf, a.taubin, a.out)

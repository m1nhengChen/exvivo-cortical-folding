#!/usr/bin/env python3
"""Find the sphere's anatomical pose by searching rotations against fsaverage.

Global rigid alignment to seed mris_register, whose rigid search spans only +-64 degrees and
cannot recover an arbitrary specimen pose on its own.  Rotate the subject sphere over a
coverage of SO(3), score each rotation by how well subject sulcal depth (and curv) match
fsaverage's at corresponding points, keep the best; mris_register then does the fine,
non-rigid part starting inside its convergence basin.

Chirality fix: prep_volumes.py builds an RAS+ grid while FreeSurfer writes vox2ras_tkr in LIA
form, so tkr is left-handed and every FreeSurfer-written surface, the sphere included, is
mirrored.  Negating x undoes it; the face winding is flipped to keep normals outward.

The KD-tree is built once on the subject sphere and the query points are rotated instead of
the surface, so each rotation costs one batched nearest-neighbour query.

    ./sphere_rotsearch.py 1
    ./sphere_rotsearch.py 7 8 9 10 11 12
"""
import os, sys, argparse
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

# defaults are relative to the directory that holds this repository; override with the
# environment variables
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FS = os.environ.get("EXVIVO_FS", os.path.join(_PROJECT, "surf/fs"))
FSAVG = f"{FS}/fsaverage"
H = "rh"
# Sizes chosen so the top candidate lands inside mris_register's +-64 deg basin.
NPROBE = 20000         # fsaverage vertices used to score a rotation
COARSE = 20000         # rotations in the global sweep
FINE_DEG, FINE_N = 10.0, 2000
CURV_W = 0.5           # curv weight; adds detail sulc alone cannot disambiguate


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def random_rotations(n, rng):
    """Uniform on SO(3) via random quaternions."""
    q = rng.normal(size=(n, 4))
    q = unit(q)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def perturbations(R0, n, deg, rng):
    """Rotations within `deg` of R0, for the refinement pass."""
    ax = unit(rng.normal(size=(n, 3)))
    ang = np.deg2rad(deg) * rng.random(n) ** (1 / 3)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    I = np.eye(3)[None]
    R = I + np.sin(ang)[:, None, None] * K + (1 - np.cos(ang))[:, None, None] * (K @ K)
    return R @ R0


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / d if d > 0 else -1.0


def score_rotations(tree, feat_s, P, feat_a, R, curv_s=None, curv_a=None):
    """Correlation between the subject feature at the nearest rotated-probe vertex and
    fsaverage's own feature.  Rotating the query points equals rotating the surface and
    lets the tree be built once."""
    out = np.empty(len(R))
    for i, r in enumerate(R):
        _, j = tree.query(P @ r.T, workers=-1)
        v = _corr(feat_s[j], feat_a)
        if curv_s is not None:
            v += CURV_W * _corr(curv_s[j], curv_a)
        out[i] = v
    return out


def run(sid, seed=0, write=True, suffix=""):
    s = f"Subject{sid}"
    S, F = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.sphere")
    feat_s = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
    curv_s = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.curv")
    A, _ = nib.freesurfer.read_geometry(f"{FSAVG}/surf/{H}.sphere")
    feat_a = nib.freesurfer.read_morph_data(f"{FSAVG}/surf/{H}.sulc")
    curv_a = nib.freesurfer.read_morph_data(f"{FSAVG}/surf/{H}.curv")

    S = S * [-1, 1, 1]                       # un-mirror; winding flips below
    F = F[:, [0, 2, 1]]
    Su = unit(S - S.mean(0))
    Au = unit(A - A.mean(0))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Au), min(NPROBE, len(Au)), replace=False)
    P, fa, ca = Au[idx], feat_a[idx], curv_a[idx]
    tree = cKDTree(Su)

    R = random_rotations(COARSE, rng)
    sc = score_rotations(tree, feat_s, P, fa, R, curv_s, ca)
    best = R[sc.argmax()]
    R2 = perturbations(best, FINE_N, FINE_DEG, rng)
    sc2 = score_rotations(tree, feat_s, P, fa, R2, curv_s, ca)
    if sc2.max() > sc.max():
        best, bestsc = R2[sc2.argmax()], sc2.max()
    else:
        bestsc = sc.max()

    # score of the existing parcellation-derived pose, when it exists
    ref = ""
    p = f"{FS}/{s}/surf/{H}.sphere.prealign"
    if os.path.exists(p):
        Sp, _ = nib.freesurfer.read_geometry(p)
        Spu = unit(Sp - Sp.mean(0))
        _, j = cKDTree(Spu).query(P)
        ref = f"   prep_sphere 参照 {_corr(feat_s[j], fa) + CURV_W * _corr(curv_s[j], ca):+.3f}"

    if write:
        out = Su * 100.0
        nib.freesurfer.write_geometry(f"{FS}/{s}/surf/{H}.sphere.rotsearch{suffix}",
                                      (out @ best.T).astype(np.float32), F)
    print(f"{s}: 粗搜索 {sc.max():+.3f} -> 精修 {bestsc:+.3f}{ref}")
    return bestsc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("subjects", nargs="*", type=int, default=[1])
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-suffix", default="", help="write to rh.sphere.rotsearch<suffix>")
    a = p.parse_args()
    for i in a.subjects:
        run(i, seed=a.seed, write=not a.no_write, suffix=a.out_suffix)

#!/usr/bin/env python3
"""Cortex-label transfer and parcellation cleanup around mris_ca_label.

  cortex        Pull fsaverage's rh.cortex.label back through rh.sphere.reg (nearest vertex
                on the registered sphere).  mris_ca_label classifies only inside this mask,
                so an incomplete mask drops cortex from the parcellation.

  clean         mris_ca_label votes per vertex with no spatial regularisation, so the raw
                output is speckled.  Fill holes, majority-filter, drop islands, then restore
                any label the filter erased outright.

Run cortex after mris_register and before mris_ca_label; run clean after.
"""
import os, sys, numpy as np, nibabel as nib
from scipy.spatial import cKDTree
from mesh_utils import adjacency, fill_holes, majority_filter, drop_islands, restore_lost, MAJORITY_PASSES, MIN_ISLAND

# defaults are relative to the directory that holds this repository; override with the
# environment variables
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FS = os.environ.get("EXVIVO_FS", os.path.join(_PROJECT, "surf/fs"))
FSAVG = os.environ.get("EXVIVO_FSAVG", f"{FS}/fsaverage")
H = "rh"


def write_label(path, V, idx):
    with open(path, "w") as f:
        f.write("#!ascii label\n%d\n" % len(idx))
        for i in idx:
            f.write(f"{i}  {V[i,0]:.3f}  {V[i,1]:.3f}  {V[i,2]:.3f} 0.0000000000\n")


def cortex(sid):
    s = f"Subject{sid}"
    V, _ = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    S, _ = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.sphere.reg")
    A, _ = nib.freesurfer.read_geometry(f"{FSAVG}/surf/{H}.sphere")
    keep = np.zeros(len(A), bool)
    keep[nib.freesurfer.read_label(f"{FSAVG}/label/{H}.cortex.label")] = True

    _, j = cKDTree(A).query(S)                     # each subject vertex -> nearest fsaverage vertex
    ctx = np.flatnonzero(keep[j])
    for name in (f"{H}.cortex.fs.label", f"{H}.cortex.label"):
        write_label(f"{FS}/{s}/label/{name}", V, ctx)
    print(f"{s}: cortex {100*len(ctx)/len(V):.1f}%  (medial wall {100*(1-len(ctx)/len(V)):.1f}%)")


def clean(sid):
    s = f"Subject{sid}"
    _, F = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    lab, ctab, names = nib.freesurfer.read_annot(f"{FS}/{s}/label/{H}.aparc.raw.annot")
    nv = len(lab)
    indptr, nbr = adjacency(F, nv)

    l = np.where(lab < 0, 0, lab).astype(np.int32)
    l = fill_holes(l, indptr, nbr)
    orig = l.copy()
    l = majority_filter(l, indptr, nbr, MAJORITY_PASSES)
    l = drop_islands(l, indptr, nbr, MIN_ISLAND)
    l = restore_lost(l, orig, indptr, nbr)

    nib.freesurfer.write_annot(f"{FS}/{s}/label/{H}.aparc.annot", l, ctab, names,
                               fill_ctab=False)
    n = len(set(l[l > 0].tolist()))
    print(f"{s}: {n} parcels")


if __name__ == "__main__":
    fn = {"cortex": cortex, "clean": clean}[sys.argv[1]]
    for i in (range(1, 7) if len(sys.argv) < 3 else [int(x) for x in sys.argv[2:]]):
        fn(i)

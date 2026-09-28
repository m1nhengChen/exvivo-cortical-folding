#!/usr/bin/env python3
"""Where do streamlines stop, relative to the white surface?

The two pipelines disagree (seed contrast 1.23x with DIPY, 1.03x with MRtrix iFOD2) and the two
obvious explanations are dead: max_cross does not matter (1 -> all raises it, 1.18x -> 1.23x) and
neither does the propagation rule (deterministic peaks 1.18x, deterministic maximum 1.08x,
probabilistic 1.18x, all on one fODF, one set of seeds, one stopping criterion).

What has not been isolated is the STOPPING CRITERION, and the estimator makes it a live suspect:
a connection is scored when a streamline's distal ENDPOINT falls within 4 mm of a node, and the
nodes sit 2 mm inside the white surface. DIPY stops on FA < 0.12, which is crossed at the
grey/white boundary - exactly the depth of the nodes. MRtrix stops on FOD amplitude < 0.05, which
is not tied to that boundary at all.

So: the signed distance from every streamline endpoint to the white surface, per pipeline. Positive
is outside the white surface (into cortex), negative is deeper in the white matter. If DIPY's
endpoints pile up at the node depth and MRtrix's do not, the disagreement is a property of where
tracking stops, not of how it propagates."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
import dk
FS = (_env.ROOT + "/surf/fs")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
OUT = (_env.ROOT + "/surf/three_hinge/validate")
SUBJ = [1, 2, 3, 4, 5, 6]; H = "rh"
SRCS = [("DIPY deterministic", "whole_brain.trk"),
        ("MRtrix SD_Stream", "mrtrix/tracks_det.tck"),
        ("MRtrix iFOD2", "mrtrix/tracks.tck")]
EDGES = np.arange(-12, 8.01, 0.5)                 # mm, signed distance to the white surface
INWARD = 2.0                                      # the node depth, for reference


def surface(sid):
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    vw = nib.affines.apply_affine(tk, v.astype(float))
    import pyvista as pv
    m = pv.PolyData(vw, np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = m.point_data["Normals"]
    if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm                                # outward, toward pial
    ctx = dk.read(s)[0] > 0
    return vw, nrm, ctx


def ends(sid, rel):
    p = f"{TR}/Subject{sid}/{rel}"
    if rel.endswith(".trk"):
        from dipy.io.streamline import load_tractogram
        from dipy.io.stateful_tractogram import Space
        t = load_tractogram(p, nib.load(f"{TR}/Subject{sid}/dwi.nii.gz"), bbox_valid_check=False)
        t.to_space(Space.RASMM)
        sl = t.streamlines
    else:
        sl = nib.streamlines.load(p).streamlines
    return np.vstack([np.r_[[np.asarray(x)[0]], [np.asarray(x)[-1]]] for x in sl])


def main():
    Hgram = np.full((len(SRCS), len(SUBJ), len(EDGES) - 1), np.nan)
    for j, sid in enumerate(SUBJ):
        vw, nrm, ctx = surface(sid)
        tree = cKDTree(vw[ctx]); idx_ctx = np.where(ctx)[0]
        for i, (name, rel) in enumerate(SRCS):
            ep = ends(sid, rel)
            d, k = tree.query(ep)
            vid = idx_ctx[k]
            # signed: project the offset from the nearest vertex onto that vertex's outward normal
            sd = np.sum((ep - vw[vid]) * nrm[vid], 1)
            sd = sd[d < 6.0]                       # endpoints far from any cortex say nothing here
            Hgram[i, j] = np.histogram(sd, bins=EDGES, density=True)[0]
            print(f"  S{sid} {name:20s} {len(sd):>8,} endpoints near cortex, "
                  f"median {np.median(sd):+.2f} mm, within 1 mm of the node depth: "
                  f"{100*np.mean(np.abs(sd + INWARD) < 1.0):.1f}%", flush=True)
    np.savez(f"{OUT}/endpoint_depth.npz", srcs=np.array([s[0] for s in SRCS]),
             edges=EDGES, hist=Hgram)
    x = (EDGES[:-1] + EDGES[1:]) / 2
    print(f"\n{'':22s}{'median':>9s}{'|d+2|<1mm':>11s}{'peak at':>9s}")
    for i, (name, _) in enumerate(SRCS):
        m = np.nanmean(Hgram[i], 0)
        band = (x > -3) & (x < -1)
        print(f"{name:22s}{x[np.argmax(m)]:>+9.2f}{100*m[band].sum()*0.5:>10.1f}%"
              f"{x[np.argmax(m)]:>+9.2f}")
    print(f"wrote {OUT}/endpoint_depth.npz")


if __name__ == "__main__":
    main()

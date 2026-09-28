#!/usr/bin/env python3
"""Equivolumetric depth sampling, cut through a three-hinge junction.

    python3 render_lamina.py            # Subject4, rh

The intracortical analysis samples nine surfaces between white and pial at equal VOLUME fractions
rho = 0.1 ... 0.9 (Bok / Waehnert): the metric distance for a given volume fraction is corrected by
the local white and pial vertex areas, so each shell encloses the same amount of cortex. That
correction matters most exactly where 3HG sit - the crown of a tight fold, where equidistant
sampling puts too much of the ribbon in the superficial bins.

This renders the shells as they actually are, rather than as a cartoon: the same `equivol_t` the
measurement calls, evaluated on one specimen's own surfaces, built into wedge cells between
consecutive shells, and cut with THE SAME PLANE render_voxels.py samples the modalities on. So the
cut is a magnified piece of that plane, not a third location: the tissue whose shells are drawn
here is the tissue the figure already showed in the raw images and in the tractogram.

Which piece of it is chosen by geometry, not by eye. Among the cortical 3HG landmarks lying within
1 mm of the plane and comfortably inside the field of view, take the one where the plane runs most
nearly perpendicular to the local cortical sheet - that is the landmark whose ribbon is cut ACROSS
rather than obliquely, and an oblique cut smears the shells into a band with no visible order. The
cut then shows what the correction does: the shells crowd on the outside of a curve and spread on
the inside, which is the geometry an equidistant sampler gets wrong at a gyral crown."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
import dk
from depth_profile import equivol_t, vertex_area

FS = (_env.ROOT + "/surf/fs")
OUT = (_env.ROOT + "/surf/three_hinge/results")
H = "rh"; SID = int(os.environ.get("SUBJ", "6"))
NODE = int(os.environ.get("NODE", "-1"))          # index into the 3HG list; -1 = pick one
CROP = float(os.environ.get("CROP", "20.0"))      # mm half-width of the wedge grid built
ZOOM = float(os.environ.get("ZOOM", "0.62"))     # fraction of CROP the camera actually frames
NEAR_PLANE = 1.0                                  # mm; how close a candidate must sit to the plane
EDGE = 14.0                                       # mm of field of view to keep clear of
PLANE_N = None
NSH = 10                                          # shells: white, 9 equivolumetric, pial


def build(sid):
    """shells in SCANNER WORLD coordinates - the cut plane comes from the DWI affine, and mixing
    tkrRAS with world silently rotates the section"""
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    W, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    P, _ = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.pial")
    W = nib.affines.apply_affine(tk, W.astype(float))
    P = nib.affines.apply_affine(tk, P.astype(float))
    Aw, Ap = vertex_area(W, f), vertex_area(P, f)
    rho = np.linspace(0, 1, NSH + 1)
    # layer k occupies the volume between rho[k] and rho[k+1] - equal volume by construction,
    # which is the whole point of the correction
    S = np.stack([W + equivol_t(r, Aw, Ap)[:, None] * (P - W) for r in rho])
    return W, P, f, S, rho


def main():
    import pyvista as pv
    s = f"Subject{SID}"
    W, Pial, f, S, rho = build(SID)
    parc = dk.read(s)[0]; ctx = parc > 0
    hinge = np.loadtxt(f"{FS}/../three_hinge/gnet/{s}/out/{H}_3hinge_ids.txt", dtype=int)
    hinge = hinge[ctx[hinge]]

    m = pv.PolyData(W, np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = m.point_data["Normals"]
    if np.mean(np.sum((W - W.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm

    global PLANE_N
    zz = np.load(f"{OUT}/_voxels_S{SID}.npz", allow_pickle=True)
    P0, E1, E2 = zz["origin"], zz["e1"], zz["e2"]; PLANE_N = zz["nrm"]
    globals().update(P0=P0, E1=E1, E2=E2)
    # render_voxels.py already chose the landmark - it had to, in order to frame the slice around
    # it - so take that choice rather than repeating the search with slightly different tie-breaks
    if NODE >= 0:
        v0 = hinge[NODE]
    elif "node" in zz.files:
        v0 = int(hinge[np.argmin(np.linalg.norm(W[hinge] - zz["node"], axis=1))])
        print(f"  landmark from render_voxels: vertex {v0}, "
              f"{np.linalg.norm(W[v0] - zz['node']):.2f} mm from its world position", flush=True)
    else:
        v0 = pick_in_plane(hinge, W, nrm, P0, E1, E2, PLANE_N)
    origin = W[v0].copy()
    n = PLANE_N
    # Up is the SLICE's up (superior), not the cortical normal at the landmark. Rolling the camera
    # to the local normal made this a rotated view of the box drawn on panel a, which the reader
    # then has to mentally un-rotate; sharing the slice's own frame makes it a plain zoom.
    nz = E2 / np.linalg.norm(E2)

    keep = np.linalg.norm(W - W[v0], axis=1) < CROP
    fk = f[keep[f].all(1)]
    # wedge cells between consecutive shells: 6 points, tri on each shell
    nv = len(W); cells, ctype, band = [], [], []
    for k in range(NSH):
        for t in fk:
            cells.append([6, *(t + k * nv), *(t + (k + 1) * nv)])
            ctype.append(pv.CellType.WEDGE); band.append(k)
    grid = pv.UnstructuredGrid(np.concatenate(cells), np.array(ctype),
                               S.reshape(-1, 3))
    grid.cell_data["band"] = np.array(band)
    cut = grid.slice(normal=n, origin=origin)
    print(f"  {s}: vertex {v0}, {keep.sum()} vertices in crop, {cut.n_cells} cut cells", flush=True)

    pl = pv.Plotter(off_screen=True, window_size=(2200, 2200), border=False, lighting=None)
    pl.background_color = "white"; pl.enable_anti_aliasing("ssaa")
    pl.add_mesh(cut, scalars="band", cmap=LAYER_CMAP(), show_scalar_bar=False,
                lighting=False, clim=(-0.5, NSH - 0.5))
    # the white surface itself, so the reader can see where rho = 0 is
    wm = pv.PolyData(W, np.c_[np.full(len(f), 3), f].ravel()).slice(normal=n, origin=origin)
    pl.add_mesh(wm, color=ns_ink(), line_width=2.0, lighting=False)
    # the landmark itself, so the cut is anchored to a 3HG rather than to an anonymous gyrus
    import natstyle as ns
    pl.add_mesh(pv.Sphere(radius=0.62, center=W[v0]), color=ns.C_3HG, lighting=False)
    pl.camera.SetParallelProjection(True)
    pl.camera.position = origin + n * 200
    pl.camera.focal_point = origin
    pl.camera.up = nz                                    # normal points up: pial at the top
    pl.camera.parallel_scale = float(zz["box"][2])       # exactly the detail box of panel a
    # where this cut sits inside the sampled plane, so panels a and c can draw the zoom box.
    # The camera is rolled (up = the in-plane part of the cortical normal), so the framed square is
    # rotated in the (E1, E2) basis and an axis-aligned box would be wrong.
    half = float(zz["box"][2])
    up2 = np.array([0.0, 1.0])                       # camera up == E2, so the box is axis-aligned
    ctr2 = np.array([(W[v0] - P0) @ E1, (W[v0] - P0) @ E2])
    # the white and pial boundaries as polylines in the slice's own (E1, E2) coordinates, so the
    # detail views can draw the cortical ribbon over the raw image
    def contour(mesh):
        c = mesh.slice(normal=n, origin=origin)
        pts = np.asarray(c.points); L = np.asarray(c.lines).reshape(-1, 3)[:, 1:]
        uv = np.c_[(pts - P0) @ E1, (pts - P0) @ E2]
        segs = uv[L]
        keep = np.all(np.abs(segs - ctr2) <= half * 1.15, (1, 2))
        return segs[keep]
    faces = np.c_[np.full(len(f), 3), f].ravel()
    cw = contour(pv.PolyData(W, faces))
    cp = contour(pv.PolyData(Pial, faces))
    print(f"  ribbon contours in the box: {len(cw)} white, {len(cp)} pial segments", flush=True)
    np.savez(f"{OUT}/_lamina_S{SID}_box.npz", ctr=ctr2, up=up2, half=half, cw=cw, cp=cp)
    pl.screenshot(f"{OUT}/_lamina_S{SID}.png", transparent_background=True)
    print(f"  wrote {OUT}/_lamina_S{SID}.png", flush=True)


def ns_ink():
    import natstyle as ns
    return ns.INK


def LAYER_CMAP():
    from matplotlib.colors import LinearSegmentedColormap
    import natstyle as ns
    return LinearSegmentedColormap.from_list("lam", [ns.C_CROWN, "#8CA3C2", "#C9D5E3",
                                                     "#EFCDBF", "#E5A084", ns.C_3HG], N=NSH)


def pick_in_plane(hinge, W, nrm, P0, E1, E2, N):
    """the landmark in the plane whose ribbon that plane cuts most nearly across

    |N . n_cortex| is the cosine between the cut plane's normal and the cortical normal. Near 0 the
    plane contains the cortical normal and the ribbon appears in true cross-section; near 1 the
    plane is tangential and the ribbon smears into a shapeless band."""
    d = (W[hinge] - P0) @ N
    u = (W[hinge] - P0) @ E1
    v = (W[hinge] - P0) @ E2
    ok = (np.abs(d) < NEAR_PLANE) & (np.abs(u) < 39.0 - EDGE) & (np.abs(v) < 39.0 - EDGE)
    if not ok.any():
        raise SystemExit("no 3HG landmark near the plane - widen NEAR_PLANE")
    cand = hinge[ok]
    obl = np.abs(nrm[cand] @ N) / np.linalg.norm(nrm[cand], axis=1)
    best = int(cand[int(np.argmin(obl))])
    print(f"  {int(ok.sum())} landmarks in plane; picked {best}, |N.n| = {obl.min():.3f}", flush=True)
    return best


if __name__ == "__main__":
    main()

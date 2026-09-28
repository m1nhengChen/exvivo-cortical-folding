#!/usr/bin/env python3
"""Close-up of one gyral junction, showing what the estimator actually samples and what the
control has to satisfy. Feeds the matching row of Fig. 1.

Everything drawn here is the real thing, at its real size:

  skeleton      the GyralNet gyral crest network (rh_surf_skelenton_allpoints_final_refine.vtk),
                on the white surface's own vertex indexing. A 3HG is where three of its branches
                meet - the panel exists so the reader can see that, rather than take it on trust.
  disks         every landmark is drawn as the 2 mm geodesic disk that disk_means() averages over,
                not as a marker of arbitrary radius.
  merge ring    landmarks whose sub-white-matter patch centres lie within SEP mm are one junction
                and are averaged into one value (hinge_ids.cluster_groups).
  exclusion     a crown landmark must sit more than CROWN_FAR mm geodesic from every 3HG, so that
                its 2 mm disk cannot contain hinge tissue.

    python3 render_match_zoom.py           # Subject4, writes results/_classes/S4_zoom.png
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
from scipy.sparse.csgraph import dijkstra
import dk
import natstyle as ns
from hinge_profile import mesh_graph, CROWN_FAR, DISK_MM, SEP
from hinge_ids import cluster_groups
import crest
from crest import skeleton_edges, tube, C_SKEL

FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
OUT = (_env.ROOT + "/surf/three_hinge/results/_classes")
H = "rh"; SID = int(os.environ.get("SUBJ", "4"))
# binary curvature shading, the convention for a surface close-up: at this scale a uniformly
# white cortex leaves the gyral/sulcal pattern to specular highlights alone and reads as drapery
GYRAL, SULCAL = (0.90, 0.895, 0.885), (0.745, 0.740, 0.730)
C_HINGE, C_CROWN = "#B85A38", "#33507A"      # the render hues of fig_main1_combined's panel
C_ARM = "#E2A183"                    # the junction's own arms: a tint of the 3HG colour, so
                                      # they read as context and the measured disks stay the
                                      # only full-strength 3HG mark
C_ELIG = "#AFC4DC"                    # cortex a crown landmark may be drawn from
BRANCH_MM = 6.0                       # how far along the crest network the junction's
                                      # own arms are highlighted
FOV = float(os.environ.get("FOV", "38"))     # mm across the rendered patch


def pick_junction(v, parc, hinge, grp, dmin_all):
    """A junction that is worth drawing: several landmarks merged into one, on the lateral
    convexity so the camera is not looking into a sulcus, and with crown-eligible cortex just
    outside the exclusion radius so the ring has something to exclude."""
    lateral = v[:, 0] > np.percentile(v[:, 0], 55) if v[:, 0].mean() > 0 else v[:, 0] < np.percentile(v[:, 0], 45)
    best = None
    for g in grp:
        ids = hinge[g]
        if not (2 <= len(ids) <= 4) or not lateral[ids].all():
            continue
        span = np.linalg.norm(v[ids] - v[ids].mean(0), axis=1).max()
        if span < 1.2:                       # all on top of each other: the merge would not show
            continue
        if best is None or abs(span - 3.0) < abs(best[0] - 3.0):
            best = (span, ids)
    return best[1] if best else hinge[grp[int(np.argmax([len(g) for g in grp]))]]


def main():
    import pyvista as pv
    os.makedirs(OUT, exist_ok=True)
    s = f"Subject{SID}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    vw = nib.affines.apply_affine(tk, v.astype(float))
    parc = dk.read(s)[0]; ctx = parc > 0
    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
    hinge = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int)
    hinge = hinge[ctx[hinge]]
    G = mesh_graph(vw, f)
    dmin = dijkstra(G, indices=hinge.tolist(), min_only=True)
    grp = cluster_groups(SID, hinge, merge=SEP)
    ids = pick_junction(vw, parc, hinge, grp, dmin)
    ctr = vw[ids].mean(0)
    print(f"  junction of {len(ids)} landmarks, span "
          f"{np.linalg.norm(vw[ids] - ctr, axis=1).max():.1f} mm", flush=True)

    dj = dijkstra(G, indices=ids.tolist(), min_only=True)          # from this junction
    lo = np.percentile(sulc[ctx], 25)
    # No crown disks. The zoom is drawn as a magnification of the boxed region of the 3HG
    # surface in the same panel, so it may show only what that surface shows: the crest network
    # and the 3HG disks. Crown landmarks live on their own surface beside it.
    crown = np.array([], int)
    edges = skeleton_edges(SID)
    rgb = np.where((sulc <= 0)[:, None], np.array(GYRAL, np.float32),
                   np.array(SULCAL, np.float32)).astype(np.float32)
    for seeds, col in ((crown, C_CROWN), (ids, C_HINGE)):
        if len(seeds):
            d = dijkstra(G, indices=np.asarray(seeds).tolist(), min_only=True,
                         limit=DISK_MM * 1.001)
            rgb[d <= DISK_MM] = pv.Color(col).float_rgb

    mesh = pv.PolyData(vw, np.c_[np.full(len(f), 3), f].ravel())
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    N = mesh.point_data["Normals"].copy()
    if np.mean(np.sum((vw - vw[ctx].mean(0)) * N, 1)) < 0:
        N = -N
    # the branches that meet at this junction, in the 3HG colour: the panel's whole point is that
    # a 3HG is a degree-3 node of this network, and that only reads if the arms are marked
    near = (dj[edges] < BRANCH_MM).all(1)
    nrm = N[ids].mean(0); nrm /= np.linalg.norm(nrm)
    if np.dot(ctr - vw[ctx].mean(0), nrm) < 0:
        nrm = -nrm
    mesh.point_data["rgb"] = (rgb * 255).astype(np.uint8)

    up = np.array([0.0, 0.0, 1.0]); up -= nrm * (up @ nrm); up /= np.linalg.norm(up)

    def shoot(name, rgb_, extra):
        pl = pv.Plotter(off_screen=True, window_size=(1500, 1150), border=False,
                        lighting="light kit")
        pl.background_color = "white"; pl.enable_anti_aliasing("ssaa")
        mesh.point_data["rgb"] = (rgb_ * 255).astype(np.uint8)
        pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True,
                    specular=0.04, specular_power=24, interpolate_before_map=False)
        pl.add_mesh(tube(vw, N, edges[~near] if extra else edges, 0.20), color=C_SKEL,
                    smooth_shading=True, specular=0.05)
        if extra:
            pl.add_mesh(tube(vw, N, edges[near], 0.22), color=C_ARM, smooth_shading=True,
                        specular=0.05)
        pl.camera.position = tuple(ctr + nrm * 200); pl.camera.focal_point = tuple(ctr)
        pl.camera.up = tuple(up)
        pl.camera.enable_parallel_projection()
        pl.camera.parallel_scale = FOV / 2 * (1150 / 1500)     # FOV is the horizontal span
        pl.screenshot(f"{OUT}/S{SID}_{name}.png", transparent_background=True)
        print(f"  wrote {OUT}/S{SID}_{name}.png  ({FOV:.0f} mm across)", flush=True)

    # 1. the 3HG surface's patch: junction disks, arms marked
    shoot("zoom", rgb, extra=True)

    # 2. the Crown surface's patch, from the SAME camera: the display draw that render_classes.py
    #    paints on the whole Crown surface, so this too is a true magnification of that surface
    from render_classes import sets_of, disk_mask
    G2, sets = sets_of(SID)
    rgb_c = np.where((sulc <= 0)[:, None], np.array(GYRAL, np.float32),
                     np.array(SULCAL, np.float32)).astype(np.float32)
    rgb_c[disk_mask(G2, sets["crown"])] = pv.Color(C_CROWN).float_rgb
    inview = sets["crown"][dj[sets["crown"]] < FOV * 0.55]
    print(f"  {len(inview)} crown landmarks of the display draw inside the patch", flush=True)
    shoot("zoom_crown", rgb_c, extra=False)

    np.savez(f"{OUT}/S{SID}_zoom_meta.npz", n_land=len(ids), n_crown=len(inview),
             sep=SEP, far=CROWN_FAR, disk=DISK_MM, fov=FOV, ctr=ctr)


if __name__ == "__main__":
    main()

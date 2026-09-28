#!/usr/bin/env python3
"""Where the three sampling classes sit on the cortex: 3HG, gyral crown, sulcal fundus.

    python3 render_classes.py            # Subject4, white surface, lateral

The three pools are built by exactly the code that measures them (hinge_profile.py): cortical 3HG
landmarks; the most-gyral quartile more than 8 mm geodesic from any 3HG; and the most-sulcal
quartile.

THE DRAW HERE IS NOT THE DRAW THAT WAS MEASURED. hinge_profile.py subsamples the crown and fundus
pools uniformly at random, which makes them regionally uneven: the global `sulc` quartile threshold
starves parcels that are systematically deep (pericalcarine 0.11x expected density, superiortemporal
0.15x, cuneus 0x) and floods ones that are systematically shallow (parahippocampal 5.4x,
parstriangularis 4.2x), and the 8 mm exclusion radius empties crown territory exactly where 3HG are
dense. Across the six specimens the per-parcel density CV is 0.96 for a uniform crown draw against
0.37 for 3HG, correlated -0.23 with the 3HG distribution.

The larger visual effect, though, is that the measured crown draw cannot be spread out at all. The
8 mm exclusion radius leaves only 1.9% of the cortical surface eligible, in thin ribbons along the
crests, and a count-matched draw packs into it at about 2 mm even when placed optimally - the draw
actually used has a nearest-neighbour median of 0.78 mm and a minimum of 0.24 mm, against 4.48 mm
median for the 3HG set. Their 2 mm-radius measuring disks then overlap almost completely and the
panel reads as blotchy rather than as a statement about where the classes sit.

The render therefore relaxes the exclusion radius (RENDER_FAR, below) and draws by max-min spacing
rather than uniformly at random. THIS IS A VISUALIZATION CHOICE ONLY - no analysis uses this draw. The regional unevenness and the tightness of the measured control are real
properties of experiment 1 that experiment 3 (`region_dk.py`, within-parcel) and the depth
analysis's `dWP` exist to address. Do not describe this panel as showing the analysed sample.

Each landmark is drawn as its actual measuring support - the 2 mm geodesic disk that `disk_means`
averages over - rather than as a marker of arbitrary size. A sphere glyph invites the reader to
compare its radius with the folding, and the radius meant nothing; the geodesic disk is the
estimator. It also makes the sampling density honest: where disks touch, the samples overlap, which
is precisely what the junction merge (4 mm on the sub-WM patch centres) exists to correct.

Subject4 by default: the smoothest of the six by curvature (mean |curv| 0.194 against 0.249 for
Subject5), so the folding reads cleanly at panel size."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
from scipy.sparse.csgraph import dijkstra
import dk, camera
import natstyle as ns
from hinge_profile import mesh_graph, CROWN_FAR, DISK_MM
import crest

# Exclusion radius for the RENDER only. hinge_profile.py measures with CROWN_FAR = 8 mm, which
# keeps a crown landmark's 2 mm disk clear of any hinge tissue - a measurement-independence device,
# not part of what "gyral crown" means. It also removes 92% of the gyral quartile (24.7% of cortex
# down to 1.9%), which is why a count-matched draw cannot be spread out. Relaxing it for display
# keeps the class definition - most-gyral quartile - and keeps the fold depth right: median sulc
# percentile 14.4 at 3 mm against 15.9 at 8 mm, both against 8.7 for the 3HG set. Raising the sulc
# threshold instead would not: sulc <= 50% puts the pool at percentile 35.2, on the gyral banks,
# which turns the contrast into gyrus-versus-sulcus.
# DISPLAY exclusion radius, an author decision (2026-09-25). The measured draw uses CROWN_FAR =
# 8 mm. At 8 mm only ~170 crown landmarks fit the eligible crests at the landmarks' own spacing
# and the whole-brain Crown surface read as sparse; at 6 mm 352 fit. The cost is stated in the
# legend and must stay stated: Fig. 2a magnifies this draw to millimetre scale beside a key that
# says ">= 8 mm from a 3HG", so crown disks at 6-8 mm from a junction are visible there. The
# analysis is untouched. Setting RENDER_FAR=8 restores a display that obeys the rule exactly.
RENDER_FAR = float(os.environ.get("RENDER_FAR", "6.0"))

FS = (_env.ROOT + "/surf/fs")
OUT = (_env.ROOT + "/surf/three_hinge/results")
CACHE = f"{OUT}/_classes"
H = "rh"; SID = int(os.environ.get("SUBJ", "4"))
GROUND = (1.00, 1.00, 1.00)          # white cortex, so the only colour on the panel is the class
ZOOM = float(os.environ.get("ZOOM", "1.45"))
SKEL_R = 0.13                        # tube radius, mm; thinner than the zoom's 0.20 so
                                     # at 32 mm panel width the net reads as a line, not a mesh
# Render colours, not palette colours. A lit 3D surface adds ambient and specular white to every
# diffuse colour, so the palette hue that reads correctly as a flat swatch in panel b comes back
# washed out here - ns.C_FUND especially, which is already a light blue and disappeared into the
# white cortex. These are the same three hues carried two steps darker, so the panel-b ordering
# (3HG orange, crown blue, fundus lighter blue) survives while the patches stay legible.
PAINT = {"3HG": "#B85A38", "crown": "#33507A", "fundus": "#6E88AC"}
CLASSES = [("3HG", ns.C_3HG), ("crown", ns.C_CROWN), ("fundus", ns.C_FUND)]


def sets_of(sid):
    s = f"Subject{sid}"
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white"); v = v.astype(float)
    parc = dk.read(s)[0]; ctx = parc > 0
    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
    hinge = np.loadtxt(f"{FS}/../three_hinge/gnet/{s}/out/{H}_3hinge_ids.txt", dtype=int)
    hinge = hinge[ctx[hinge]]
    G = mesh_graph(v, f)
    dmin = dijkstra(G, indices=hinge.tolist(), min_only=True)
    lo, hi = np.percentile(sulc[ctx], 25), np.percentile(sulc[ctx], 75)
    cp = np.where((sulc <= lo) & (dmin > RENDER_FAR) & ctx)[0]
    fp = np.where((sulc >= hi) & ctx)[0]
    rng = np.random.RandomState(sid)
    crown, sep = spread(cp, hinge, v, rng)
    fund, _ = spread(fp, hinge, v, rng, sep)
    return G, {"3HG": hinge, "crown": crown, "fundus": fund}


def spread(pool, hinge, xyz, rng, min_sep=None):
    """max-min draw from pool, stopped when the next point would come closer than min_sep

    Count-matching the control to the 3HG set cannot be spread evenly, because the crown pool is
    not big enough to hold that many well-separated points: the gyral quartile is 24.7% of the
    cortex, but excluding 8 mm around every 3HG cuts it to 2.6%, and at the 3HG set's own median
    nearest-neighbour spacing only about half the required number of crown points fit. Forcing the
    full count packs them at ~2 mm - their 2 mm-radius measuring disks then overlap almost
    completely, which is what made the panel look blotchy.

    So the render matches the 3HG set's SPACING rather than its COUNT, and the caption has to say
    the counts differ. Illustration only - see the module docstring."""
    P = xyz[pool]
    k = len(hinge)
    if min_sep is None:
        D = xyz[hinge]
        nn = np.array([np.partition(np.linalg.norm(D - d, axis=1), 1)[1] for d in D])
        min_sep = float(np.median(nn))
    out = [rng.randint(len(P))]
    d = np.linalg.norm(P - P[out[0]], axis=1)
    while len(out) < k:
        i = int(np.argmax(d))
        if d[i] < min_sep:
            break
        out.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.sort(pool[np.array(out)]), min_sep


def disk_mask(G, seeds, radius=DISK_MM):
    """vertices inside the union of the per-landmark geodesic disks the estimator averages over"""
    d = dijkstra(G, indices=np.asarray(seeds).tolist(), min_only=True, limit=radius * 1.001)
    return d <= radius


def _callout(pl, vw):
    """Record where the zoom patch lands on this render, in image pixels.

    render_match_zoom.py picks the junction and saves its world centre; projecting that centre
    through this plotter's camera is what lets Fig. 2a draw the zoom as a magnification of the
    brain below it instead of leaving the reader to guess where it came from."""
    import numpy as _np, os as _os
    from vtkmodules.vtkRenderingCore import vtkCoordinate
    f = f"{CACHE}/S{SID}_zoom_meta.npz"
    if not _os.path.exists(f):
        return
    m = _np.load(f)
    ctr, fov = m["ctr"], float(m["fov"])
    def to_px(pt):
        c = vtkCoordinate(); c.SetCoordinateSystemToWorld(); c.SetValue(*[float(x) for x in pt])
        return _np.array(c.GetComputedDisplayValue(pl.renderer), float)
    # size the box from the rendered patch itself: project two points fov/2 apart on screen
    cam = pl.camera
    right = _np.cross(_np.array(cam.focal_point) - _np.array(cam.position), cam.up)
    right /= _np.linalg.norm(right)
    p0, p1 = to_px(ctr), to_px(ctr + right * fov / 2)
    half = abs(p1[0] - p0[0])
    h = pl.window_size[1]
    _np.savez(f"{CACHE}/S{SID}_callout.npz", x=p0[0], y=h - p0[1],   # to image row order
              half=half, win=_np.array(pl.window_size))
    print(f"  callout at ({p0[0]:.0f}, {h - p0[1]:.0f}) px, half-width {half:.0f} px", flush=True)


def main():
    import pyvista as pv
    os.makedirs(CACHE, exist_ok=True)
    s = f"Subject{SID}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    vw = nib.affines.apply_affine(tk, v)
    mesh = pv.PolyData(vw, np.c_[np.full(len(f), 3), f].ravel())
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    parc = dk.read(s)[0]
    view, up = camera.frame(vw, parc)
    N = mesh.point_data["Normals"].copy()
    if np.mean(np.sum((vw - vw[parc > 0].mean(0)) * N, 1)) < 0:
        N = -N
    edges = crest.skeleton_edges(SID)
    G, sets = sets_of(SID)
    for name, col in CLASSES:
        m = disk_mask(G, sets[name])
        rgb = np.tile(np.array(GROUND, np.float32), (len(vw), 1))
        rgb[m] = pv.Color(PAINT[name]).float_rgb
        mesh.point_data["rgb"] = (rgb * 255).astype(np.uint8)
        pl = pv.Plotter(off_screen=True, window_size=(1500, 1150), border=False,
                        lighting="light kit")
        pl.background_color = "white"; pl.enable_anti_aliasing("ssaa")
        # interpolate_before_map off: the disk edge stays a disk edge instead of being smeared
        # across the triangles that straddle it
        pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True,
                    specular=0.10, specular_power=18, interpolate_before_map=False)
        # The crest network on every surface. 3HG are defined as its degree-3 nodes and the
        # crown by its distance from them, so it is shared context for all three classes - and
        # the zoom above is a magnification of the 3HG surface, which it must therefore carry.
        pl.add_mesh(crest.tube(vw, N, edges, SKEL_R), color=crest.C_SKEL, smooth_shading=True,
                    specular=0.05)
        camera.place(pl, mesh, view, up, zoom=ZOOM)
        if name == "3HG":
            _callout(pl, vw)
        pl.screenshot(f"{CACHE}/S{SID}_{name}.png", transparent_background=True)
        print(f"  {name}: {len(sets[name])} landmarks, {int(m.sum())} vertices "
              f"({100 * m.sum() / (parc > 0).sum():.1f}% of cortex)", flush=True)


if __name__ == "__main__":
    main()

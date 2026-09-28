#!/usr/bin/env python3
"""Export the hinge-network scene (same cache as render_hinge_net*.py) as PLY meshes plus
camera definitions, for the Blender/Cycles render (blender_net.py)."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, json, numpy as np, nibabel as nib
import pyvista as pv
import camera
FS = (_env.ROOT + "/surf/fs")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
OUT = f"{TR}/blender_scene"
import os; os.makedirs(OUT, exist_ok=True)

z = np.load(f"{TR}/_net_cache_S5.npz")
pts, lens, vw, f = z["pts"], z["lens"], z["vw"], z["faces"]
lines, off = [], 0
for k in lens:
    lines += [int(k)] + list(range(off, off + int(k))); off += int(k)
tract = pv.PolyData(); tract.points = pts; tract.lines = np.array(lines)

# ---- anatomical frame, needed before the tube colours ---------------------------------------
lat, up = camera.frame(vw, z["parc"])
ant = camera._com(vw, z["parc"], camera.ANT) - camera._com(vw, z["parc"], camera.POS)
e_si = up / np.linalg.norm(up)
e_ap = ant - e_si * (ant @ e_si); e_ap /= np.linalg.norm(e_ap)
e_lr = np.cross(e_ap, e_si); e_lr /= np.linalg.norm(e_lr)

# Directionally-encoded colour. The tangent is projected onto the ANATOMICAL axes, not onto the
# voxel axes: these are ex-vivo specimens in arbitrary scanner orientation, so |tangent| in image
# coordinates would paint superior-inferior fibres green and quietly break the convention every
# reader applies (red left-right, green anterior-posterior, blue superior-inferior).
# DEC_SAT pulls the colour toward its own luminance; it is 1.0 (no desaturation) because a
# tinted version read grey and pink and lost the green/cyan distinction.
#
# DEC_MODE picks what "direction" means. 'point' is the per-point local tangent, the convention
# tractography software draws; it makes every tube change hue along its length, so a bundle of a
# dozen fibres reads as flickering noise rather than as one rope. 'line' takes each streamline's
# own end-to-end direction, giving it a single colour - the bundle then reads as a coloured rope
# and the colour states where that connection runs, which is what this panel is about.
SAT = float(os.environ.get("DEC_SAT", 1.00))
DEC_MODE = os.environ.get("DEC_MODE", "point")
off_, tan = 0, np.empty_like(pts, dtype=np.float32)
for k in lens:
    k = int(k)
    seg = pts[off_:off_ + k]
    if DEC_MODE == "line":
        v = seg[-1] - seg[0]
        tan[off_:off_ + k] = v / (np.linalg.norm(v) + 1e-9)
    else:
        d = np.gradient(seg, axis=0)
        tan[off_:off_ + k] = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    off_ += k
c = np.abs(np.stack([tan @ e_lr, tan @ e_ap, tan @ e_si], 1)).astype(np.float32)
c /= np.maximum(c.max(1, keepdims=True), 1e-9)          # full-value DEC, as the convention draws it
lum = (c * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(1, keepdims=True)
c = np.clip(lum + (c - lum) * SAT, 0, 1)
tract.point_data["rgb"] = (c * 255).round().astype(np.uint8)
tubes = tract.tube(radius=0.125, n_sides=16, capping=True).triangulate()
tubes.save(f"{OUT}/tubes.ply", texture="rgb")

white = pv.PolyData(vw, np.c_[np.full(len(f), 3), f].ravel()).triangulate()
white.save(f"{OUT}/white.ply")
# the clay panel can use either surface as its backdrop; smoothed to match pial.ply's treatment
white_s = white.smooth_taubin(n_iter=60, pass_band=0.02).triangulate()
white_s.save(f"{OUT}/white_s.ply")
mgh = nib.load(f"{FS}/Subject5/mri/wm.mgz")
tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
vp, fp = nib.freesurfer.read_geometry(f"{FS}/Subject5/surf/rh.pial")
pial = pv.PolyData(nib.affines.apply_affine(tk, vp.astype(float)),
                   np.c_[np.full(len(fp), 3), fp].ravel())
pial = pial.smooth_taubin(n_iter=60, pass_band=0.02).triangulate()
pial.save(f"{OUT}/pial.ply")

bh = pv.Sphere(radius=1.25, theta_resolution=32, phi_resolution=32)
bc = pv.Sphere(radius=0.52, theta_resolution=20, phi_resolution=20)
pv.PolyData(z["nodes_h"]).glyph(geom=bh, scale=False, orient=False).triangulate().save(f"{OUT}/hinge.ply")
pv.PolyData(z["nodes_c"]).glyph(geom=bc, scale=False, orient=False).triangulate().save(f"{OUT}/crown.ply")

ctr = np.array(white.center)
ant = e_ap                                   # the orthonormal frame built for the DEC colours
# the white hull does not contain the tract ends - tracking runs on into the cortical ribbon -
# so frame on the union, or the frame edge crops the very fibres the panel is about
HULL = {"pial": np.asarray(pial.points),
        "white": np.vstack([np.asarray(white_s.points), pts])}
obl = 0.92 * lat + 0.36 * ant + 0.28 * up   # gentle 3/4: keeps the lateral profile readable
obl /= np.linalg.norm(obl)
cams = {}
for tag, view, upv, res, persp in (("lat", lat, up, (3400, 2700), False),
                                   ("dor", up, ant, (900, 1700), False),
                                   ("obl", obl, up, (2600, 1700), True),
                                   ("clay", lat, up, None, False),
                                   ("clay_w", lat, up, None, False)):
    # frame on whichever surface this camera actually shows, else the white brain sits small in
    # a frame solved for the larger pial hull and every edge reads as overspill
    V = HULL["white" if tag == "clay_w" else "pial"] - ctr
    fwd = -view
    right = np.cross(fwd, upv); right /= np.linalg.norm(right)
    upo = np.cross(right, fwd)
    ex = np.abs(V @ right).max(); ey = np.abs(V @ upo).max()
    ez = np.abs(V @ view).max()
    if res is None:
        # frame the hull itself instead of padding a fixed canvas: the clay view is the only
        # thing in panel f, so every empty pixel is panel width thrown away
        res = (3600, int(round(3600 * ey / ex / 16) * 16))
    aspect = res[1] / res[0]
    scale = 2 * max(ex, ey / aspect) * 1.045
    # perspective: 80 mm lens on a 36 mm sensor. Solve the camera distance and the in-plane
    # offset by actually projecting the hull - a bbox estimate ignores that near geometry
    # projects larger, and the render then ran off the top and right edges.
    tanh = 18.0 / 80.0
    dist, shift = 500.0, np.zeros(3)
    if persp:
        P = HULL["white" if tag == "clay_w" else "pial"]
        dist = max(ex, ey / aspect) / tanh * 1.25 + ez
        for _ in range(80):
            cam_pos = ctr + view * dist + shift
            d = P - cam_pos
            zz = np.maximum(-(d @ view), 1e-3)     # depth in front of the camera
            sx = (d @ right) / zz / tanh
            sy = (d @ upo) / zz / (tanh * aspect)
            cx, cy = (sx.max() + sx.min()) / 2, (sy.max() + sy.min()) / 2
            need = max(sx.max() - sx.min(), sy.max() - sy.min()) / 2 / 0.94
            shift = shift + (right * cx + upo * cy) * dist * tanh
            dist *= need
            if abs(need - 1) < 2e-3 and abs(cx) < 2e-3 and abs(cy) < 2e-3:
                break
    cams[tag] = dict(view=list(map(float, view)), up=list(map(float, upo)),
                     right=list(map(float, right)), center=list(map(float, ctr + shift)),
                     ortho_scale=float(scale), res=list(res),
                     persp=bool(persp), dist=float(dist), halfdepth=float(ez))
json.dump(cams, open(f"{OUT}/cams.json", "w"), indent=1)
print("exported", {k: v.n_points for k, v in
      (("tubes", tubes), ("white", white), ("pial", pial))}, "cams ok")

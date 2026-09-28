#!/usr/bin/env python3
"""Whole-brain schematic of the hinge network: translucent cortex, 3HG nodes, and the actual
streamlines that link one 3HG to another.

The streamlines drawn here are exactly the ones the connectivity matrix counts: a streamline is
kept if it passes within R mm of the sub-WM patch of one 3HG node AND its distal endpoint falls
within NEAR mm of the patch of a different 3HG node. Nothing is idealized or redrawn - the panel
is the anatomy behind the 3HG-3HG block of Fig. 4a."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, nibabel as nib
from collections import defaultdict
from scipy import sparse
from scipy.spatial import cKDTree
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
import dk, camera
import natstyle as ns

FS = (_env.ROOT + "/surf/fs")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
OUT = (_env.ROOT + "/surf/three_hinge/results")
H = "rh"; SID = 5; R = 3.0; INWARD = 2.0; NEAR = 4.0
# What gets drawn. Thinning at the STREAMLINE level (the old KMAX=1 over all 576 pairs) put one
# lone fibre on each connection, so the panel showed 460 unrelated hairs crossing in the centrum
# semiovale - a tangle, not anatomy. Thinning at the PAIR level instead keeps every streamline a
# drawn connection owns, so each connection reads as the small bundle it actually is.
# Even sampling still cannot bundle: the median connection carries only 2 streamlines, so most
# drawn pairs would contribute a lone hair whatever the cap. Taking the pairs by streamline count
# instead puts a floor under every bundle - at NPAIR = 120 the thinnest drawn connection still
# has 7 fibres - and what the panel then shows is the strongest 3HG-3HG connections, which the
# legend states. The block statistics in b-e are untouched by this; they use every pair.
NPAIR = 120                                # connected 3HG-3HG pairs drawn, the most streamlined
KMAX = 12                                  # cap on streamlines per drawn pair
# streamlines are coloured by local direction (|unit tangent| -> RGB), the convention for
# tractography; the node classes keep the paper's palette, orange 3HG / blue crown
CORTEX = "#dedbd6"; NODE_H, NODE_C = ns.C_3HG, ns.C_CROWN
CACHE = f"{TR}/_net_cache_S{SID}.npz"
SEL = f"{TR}/_net_sel_S{SID}.npz"          # full candidate set; thin() re-derives CACHE from it


def select():
    """Expensive pass: index the tractogram and keep the 3HG-to-3HG streamlines. Cached."""
    import pyvista as pv
    s = f"Subject{SID}"
    z = np.load(f"{TR}/{s}/conn_matrix.npz")
    kind, vert = z["kind"], z["vert"]
    hmask = kind == 0

    dwi = nib.load(f"{TR}/{s}/dwi.nii.gz")
    trk = load_tractogram(f"{TR}/{s}/whole_brain.trk", dwi, bbox_valid_check=False)
    trk.to_space(Space.RASMM); SL = trk.streamlines
    ep0 = np.array([sl[0] for sl in SL]); ep1 = np.array([sl[-1] for sl in SL])

    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    m = pv.PolyData(v.astype(float), np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = (tk[:3, :3] @ m.point_data["Normals"].T).T
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    vw = nib.affines.apply_affine(tk, v)
    if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    P = (vw - INWARD * nrm)[vert]                       # node patch centres, matrix order

    # ---- incidence, exactly as conn_matrix.py builds it -------------------------------------
    tree = cKDTree(P); n = len(P)
    pts = np.concatenate([np.asarray(sl) for sl in SL])
    owner = np.repeat(np.arange(len(SL)), [len(sl) for sl in SL])
    hits = tree.query_ball_point(pts, R)
    rows = np.concatenate([np.full(len(h), o) for o, h in zip(owner, hits) if h])
    cols = np.concatenate([h for h in hits if h])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows.astype(int), cols.astype(int))),
                          shape=(len(SL), n))
    A.sum_duplicates(); A.data[:] = 1.0; A = A.tocsr()
    e0 = tree.query_ball_point(ep0, NEAR); e1 = tree.query_ball_point(ep1, NEAR)

    # ---- keep the streamlines that link one 3HG to a different 3HG ---------------------------
    rng = np.random.RandomState(0)
    per_pair = defaultdict(list)
    for sl_id in np.unique(rows).astype(int):
        seeds = A.indices[A.indptr[sl_id]:A.indptr[sl_id + 1]]
        seeds = seeds[hmask[seeds]]
        if not len(seeds):
            continue
        d0 = np.linalg.norm(P[seeds] - ep0[sl_id], axis=1)
        d1 = np.linalg.norm(P[seeds] - ep1[sl_id], axis=1)
        for tg, sel in ((e0[sl_id], d0 > d1), (e1[sl_id], d1 >= d0)):
            tg = [t for t in tg if hmask[t]]
            if not len(tg) or not sel.any():
                continue
            for a in seeds[sel]:
                for b in tg:
                    if a != b:
                        per_pair[(min(a, b), max(a, b))].append(sl_id)
    # save every candidate, with its pair membership as a flat edge list: the tractogram pass
    # above costs minutes, and how much of it to draw is a question worth re-asking cheaply.
    keys = sorted(per_pair)
    ids = np.unique(np.concatenate([np.unique(per_pair[k]) for k in keys]))
    pos_of = {sl: i for i, sl in enumerate(ids)}
    edges = np.array([[pi, pos_of[sl]] for pi, k in enumerate(keys)
                      for sl in np.unique(per_pair[k])], dtype=np.int32)
    cnt = np.bincount(edges[:, 0])
    print(f"{len(keys)} connected 3HG-3HG pairs, {len(ids)} candidate streamlines; "
          f"per pair: median {np.median(cnt):.0f}, mean {cnt.mean():.1f}, max {cnt.max()}, "
          f"{(cnt >= 5).mean() * 100:.0f}% have >= 5", flush=True)

    streams = [np.asarray(SL[i]) for i in ids]
    # pair_nodes/hidx let thin() name the 3HG nodes a drawn connection actually lands on, so the
    # panel need not scatter spheres that no drawn fibre reaches
    np.savez(SEL, pts=np.vstack(streams), lens=np.array([len(sl) for sl in streams]),
             edges=edges, npair=len(keys), pair_nodes=np.array(keys, dtype=np.int32),
             hidx=np.where(hmask)[0].astype(np.int32),
             nodes_h=P[hmask], nodes_c=P[~hmask], vw=vw, faces=f, parc=dk.read(s)[0])
    return SEL


def thin(npair=NPAIR, kmax=KMAX, seed=0, mode="top"):
    """Pick pairs, then keep up to kmax of each drawn pair's own streamlines.

    mode 'top' takes the pairs carrying the most streamlines, so every drawn connection is a
    bundle; 'even' samples the pair list uniformly, which covers the cortex more evenly but
    leaves most connections as single fibres."""
    z = np.load(SEL)
    lens, edges = z["lens"], z["edges"]
    off = np.r_[0, np.cumsum(lens)]
    rng = np.random.RandomState(seed)
    npair_total = int(z["npair"])
    cnt = np.bincount(edges[:, 0], minlength=npair_total)
    if npair_total <= npair:
        take = np.arange(npair_total)
    elif mode == "top":
        take = np.sort(np.argsort(-cnt, kind="stable")[:npair])
    else:
        take = np.linspace(0, npair_total - 1, npair).astype(int)
    keep = []
    for pi in take:
        m = edges[edges[:, 0] == pi, 1]
        keep += list(m if len(m) <= kmax else rng.choice(m, kmax, replace=False))
    keep = np.unique(keep)
    streams = [z["pts"][off[i]:off[i + 1]] for i in keep]
    rgb = []
    for sl in streams:                                 # local tangent -> directional colour
        d = np.gradient(sl, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        rgb.append(np.abs(d))
    # only the 3HG nodes the drawn connections land on: a sphere with no fibre reaching it says
    # nothing the panel is about, and 456 of the 576 pairs are not drawn
    used = np.unique(z["pair_nodes"][take])
    nh = z["nodes_h"][np.searchsorted(z["hidx"], used)]
    np.savez(CACHE, pts=np.vstack(streams), lens=np.array([len(sl) for sl in streams]),
             rgb=(np.clip(np.vstack(rgb), 0, 1) * 255).astype(np.uint8),
             nodes_h=nh, nodes_h_all=z["nodes_h"], nodes_c=z["nodes_c"], vw=z["vw"],
             faces=z["faces"], parc=z["parc"], npair=npair_total, ndrawn_pairs=len(take))
    print(f"drew {len(take)} of {npair_total} pairs, {len(keep)} streamlines "
          f"({len(keep) / len(take):.1f} per drawn pair, thinnest bundle {cnt[take].min()}), "
          f"{len(nh)} of {len(z['nodes_h'])} 3HG nodes reached", flush=True)
    return CACHE


def render():
    import pyvista as pv
    z = np.load(CACHE)
    pts, lens, vw, f = z["pts"], z["lens"], z["vw"], z["faces"]
    lines, off = [], 0
    for k in lens:
        lines += [int(k)] + list(range(off, off + int(k))); off += int(k)
    pdata = pv.PolyData(); pdata.points = pts; pdata.lines = np.array(lines)
    pdata.point_data["rgb"] = z["rgb"]

    surf = pv.PolyData(vw, np.c_[np.full(len(f), 3), f].ravel())
    view, up = camera.frame(vw, z["parc"])
    pl = pv.Plotter(off_screen=True, window_size=(2200, 1750), border=False, lighting="light kit")
    pl.background_color = "white"; pl.enable_anti_aliasing("ssaa")
    pl.add_mesh(surf, color=CORTEX, opacity=0.10, smooth_shading=True, specular=0.0)
    pl.add_mesh(pdata.tube(radius=0.105, n_sides=6), scalars="rgb", rgb=True,
                smooth_shading=True, specular=0.10)
    # nodes as real spheres in world units, so they keep their size relative to the brain at any
    # output scale; point_size is in pixels and vanishes when the panel is 55 mm wide
    ball = pv.Sphere(radius=1.05, theta_resolution=16, phi_resolution=16)
    for key, col in (("nodes_c", NODE_C), ("nodes_h", NODE_H)):
        g = pv.PolyData(z[key]).glyph(geom=ball, scale=False, orient=False)
        pl.add_mesh(g, color=col, smooth_shading=True, specular=0.30, specular_power=20)
    camera.place(pl, surf, view, up, zoom=1.52)
    pl.screenshot(f"{OUT}/_net_lat.png", transparent_background=True)
    print(f"wrote _net_lat.png  ({len(lens)} streamlines, {len(z['nodes_h'])} 3HG + "
          f"{len(z['nodes_c'])} crown nodes), anterior on the "
          f"{camera.anterior_on_screen(vw, z['parc'], view, up)}", flush=True)


if __name__ == "__main__":
    import os
    if not os.path.exists(SEL):
        select()
    thin()
    if "--thin-only" not in sys.argv:
        render()

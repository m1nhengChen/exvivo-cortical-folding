#!/usr/bin/env python3
"""The GyralNet crest network as drawable geometry. Shared by render_classes.py (whole-brain
surfaces) and render_match_zoom.py (the magnified patches), so the two draw the same lines."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np

GNET = (_env.ROOT + "/surf/three_hinge/gnet")
C_SKEL = "#2B2B28"                    # ink; must not be confused with the shading boundary


def skeleton_edges(sid, hemi="rh"):
    """the crest network as vertex-index pairs on the white surface's own indexing"""
    import pyvista as pv
    m = pv.read(f"{GNET}/Subject{sid}/out/{hemi}_surf_skelenton_allpoints_final_refine.vtk")
    L, i, seg = m.lines, 0, []
    while i < len(L):
        k = int(L[i]); c = L[i + 1:i + 1 + k]
        seg.append(np.c_[c[:-1], c[1:]]); i += k + 1
    return np.vstack(seg)


def tube(vw, nrm, edges, radius):
    """the network as a thin tube lifted just off the surface, instead of painted vertices: a
    painted band is one vertex ring wide - about 1 mm - and staircases along the mesh"""
    import pyvista as pv
    pd = pv.PolyData()
    pd.points = vw + nrm * 0.10
    pd.lines = np.c_[np.full(len(edges), 2), edges].ravel()
    return pd.extract_points(np.unique(edges), adjacent_cells=True).extract_surface() \
             .tube(radius=radius, n_sides=14)

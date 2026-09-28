#!/usr/bin/env python3
"""Camera frames for the surfaces.

The specimens sit in the scanner at an arbitrary orientation, so the view has to be
derived from the data.  Principal axes are not enough -- their signs are arbitrary,
which is what rendered the first figures upside down, and re-deriving the frame per
geometry made the inflated surface come out as a mirror of the white one.

For white/pial: superior/inferior and anterior/posterior parcels fix `up` and the
anterior direction; the medial wall fixes which side is lateral.  That gives the
conventional right-hemisphere lateral view -- anterior on the right, because standing
to someone's right and looking at their right hemisphere, their nose points your way.

The same frame serves every geometry, because mris_inflate does not rotate (a Kabsch fit
between white and inflated comes out at 0.0 degrees).  It only works once all of them sit
in the same handedness -- see the tk2ras note in surf_to_vtk.py.  Deriving a frame per
geometry from surfaces in mirror-related frames is what made the inflated surface render
as a left hemisphere.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np

SUP = [2028, 2029, 2017, 2022, 2024]   # superiorfrontal, superiorparietal, paracentral,
                                       # postcentral, precentral
INF = [2033, 2009, 2007, 2006, 2016]   # temporalpole, inferiortemporal, fusiform,
                                       # entorhinal, parahippocampal
ANT = [2032, 2027, 2012, 2014]         # frontalpole, rostralmiddlefrontal, orbitofrontal
POS = [2011, 2005, 2021, 2013]         # lateraloccipital, cuneus, pericalcarine, lingual


def _com(points, parcel, ids):
    return points[np.isin(parcel, ids)].mean(0) - points.mean(0)


def frame(points, parcel):
    """(view, up) for a right-hemisphere lateral view: anterior right, superior up."""
    up = _com(points, parcel, SUP) - _com(points, parcel, INF)
    up /= np.linalg.norm(up)
    ant = _com(points, parcel, ANT) - _com(points, parcel, POS)
    ant -= up * (ant @ up)
    ant /= np.linalg.norm(ant)
    lat = np.cross(ant, up)                            # anterior x superior = right
    lat /= np.linalg.norm(lat)
    if _com(points, parcel, [0]) @ lat > 0:            # ...pointing away from the medial wall
        lat = -lat
    return lat, up


def anterior_on_screen(points, parcel, view, up):
    """'left' or 'right' -- which side of the image the frontal pole lands on."""
    right = np.cross(-view, up)                        # VTK screen-right
    return "right" if _com(points, parcel, [2032]) @ right > 0 else "left"


def place(pl, mesh, view, up, zoom=1.45, dist=400):
    pl.camera.position = tuple(mesh.center + view * dist)
    pl.camera.focal_point = tuple(mesh.center)
    pl.camera.up = tuple(up)
    pl.reset_camera()
    pl.camera.zoom(zoom)

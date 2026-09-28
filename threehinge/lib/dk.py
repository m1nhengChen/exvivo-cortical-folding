#!/usr/bin/env python3
"""Read a DK annot into FreeSurfer label numbers, whichever naming it uses.

`rh.aparc.annot` is now FreeSurfer's own parcellation, whose regions carry real DK names
("bankssts") and the official colour table.  The dataset's own version is kept alongside
as `rh.aparc.dataset.annot`, and its regions are named `ctx-rh-2001` etc. -- numbers,
because that is all the volume labels gave us.  Both resolve to the same 2001-2035 ids.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os
import numpy as np
import nibabel as nib

FS = (_env.ROOT + "/surf/fs")
LUT = os.environ.get("FREESURFER_HOME", _env.FREESURFER_HOME) + "/FreeSurferColorLUT.txt"


def name2id():
    m = {}
    with open(LUT) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5 and p[0].isdigit() and 2000 <= int(p[0]) <= 2035:
                m[p[1].replace("ctx-rh-", "")] = int(p[0])
    return m


def id2name():
    return {v: k for k, v in name2id().items()}


def read(subj, annot="rh.aparc.annot"):
    """(labels as 2001-2035 ids, ctab, names).  Non-cortical vertices come back as 0."""
    lab, ctab, names = nib.freesurfer.read_annot(f"{FS}/{subj}/label/{annot}")
    lut = name2id()
    out = np.zeros(len(lab), np.int32)
    for i, n in enumerate(names):
        n = (n.decode() if isinstance(n, bytes) else n).replace("ctx-rh-", "")
        if n.isdigit():
            out[lab == i] = int(n)                       # dataset's numeric naming
        elif n in lut and n not in ("unknown", "corpuscallosum"):
            out[lab == i] = lut[n]                       # FreeSurfer's DK names
    return out, ctab, names


def rgb(subj, annot="rh.aparc.annot"):
    """Per-vertex colour straight from the annot's own table."""
    lab, ctab, names = nib.freesurfer.read_annot(f"{FS}/{subj}/label/{annot}")
    col = np.full((len(lab), 3), 190, np.uint8)
    for i in range(len(names)):
        m = lab == i
        if m.any():
            col[m] = ctab[i, :3]
    return col

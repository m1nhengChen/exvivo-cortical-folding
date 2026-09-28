#!/usr/bin/env python3
"""Stage 3: put the dataset's own labels and parameter maps onto the surfaces.

Two products per subject:

  rh.aparc.dataset.annot   the *dataset's* CortexSeg parcellation, voted onto vertices.
                   NOT the default: rh.aparc.annot is FreeSurfer's own DK, which follows
                   this individual's sulci (56.6% of its boundaries fall in a sulcus, vs
                   49.2% for the dataset's -- chance is 47.0%).  See fs_dk.sh.

  rh.<map>.mgh     every modality sampled at a set of cortical depths.  All four
                   acquisition spaces already share world coordinates (verified:
                   centres of mass agree to <0.7 mm), so a map is read straight
                   at the vertex world coordinate -- no registration involved.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
from scipy import ndimage as ndi
from collections import Counter

ROOT = (_env.ROOT + "/28211843")
FS = (_env.ROOT + "/surf/fs")
H = "rh"
DEPTHS = np.linspace(0.1, 0.9, 5)          # fractional depth, 0 = white, 1 = pial
PARC_DEPTHS = np.linspace(0.15, 0.85, 9)   # denser for the parcellation vote
MAJORITY_PASSES = 6
MIN_ISLAND = 1500                          # vertices; smaller non-largest fragments absorbed

MAPS = {                                    # name -> path template
    "T2w":            "{s}_T2w/T2w.nii.gz",
    "FA":             "{s}_DTI/FA.nii.gz",
    "AD":             "{s}_DTI/AD.nii.gz",
    "RD":             "{s}_DTI/RD.nii.gz",
    "ADC":            "{s}_DTI/ADC.nii.gz",
    "QSM":            "{s}_QSM/QSM.nii.gz",
    "QSM_pos":        "{s}_QSM/QSM_pos.nii.gz",
    "QSM_neg":        "{s}_QSM/QSM_neg.nii.gz",
    "T2star":         "{s}_QSM/T2star_map.nii.gz",
    "T1":             "{s}_Mapping/T1_map.nii.gz",
    "T2":             "{s}_Mapping/T2_map.nii.gz",
    "fiber_ratio":    "{s}_DBSI/fiber_ratio_map.nii.gz",
    "restricted_ratio": "{s}_DBSI/restricted_ratio_map.nii.gz",
    "hindered_ratio": "{s}_DBSI/hindered_ratio_map.nii.gz",
    "water_ratio":    "{s}_DBSI/water_ratio_map.nii.gz",
}


def surf_world(subj, name):
    """Vertex coords of a FreeSurfer surface in scanner/world mm.

    `v + cras` only holds for LIA volumes; the grids built by prep_volumes.py are
    RAS+, so go through the volume's own tkr matrix.
    """
    mgh = nib.load(f"{FS}/{subj}/mri/brain.finalsurfs.mgz")
    tk2ras = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{subj}/surf/{name}")
    return nib.affines.apply_affine(tk2ras, v), f


def sample(img, xyz, order=1):
    inv = np.linalg.inv(img.affine)
    ijk = nib.affines.apply_affine(inv, xyz).T
    data = np.asarray(img.dataobj, dtype=np.float32)
    return ndi.map_coordinates(data, ijk, order=order, mode="constant", cval=np.nan)


def adjacency(faces, nv):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                   faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]])
    e = e[np.argsort(e[:, 0], kind="stable")]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(e[:, 0], minlength=nv))])
    return indptr, e[:, 1]


def _neighbour_modes(lab, indptr, nbr, ignore_zero=True):
    """Most common neighbouring label of every vertex."""
    n = len(indptr) - 1
    ids = np.unique(lab)
    k = np.searchsorted(ids, lab)[nbr]                      # neighbour labels, compacted
    row = np.repeat(np.arange(n), np.diff(indptr))
    cnt = np.zeros((n, len(ids)), np.int16)
    np.add.at(cnt, (row, k), 1)
    if ignore_zero and ids[0] == 0:
        cnt[:, 0] = 0
    return ids[cnt.argmax(1)], cnt.max(1)


def fill_holes(lab, indptr, nbr):
    """Fill unlabelled vertices, but leave the medial wall alone.

    The medial wall is genuinely non-cortical, so it must stay 0.  Dilating labels into
    every unlabelled vertex (the old behaviour) ate ~12 vertex rings off its rim.
    Keep the largest unlabelled component; fill the rest.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(lab)
    zero = np.flatnonzero(lab == 0)
    if not len(zero):
        return lab
    mask = np.zeros(n, bool); mask[zero] = True
    keep = mask[nbr] & np.repeat(mask, np.diff(indptr))
    row = np.repeat(np.arange(n), np.diff(indptr))
    g = csr_matrix((np.ones(keep.sum()), (row[keep], nbr[keep])), shape=(n, n))
    ncc, cc = connected_components(g, directed=False)
    sz = np.bincount(cc[zero])
    wall = zero[cc[zero] == sz.argmax()]                     # the medial wall itself

    lab = lab.copy()
    protect = np.zeros(n, bool); protect[wall] = True
    for _ in range(30):
        holes = (lab == 0) & ~protect
        if not holes.any():
            break
        mode, cnt = _neighbour_modes(lab, indptr, nbr)
        fill = holes & (cnt > 0)
        if not fill.any():
            break
        lab[fill] = mode[fill]
    return lab


def majority_filter(lab, indptr, nbr, passes):
    """Relabel a vertex to its neighbours' majority -- smooths boundaries, kills speckle."""
    lab = lab.copy()
    cortex = lab > 0
    for _ in range(passes):
        mode, cnt = _neighbour_modes(lab, indptr, nbr)
        # only flip when the neighbourhood clearly disagrees, and never touch the wall
        flip = cortex & (cnt >= 4) & (mode != lab) & (mode > 0)
        if not flip.any():
            break
        lab[flip] = mode[flip]
    return lab


def restore_lost(lab, orig, indptr, nbr):
    """Put back any label the cleanup erased outright.

    The majority filter has no floor: a region that is both small and thin can be eaten
    entirely before drop_islands (which does protect each label's largest component) ever
    sees it.  That is how Subject6 lost parahippocampal.  So stamp any vanished label back
    from its largest component in the pre-cleanup labelling.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(lab)
    deg = np.diff(indptr)
    row = np.repeat(np.arange(n), deg)
    lab = lab.copy()
    for p in set(np.unique(orig[orig > 0]).tolist()) - set(np.unique(lab[lab > 0]).tolist()):
        v = np.flatnonzero(orig == p)
        keep = np.isin(nbr, v) & np.isin(row, v)
        idx = -np.ones(n, int); idx[v] = np.arange(len(v))
        g = csr_matrix((np.ones(keep.sum()), (idx[row[keep]], idx[nbr[keep]])),
                       shape=(len(v), len(v)))
        _, cc = connected_components(g, directed=False)
        lab[v[cc == np.bincount(cc).argmax()]] = p
    return lab


def drop_islands(lab, indptr, nbr, min_size):
    """Absorb parcel fragments below `min_size` into whatever surrounds them.

    A label's own largest component is always protected, whatever its size -- otherwise a
    genuinely small parcel (frontal pole, temporal pole) gets eaten whole, and the subject
    ends up with 33 regions instead of 34.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(lab)
    lab = lab.copy()
    deg = np.diff(indptr)
    row = np.repeat(np.arange(n), deg)
    for _ in range(30):
        keep = (lab[nbr] == np.repeat(lab, deg)) & np.repeat(lab > 0, deg)
        g = csr_matrix((np.ones(keep.sum()), (row[keep], nbr[keep])), shape=(n, n))
        _, cc = connected_components(g, directed=False)
        sz = np.bincount(cc)

        biggest = {}                                  # label -> its largest component id
        for c in np.unique(cc[lab > 0]):
            l = lab[cc == c][0]
            if l not in biggest or sz[c] > sz[biggest[l]]:
                biggest[l] = c
        protected = np.zeros(len(sz), bool)
        protected[list(biggest.values())] = True

        small = (lab > 0) & (sz[cc] < min_size) & ~protected[cc]
        if not small.any():
            break
        mode, cnt = _neighbour_modes(np.where(small, 0, lab), indptr, nbr)
        fix = small & (cnt > 0)
        # A fragment fully ringed by the medial wall (cnt==0, no labelled neighbour) never
        # gets absorbed by the line above, so tiny cortical slivers hanging off the medial
        # edge survive forever.  Those are noise (1-45 vertices); fold them into the medial
        # wall.  Only when the fragment is genuinely small (< min_size), so a real parcel is
        # never eaten this way.
        strays = small & (cnt == 0)
        if not fix.any() and not strays.any():
            break
        lab[fix] = mode[fix]
        lab[strays] = 0
    return lab


def run(sid):
    s = f"Subject{sid}"
    w, faces = surf_world(s, f"{H}.white")
    p, _ = surf_world(s, f"{H}.pial")
    nv = len(w)
    os.makedirs(f"{FS}/{s}/surf_maps", exist_ok=True)

    # ---- parcellation: majority vote over depths, nearest-neighbour ----
    seg = nib.load(f"{ROOT}/{s}/{s}_Mask/T2w_CortexSeg_mask.nii.gz")
    votes = np.stack([sample(seg, w + d * (p - w), order=0) for d in PARC_DEPTHS])
    votes = np.nan_to_num(votes, nan=0).astype(np.int32)
    # majority vote per vertex, vectorised: compact the label ids then bincount rows
    uniq = np.unique(votes)
    comp = np.searchsorted(uniq, votes)                       # (ndepth, nv) -> 0..len(uniq)-1
    cnt = np.zeros((nv, len(uniq)), np.int16)
    np.add.at(cnt, (np.tile(np.arange(nv), len(PARC_DEPTHS)), comp.ravel()), 1)
    cnt[:, uniq == 0] = 0                                     # background never wins
    lab = np.where(cnt.max(1) > 0, uniq[cnt.argmax(1)], 0).astype(np.int32)

    # Nearest-neighbour sampling of a volume label carries no spatial regularity, so the
    # raw vote leaves the surface speckled: 34 parcels came out as 670 connected
    # components, 590 of the extra islands under 50 vertices.  Clean it up on the mesh.
    indptr, nbr = adjacency(faces, nv)
    lab = fill_holes(lab, indptr, nbr)         # small unlabelled gaps, medial wall kept
    orig = lab.copy()
    lab = majority_filter(lab, indptr, nbr, MAJORITY_PASSES)
    lab = drop_islands(lab, indptr, nbr, MIN_ISLAND)
    lab = restore_lost(lab, orig, indptr, nbr)

    ids = sorted(set(lab[lab > 0].tolist()))
    names, ctab = ["unknown"], [[25, 5, 25, 0]]
    rng = np.random.default_rng(42)
    for v in ids:
        names.append(f"ctx-rh-{v}")
        ctab.append(list(rng.integers(30, 230, 3)) + [0])
    ctab = np.array(ctab, np.int32)
    ctab = np.column_stack([ctab, ctab[:, 0] + ctab[:, 1] * 2**8 + ctab[:, 2] * 2**16])
    remap = {0: 0, **{v: i + 1 for i, v in enumerate(ids)}}
    nib.freesurfer.write_annot(f"{FS}/{s}/label/{H}.aparc.dataset.annot",
                               np.array([remap[x] for x in lab], np.int32),
                               ctab, names, fill_ctab=False)

    # ---- parameter maps at each cortical depth ----
    ok = []
    for name, tpl in MAPS.items():
        path = f"{ROOT}/{s}/{tpl.format(s=s)}"
        if not os.path.exists(path):
            continue
        img = nib.load(path)
        prof = np.stack([sample(img, w + d * (p - w)) for d in DEPTHS])   # (depth, vertex)
        nib.save(nib.MGHImage(prof.T[:, None, :].astype(np.float32), np.eye(4)),
                 f"{FS}/{s}/surf_maps/{H}.{name}.profile.mgh")            # nv x 1 x ndepth
        mid = np.nanmean(prof, axis=0)
        nib.freesurfer.write_morph_data(f"{FS}/{s}/surf_maps/{H}.{name}.mid", mid)
        ok.append(name)

    th = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.thickness")
    print(f"{s}: {nv} vertices | {len(ids)} parcels | thickness "
          f"{np.mean(th):.2f}+-{np.std(th):.2f} mm | maps: {len(ok)}")


if __name__ == "__main__":
    for i in (range(1, 7) if len(sys.argv) < 2 else [int(sys.argv[1])]):
        try:
            run(i)
        except FileNotFoundError as e:
            print(f"Subject{i}: skipped ({e.filename} missing)")

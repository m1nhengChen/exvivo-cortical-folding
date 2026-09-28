#!/usr/bin/env python3
"""Mesh-label cleanup utilities on a triangulated surface: adjacency, hole filling,
majority smoothing, island removal, and restoration of labels erased by smoothing."""
import numpy as np


MAJORITY_PASSES = 6

MIN_ISLAND = 1500

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
    """Fill unlabelled vertices, leaving the medial wall alone.

    The medial wall is non-cortical and must stay 0, so the largest unlabelled component
    is kept and only the remaining gaps are filled.
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
    """Restore any label the cleanup erased outright.

    The majority filter has no floor, so a small, thin region can be eaten entirely before
    drop_islands (which protects each label's largest component) sees it.  Any vanished
    label is stamped back from its largest component in the pre-cleanup labelling.
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
    """Absorb parcel fragments below `min_size` into the surrounding label.

    A label's largest component is always protected, whatever its size, so a genuinely
    small parcel (frontal pole, temporal pole) is never eaten whole.
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
        # A fragment fully ringed by the medial wall (cnt==0, no labelled neighbour) is not
        # absorbed above, leaving tiny cortical slivers on the medial edge.  Fold these into
        # the medial wall, but only when genuinely small (< min_size) so no real parcel is
        # eaten this way.
        strays = small & (cnt == 0)
        if not fix.any() and not strays.any():
            break
        lab[fix] = mode[fix]
        lab[strays] = 0
    return lab

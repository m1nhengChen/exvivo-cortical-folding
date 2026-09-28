#!/usr/bin/env python3
"""One anatomical plane, sampled from every modality the analysis uses, plus the tractogram on it.

    python3 render_voxels.py            # Subject4

The three volumes live on different grids - T2w at 0.475 mm, DBSI and DWI at 0.88 mm - so slicing
each at "its own middle slice" would show three different pieces of anatomy. Instead one plane is
fixed in WORLD space and every volume is sampled onto the same 2D grid, so the panels are the same
tissue and the resolution difference shows as blur rather than as a change of subject.

Also selects the streamlines that run within a thin slab of that plane and projects them onto it,
which is the tractography seen in voxel space rather than draped over a surface.

The plane is not "the middle slice". It is chosen so that render_lamina.py has something to cut:
of the axial slices, take the one holding the most frontal-lobe 3HG landmarks whose ribbon that
plane cuts nearly across, and centre the field of view on them. Frontal because that is where the
regional map puts the largest effects, and because the cut has to land on a gyral crown with a
3HG on it, which is not guaranteed at an arbitrary slice."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
from scipy import ndimage as ndi
import dk

ROOT = (_env.ROOT + "/28211843")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
OUT = (_env.ROOT + "/surf/three_hinge/results")
# Subject6: best diffusion image quality of the six by a clear margin - b0 SNR 35.1 against
# 21.6 / 22.5 / - / 5.1 / 19.3, mean-DW SNR 21.0 against 11.5 / 12.1 / - / 3.2 / 11.9, and the
# sharpest edges. Subject4, used earlier here, is the worst of the set.
SID = int(os.environ.get("SUBJ", "6"))
FOV = float(os.environ.get("FOV", "60.0"))      # mm across the sampled plane
PIX = 0.18                                       # mm of the common sampling grid
PIX_D = 0.06                                     # mm, the detail grid inside the box
BOX = float(os.environ.get("BOX", "12.0"))       # mm half-width of the detail box
SLAB = float(os.environ.get("SLAB", "3.0"))      # mm half-thickness for streamline selection
MINLEN = 20.0
H = "rh"
FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
LOBE = os.environ.get("LOBE", "frontal")
OBLIQ = 0.30                                     # |plane normal . cortical normal| ceiling
EDGE = float(os.environ.get("EDGE", "12.0"))     # mm of frame the landmark must stay clear of
COVER_MIN = float(os.environ.get("COVER_MIN", "0.45"))   # tissue floor, as a fraction of the best
PULL = float(os.environ.get("PULL", "0.45"))     # how far the frame slides from landmark to tissue
REGIONS = {"frontal": {"superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal",
                       "parsopercularis", "parstriangularis", "parsorbitalis",
                       "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                       "paracentral", "frontalpole"}}


# Anatomical axes from GROUPS of parcels, not from one pair. A single pair such as
# superiorfrontal - lateralorbitofrontal is a local frontal vector: on a tilted specimen it misses
# the true superior direction by 30-40 degrees and the slice comes out visibly rolled.
ANAT = {"sup": (("superiorfrontal", "precentral", "postcentral", "superiorparietal", "paracentral"),
                ("lateralorbitofrontal", "medialorbitofrontal", "inferiortemporal",
                 "temporalpole", "entorhinal", "fusiform")),
        "ant": (("frontalpole", "rostralmiddlefrontal", "parsorbitalis", "medialorbitofrontal",
                 "lateralorbitofrontal"),
                ("lateraloccipital", "cuneus", "pericalcarine", "lingual"))}


def tkr2world(sid):
    """surface (tkrRAS) -> scanner world, the transform the rest of this codebase uses

    read_geometry returns tkrRAS. The DWI affine, and therefore the cut plane, is in scanner world.
    Mixing the two silently rotates everything: for Subject4 the superior direction reads
    (0.16, 0.55, -0.82) in tkrRAS and (-0.16, 0.82, 0.55) in world, so anatomical axes derived in
    the wrong frame put the slice at an arbitrary angle."""
    mgh = nib.load(f"{FS}/Subject{sid}/mri/wm.mgz")
    return mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())


def axis(sid, which, nrm):
    """a world direction defined by two parcel centroids, projected into the slice

    Ex-vivo hemispheres are not scanned in a standard pose, so "up" has to be read off the anatomy
    rather than assumed to be +z."""
    parc = dk.read(f"Subject{sid}")[0]
    W, _ = nib.freesurfer.read_geometry(f"{FS}/Subject{sid}/surf/{H}.white")
    W = nib.affines.apply_affine(tkr2world(sid), W.astype(float))
    i2n = dk.id2name()
    nm = {str(i2n.get(int(p), i2n.get(int(p) - 2000, ""))).replace("ctx-rh-", ""): p
          for p in np.unique(parc[parc > 0])}
    hi, lo = ANAT[which]
    cen = lambda names: np.concatenate([W[parc == nm[n]] for n in names if n in nm]).mean(0)
    v = cen(hi) - cen(lo)
    if np.any(nrm):
        v = v - (v @ nrm) * nrm
    return v / np.linalg.norm(v)


def landmarks(sid, whole=False):
    """cortical 3HG of the requested lobe, with their outward white-surface normals"""
    import pyvista as pv
    s = f"Subject{sid}"
    W, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white"); W = W.astype(float)
    m = pv.PolyData(W, np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    tk = tkr2world(sid)
    nrm = (tk[:3, :3] @ m.point_data["Normals"].T).T          # rotate into world with the points
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    W = nib.affines.apply_affine(tk, W)
    if np.mean(np.sum((W - W.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    parc = dk.read(s)[0]
    hinge = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int)
    hinge = hinge[parc[hinge] > 0]
    if whole:
        return W[parc > 0], nrm[parc > 0]
    i2n = dk.id2name()
    want = None if whole else REGIONS.get(LOBE)
    if want:
        nm = [str(i2n.get(int(parc[h]), i2n.get(int(parc[h]) - 2000, ""))).replace("ctx-rh-", "")
              for h in hinge]
        hinge = hinge[np.isin(nm, list(want))]
    n = nrm[hinge] / np.linalg.norm(nrm[hinge], axis=1, keepdims=True)
    return W[hinge], n


def sample(img, XYZ, vol=None):
    A = np.asarray(img.dataobj, np.float32)
    if vol is not None:
        A = A[..., vol].mean(-1) if np.ndim(vol) else A[..., vol]
    ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), XYZ.reshape(-1, 3)).T
    return ndi.map_coordinates(A, ijk, order=1, mode="constant", cval=0).reshape(XYZ.shape[:2])


def main():
    s = f"Subject{SID}"
    dwi = nib.load(f"{TR}/{s}/dwi.nii.gz")
    msk = nib.load(f"{TR}/{s}/mask.nii.gz")
    M = np.asarray(msk.dataobj) > 0

    # ---- an ANATOMICAL plane, not a scanner slice ----
    # The specimen was not scanned in a standard pose, so the acquisition's own slices cut the
    # anatomy obliquely: no amount of in-plane rotation makes a constant-k slice look like a
    # sagittal section. Everything here is resampled anyway, so the plane is defined from the
    # anatomy instead - superior and anterior from parcel centroids, left-right as their cross
    # product - and swept along that axis to choose where to cut.
    sup = axis(SID, "sup", np.zeros(3))
    ant = axis(SID, "ant", np.zeros(3))
    ant = ant - (ant @ sup) * sup; ant /= np.linalg.norm(ant)
    nrm = np.cross(ant, sup); nrm /= np.linalg.norm(nrm)
    e1, e2 = ant, sup                                    # right = anterior, up = superior

    LW, LN = landmarks(SID)
    WALL, WN = landmarks(SID, whole=True)                # the whole hemisphere, for its extent
    T2 = nib.load(f"{ROOT}/{s}/{s}_T2w/T2w.nii.gz")

    def boundary_contrast(i):
        """how cleanly the white surface separates tissue on T2w around this landmark

        A cut is only worth showing if the segmentation is right there. Sample T2w 0.75 mm inside
        and 0.75 mm outside the white surface over the vertices near the landmark and near the
        plane, and take the standardized difference. A crisp boundary gives a large value; where
        the surface has strayed into grey matter, or grey and white are not separable in the first
        place, the two samples overlap and it collapses."""
        sel = ((np.linalg.norm(WALL - LW[i], axis=1) < 12.0) &
               (np.abs((WALL - LW[i]) @ nrm) < 3.0))
        if sel.sum() < 200:
            return -np.inf
        a = sample(T2, (WALL[sel] - 0.75 * WN[sel])[None])
        b = sample(T2, (WALL[sel] + 0.75 * WN[sel])[None])
        sd = np.sqrt((a.var() + b.var()) / 2)
        return float(abs(a.mean() - b.mean()) / sd) if sd > 0 else -np.inf
    obl = np.abs(LN @ nrm)
    ok = obl < OBLIQ
    if not ok.any():
        raise SystemExit(f"no {LOBE} 3HG cut across by a sagittal plane - raise OBLIQ")

    # tissue coverage of a candidate plane, on a coarse grid - the sweep runs ~60 offsets
    ijk_of = lambda X: nib.affines.apply_affine(np.linalg.inv(msk.affine), X)
    g = np.arange(-FOV / 2, FOV / 2, 1.5)
    GU, GV = np.meshgrid(g, g, indexing="xy")

    def coverage(o):
        X = o + GU[..., None] * e1 + GV[..., None] * e2
        c = np.round(ijk_of(X.reshape(-1, 3))).astype(int)
        inb = np.all((c >= 0) & (c < np.array(M.shape)), 1)
        out = np.zeros(len(c), bool)
        out[inb] = M[c[inb, 0], c[inb, 1], c[inb, 2]]
        return out.sum(), X.reshape(-1, 3)[out]

    # Sweep the plane along the left-right axis and keep every position that carries a usable
    # landmark and enough tissue, then take the most LATERAL of them. Scoring by tissue coverage
    # alone lands on the midline every time - the mid-sagittal cross-section is the largest one -
    # and a mid-sagittal slice shows the corpus callosum instead of the gyral convexity the paper
    # is about.
    WD = (WALL - WALL.mean(0)) @ nrm                     # the hemisphere's own extent
    d_all = (LW - WALL.mean(0)) @ nrm
    rows = []
    for t in np.arange(WD.min() + 4, WD.max() - 4, 1.5):
        o = WALL.mean(0) + t * nrm
        near = ok & (np.abs(d_all - t) < 1.2)
        if not near.any():
            continue
        cov, pts = coverage(o)
        if cov < 200:
            continue
        # Frame around the LANDMARK, pulled part of the way toward the slice's tissue centroid.
        # Centring on the centroid alone rejected every candidate: the field of view is 78 mm and
        # the hemisphere is longer than that, so a frontal landmark sits 30-50 mm anterior of the
        # centre of the slice and never cleared the edge test.
        tc = pts.mean(0); tc = tc - ((tc - o) @ nrm) * nrm
        # Among the landmarks this plane cuts across, take the one with the cleanest white-matter
        # boundary rather than the one cut most exactly perpendicular. Obliquity only has to be
        # small enough for the ribbon to read; a landmark sitting where the segmentation is poor
        # makes the whole detail row look wrong however well it is sectioned.
        cand = np.where(near)[0]
        best_i = cand[int(np.argmax([boundary_contrast(i) for i in cand]))]
        lm = LW[best_i] - ((LW[best_i] - o) @ nrm) * nrm
        ctr = lm + PULL * (tc - lm)
        if max(abs((LW[best_i] - ctr) @ e1), abs((LW[best_i] - ctr) @ e2)) > FOV / 2 - EDGE:
            continue
        rows.append((t, cov, ctr, best_i))
    if not rows:
        raise SystemExit("no plane carries both tissue and a usable landmark")
    cmax = max(r[1] for r in rows)
    keep = [r for r in rows if r[1] >= COVER_MIN * cmax]
    lat = [r for r in keep if r[0] >= np.median(WD)] or keep      # lateral half if there is one
    t, cov, origin, node = max(lat, key=lambda r: boundary_contrast(r[3]))
    fov = FOV

    # Which side are we looking from? A cut in the lateral half of the hemisphere is read as a
    # lateral view - eye on the lateral side - and then screen-right = forward x up = anterior. A
    # medial cut is read from the other side and anterior goes to the left. Getting this wrong
    # mirrors the brain, which is the kind of error a reader spots and an author does not.
    if t < np.median(WD):
        e1 = -e1
    print(f"  {s}: sagittal plane at {t:+.0f} mm on a {WD.min():.0f}..{WD.max():.0f} mm axis, "
          f"{'lateral' if t >= np.median(WD) else 'medial'} view, {int(cov)} tissue samples, "
          f"landmark |N.n| = {obl[node]:.3f}, white-boundary contrast "
          f"{boundary_contrast(node):.2f} SD", flush=True)

    ctr_box = LW[node] - ((LW[node] - origin) @ nrm) * nrm
    bu = float((ctr_box - origin) @ e1); bv = float((ctr_box - origin) @ e2)

    def grid(c, half, pix):
        t = np.arange(-half, half, pix)
        U, V = np.meshgrid(t, t, indexing="xy")
        return c + U[..., None] * e1 + V[..., None] * e2

    FB = nib.load(f"{ROOT}/{s}/{s}_DBSI/fiber_ratio_map.nii.gz")
    bv_ = np.loadtxt(f"{TR}/{s}/bvals.txt")
    dwv = np.where(bv_ > bv_.min() + 10)[0]                # the diffusion-weighted volumes

    def shot(XYZ, pix, win=None):
        """the three contrasts on one grid, each scaled inside the brain and blanked outside

        The DBSI fit fails in the partial-volume voxels at the brain edge and returns speckle;
        unmasked it draws a ragged rind that reads as an artefact of the figure. Eroding the mask
        by ~1 voxel removes it without touching cortex.

        `win` carries the intensity window from one grid to another. Scaling each grid on its own
        percentiles made the context view and the detail view of the SAME contrast look like two
        different acquisitions - the small box holds a different mix of grey and white matter, so
        its 1-99 percentile window is not the plane's."""
        mk = sample(msk, XYZ) > 0.85
        mk = ndi.binary_erosion(mk, iterations=max(1, int(round(1.0 / pix))))
        out, w = {}, {}
        for k_, A in (("T2w", sample(T2, XYZ)), ("DWI", sample(dwi, XYZ, vol=dwv)),
                      ("DBSI", sample(FB, XYZ))):
            if win is None:
                lo, hi = np.percentile(A[mk], [1, 99]) if mk.any() else (A.min(), A.max())
            else:
                lo, hi = win[k_]
            w[k_] = (lo, hi)
            out[k_] = np.where(mk, np.clip((A - lo) / max(hi - lo, 1e-6), 0, 1), np.nan)
        return out, w

    planes, win = shot(grid(origin, fov / 2, PIX), PIX)
    detail, _ = shot(grid(ctr_box, BOX, PIX_D), PIX_D, win)   # same window as the context view
    mask2 = np.isfinite(planes["T2w"])

    # ---- streamlines through the slab, projected onto the plane ----
    from dipy.io.streamline import load_tractogram
    from dipy.io.stateful_tractogram import Space
    from dipy.tracking.streamline import length as sl_length
    tg = load_tractogram(f"{TR}/{s}/whole_brain.trk", dwi, to_space=Space.RASMM,
                         bbox_valid_check=False)
    SL = tg.streamlines
    keep = np.asarray(sl_length(SL)) > MINLEN
    segs, cols, inbox = [], [], []
    for i in np.where(keep)[0]:
        p = np.asarray(SL[i])
        d = (p - origin) @ nrm
        m = np.abs(d) < SLAB
        if m.sum() < 4:
            continue
        idx = np.where(m)[0]
        for grp in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(grp) < 4:
                continue
            q = p[grp]
            uv = np.c_[(q - origin) @ e1, (q - origin) @ e2]
            if np.abs(uv).max() > fov / 2:
                continue
            v3 = q[-1] - q[0]
            nn = np.linalg.norm(v3)
            segs.append(uv)
            cols.append(np.abs(v3 / nn) if nn > 0 else np.zeros(3))
            inbox.append(bool(np.abs(uv[:, 0] - bu).min() < BOX and
                              np.abs(uv[:, 1] - bv).min() < BOX))
    print(f"  {s}: {len(segs)} streamline segments in a {2*SLAB:.1f} mm slab, "
          f"{int(np.sum(inbox))} of them inside the {2*BOX:.0f} mm box", flush=True)
    # the plane travels with the data: render_lamina.py cuts its shells with THIS plane, so
    # panels a, b and c are one location rather than three
    np.savez_compressed(f"{OUT}/_voxels_S{SID}.npz",
                        extent=np.array([-fov / 2, fov / 2, -fov / 2, fov / 2]),
                        origin=origin, e1=e1, e2=e2, nrm=nrm, node=LW[node],
                        segs=np.array(segs, object), cols=np.array(cols),
                        inbox=np.array(inbox), box=np.array([bu, bv, BOX]),
                        **{f"d_{k}": v for k, v in detail.items()}, **planes)
    print(f"  wrote {OUT}/_voxels_S{SID}.npz", flush=True)


if __name__ == "__main__":
    main()

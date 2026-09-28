#!/usr/bin/env python3
"""Cohort plate: the six ex-vivo hemispheres, each in three views, with a DK lobe parcellation.

    python3 fig_cohort.py                  # full render + compose
    python3 fig_cohort.py --compose-only   # re-lay-out from the cached renders

Per specimen, two rows of two. Top: the same para-sagittal level twice - once from the original
acquisition and once from brain.image.b50 - so the two images differ only in source. Bottom: the
pial surface, lateral and medial. Two cells in each row means both rows fill the panel at the same
scale, so a brain is the same size whether it is a section or a render.

Why sagittal rather than axial or coronal. On half a brain the axial plane runs the full 170 mm
anterior-posterior length across only 73 mm of width and crosses the excised brainstem, and the
coronal is a 73 mm strip; neither reads at plate size.

The original acquisition (T2w.nii.gz) is already in the surfaces' physical space - grey 506 against
white 393 sampled at the mid-cortical and deep-white positions, coverage 1.00 - so it is sampled
with no registration step. Grey is brighter than white, which is what T2 weighting does. Set
ORIG=orig to use FreeSurfer's orig.mgz instead, but note that these recons were driven by images
synthesised from a tissue segmentation, so orig / T1 / norm come back in flat grey steps that read
as a segmentation rather than an image. Bottom: the pial surface in lateral and medial view, then the lateral
view again coloured by lobe. The sections show that the reconstruction has real tissue under it;
the plain surfaces carry the folding pattern, which is what a reader compares across specimens;
the coloured one gives the anatomical frame so the comparison is anchored.

The sections are NOT voxel planes. These hemispheres sat in the scanner at arbitrary orientations,
so a plane through the voxel grid is an oblique cut through the anatomy. Each section is resampled
on the specimen's own anatomical axes - the same superior and lateral directions the camera is
built from - so "sagittal" means sagittal for that specimen.

Each plane is placed at its level of MAXIMUM cross-sectional area, not through the surface
centroid. The centroid of a hemisphere's white surface sits low, and an axial cut there lands in
the temporal lobe beside the excised brainstem, which produces a notched sliver that no reader
recognises as axial. Maximum area is a stated rule, it is reproducible, and it puts each section
where the anatomy is most legible.

Sizes are TRUE relative sizes. The camera uses parallel projection with one fixed scale shared by
every render, instead of PyVista's reset_camera, which fits each mesh to its own viewport and would
silently normalise six differently sized hemispheres to the same apparent size."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
import dk, camera
import natstyle as ns

FS = (_env.ROOT + "/surf/fs")
OUT = (_env.ROOT + "/surf/three_hinge/results")
CACHE = f"{OUT}/_cohort"
H = "rh"; SUBJ = [1, 2, 3, 4, 5, 6]
# White, not pial: every landmark, sample and patch in this work sits on the white surface, so the
# plate should show the surface the analysis actually runs on. Pial photographs better - the sulci
# close over and the specimen looks like an intact brain - but it is not what anything is measured
# on, and an ex-vivo white surface at 0.5 mm carries reconstruction roughness a reader should see.
SURF = os.environ.get("SURF", "white")
# Three-hinge landmarks drawn as spheres on the surface. Medial-wall vertices are dropped first:
# GyralNet runs on the whole surface and reports junctions on the medial wall, where there is no
# folding to be a junction of. Every analysis in this work masks them, so the plate must too, or
# it advertises landmarks the statistics never used.
SHOW_3HG = os.environ.get("SHOW_3HG", "1") == "1"
# Collapse landmarks the analysis cannot resolve, exactly as tracto/conn_matrix.py does: single
# linkage at MERGE_MM on the sub-white-matter PATCH centres (2 mm inside the white surface, which
# is where the 3 mm spheres actually sit), keeping each cluster's medoid vertex. Clustering on the
# surface vertices instead would give different clusters, because two points on opposite banks of
# a sulcus are far apart on the surface and close underneath it. 0 disables merging.
MERGE_MM = float(os.environ.get("MERGE", "6.0"))
# separate output per merge setting, so the two versions do not overwrite each other
# The gyral skeleton itself: GyralNet's curve network along the gyral crests, of which a 3HG is
# simply a vertex of degree >= 3. Drawing it makes the landmark definition visible instead of
# asserted - the spheres sit where three branches meet.
SHOW_SKEL = os.environ.get("SKEL", "0") == "1"
# At plate size a hemisphere is ~20 mm wide for ~140 mm of brain, a 1:7 reduction, so a
# 0.34 mm tube lands at 0.05 mm on the page and simply is not there.
SKEL_R = float(os.environ.get("SKEL_R", "0.95"))   # tube radius, world mm
SKEL_LIFT = 0.70                                   # nudge off the surface so it is not embedded
# The skeleton follows mesh edges, so its path zigzags at the mesh resolution: the median turn at
# a path point is 146 deg (33 deg off straight) and 43% of joints turn more than 40 deg. With a
# tube twice as thick as an edge is long, every joint bulges and the curve reads as jittery.
# Laplacian smoothing ALONG THE SKELETON GRAPH fixes it - 12 passes take the median turn to 175
# deg with no sharp joints left, and pull the curve at most 1.8 mm off the surface. Lifting along
# the normals is not the cause and smoothing the normals would not help.
SKEL_SMOOTH = int(os.environ.get("SKEL_SMOOTH", "12"))
SKEL_COL = "#2F6E8C"        # cool, so it separates from the warm 3HG orange
NAME = ("FIG_cohort" if MERGE_MM > 0 else "FIG_cohort_unmerged") + ("_skel" if SHOW_SKEL else "")
INWARD = 2.0
# Sphere radius follows the spacing. Merged landmarks are never closer than MERGE_MM, so 1.9 mm
# markers stay separate and are big enough to read at plate size; unmerged ones sit 3.5-5.3 mm
# apart and need to be half that, or ~450 markers fuse into a crust.
HINGE_R = float(os.environ.get("HINGE_R", "1.90" if MERGE_MM > 0 else "0.95"))
# When landmarks are drawn the base tone stops being the subject and becomes the ground: the
# flesh colour and the 3HG orange sit at almost the same hue and lightness, so the spheres
# disappear into it. Desaturate the surface instead of recolouring the landmarks - orange is
# 3HG everywhere else in this work and should not change meaning here.
GROUND = "#D8D2CA"
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
FONTS = ["Arial", "Liberation Sans", "Nimbus Sans", "Helvetica", "DejaVu Sans"]
FLESH = "#E4B49E"                                  # the plate's base tone, warm and low-chroma
LOBES = [("frontal", "#E79FB4",
          ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
           "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
           "precentral", "paracentral", "frontalpole"]),
         ("parietal", "#5BBFBE",
          ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"]),
         ("temporal", "#8CC04F",
          ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
           "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"]),
         ("occipital", "#E8697D",
          ["lateraloccipital", "lingual", "cuneus", "pericalcarine"]),
         ("cingulate", "#F0C24B",
          ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate",
           "isthmuscingulate"]),
         ("insula", "#9A85C4", ["insula"])]
VIEWS = ["lat", "med"]
SUBJ_FS = 7.5        # specimen label point size; the composite plate raises it
# Internal spacing, in mm. Exposed so a composite plate can run tighter than the standalone
# figure, where the block has a whole canvas to itself and can afford to breathe.
SP = dict(GX=4.0, GAP=1.2, TITLE=4.4, VGAP=1.4, ROWGAP=6.0)
# Sagittal level, as a fraction of the medial-face-to-lateral-convexity span. A far-lateral cut
# is all cortical ribbon and no landmarks; a para-sagittal one carries the ventricle, thalamus and
# callosal body, which is what makes a section readable as anatomy rather than as texture.
# One default with per-specimen overrides: the same fraction does not land on the same anatomy in
# every hemisphere. In Subject4 it falls through the widest part of the lateral ventricle and the
# section comes out as a black void; 0.42 clears it and keeps thalamus, temporal lobe and ribbon.
LEVEL = 0.30
LEVEL_BY_SUBJ = {4: 0.42}
# (label shown under the section, source volume)
SLICES = [(os.environ.get("ORIG", "raw"), os.environ.get("ORIG", "raw")), ("b50", "b50")]
# brain.image.b50 is the only conformed volume here that has not been posterised by intensity
# normalisation; orig/T1/norm/finalsurfs all come back in flat grey steps that read as a
# segmentation rather than an image.
ROOT = (_env.ROOT + "/28211843")
SRC = {"raw": "{r}/Subject{i}/Subject{i}_T2w/T2w.nii.gz",
       "orig": "{f}/Subject{i}/mri/orig.mgz",
       "b50": "{f}/Subject{i}/mri/brain.image.b50.mgz"}
NICE = {"raw": "T2w", "orig": "orig", "b50": "b50"}
NPIX = 460


def skeleton_lines(sid, hemi="rh"):
    """the skeleton's edges as vertex-index pairs; its points are the white-surface vertices"""
    import pyvista as pv
    m = pv.read(f"{GNET}/Subject{sid}/out/{hemi}_surf_skelenton_allpoints_final_refine.vtk")
    L = np.asarray(m.lines); out = []; i = 0
    while i < len(L):
        k = int(L[i]); seg = L[i + 1:i + 1 + k]
        out += [(int(seg[j]), int(seg[j + 1])) for j in range(k - 1)]
        i += k + 1
    return np.array(out, int)


def lobe_rgb(sid):
    """per-vertex RGB: flesh everywhere, lobe colour on cortex"""
    parc = dk.read(f"Subject{sid}")[0]
    n2i = dk.name2id()
    rgb = np.tile(np.array([int(FLESH[i:i + 2], 16) for i in (1, 3, 5)], np.uint8), (len(parc), 1))
    for _, col, names in LOBES:
        ids = [n2i[n] for n in names if n in n2i]
        c = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        rgb[np.isin(parc, ids)] = c
    return rgb


def scale(aspect):
    """One parallel scale for the whole plate.

    parallel_scale is the viewport HALF-HEIGHT in world units, so it has to be derived from the
    brain's extent along the screen axes, not from its bounding radius - the radius is set by the
    longest anatomical axis and leaves the view two-thirds empty. Screen up is `up`, screen right
    is up x view; take whichever of the two needs more room once the window aspect is applied."""
    need = []
    for sid in SUBJ:
        s_ = f"Subject{sid}"
        mgh = nib.load(f"{FS}/{s_}/mri/wm.mgz")
        tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
        v = nib.affines.apply_affine(tk, nib.freesurfer.read_geometry(
            f"{FS}/{s_}/surf/{H}.{SURF}")[0])
        view, up = camera.frame(v, dk.read(s_)[0])
        d = v - v.mean(0)
        right = np.cross(up, view); right /= np.linalg.norm(right)
        need.append(max(np.abs(d @ up).max(), np.abs(d @ right).max() / aspect))
    return float(max(need)) * 1.06


def render():
    import pyvista as pv
    os.makedirs(CACHE, exist_ok=True)
    WIN = (1100, 820)
    ps = scale(WIN[0] / WIN[1])
    print(f"shared parallel scale {ps:.1f} mm  (window aspect {WIN[0] / WIN[1]:.2f})")
    for sid in SUBJ:
        s = f"Subject{sid}"
        mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
        tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
        v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.{SURF}")
        vw = nib.affines.apply_affine(tk, v)
        mesh = pv.PolyData(vw, np.c_[np.full(len(f), 3), f].ravel())
        mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
        mesh.point_data["rgb"] = lobe_rgb(sid)
        parc = dk.read(s)[0]
        hv = None
        if SHOW_3HG:
            # shared loader: the mesh here is already in world coordinates, so rotating its
            # normals by tk again - as an earlier version did - gives the wrong patch centres
            # and a different clustering (188 nodes instead of 168 for Subject1).
            from hinge_ids import load_3hg
            hv = load_3hg(sid, merge=MERGE_MM, hemi=H, verbose=True)
        view, up = camera.frame(vw, parc)
        for tag in VIEWS:
            direction = -view if tag == "med" else view
            pl = pv.Plotter(off_screen=True, window_size=WIN, border=False,
                            lighting="light kit")
            pl.background_color = "white"; pl.enable_anti_aliasing("ssaa")
            if tag == "lobe":
                pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True,
                            specular=0.12, specular_power=16)
            else:
                pl.add_mesh(mesh, color=GROUND if SHOW_3HG else FLESH,
                            smooth_shading=True, specular=0.14, specular_power=18)
            if SHOW_SKEL:
                e = skeleton_lines(sid, H)
                # Mask the medial wall on the skeleton too, not just on the landmarks. GyralNet
                # traces a network there because the closed medial face has ridges, but there is
                # no folding to be a gyral crest, and drawing it advertises structure the
                # analysis excludes. An edge survives only if BOTH endpoints are cortical.
                keep_e = (parc[e[:, 0]] > 0) & (parc[e[:, 1]] > 0)
                if tag == VIEWS[0]:
                    print(f"    skeleton: {len(e)} edges -> {int(keep_e.sum())} cortical",
                          flush=True)
                e = e[keep_e]
                nrm = mesh.point_data["Normals"]
                pts = vw + SKEL_LIFT * nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
                if SKEL_SMOOTH:
                    from scipy import sparse as _sp
                    n = len(pts)
                    A = _sp.csr_matrix((np.ones(2 * len(e)),
                                        (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                                       shape=(n, n))
                    deg_ = np.asarray(A.sum(1)).ravel()
                    on = deg_ > 0
                    inv = np.zeros(n); inv[on] = 1.0 / deg_[on]
                    for _ in range(SKEL_SMOOTH):
                        nb = A @ pts
                        pts[on] = 0.5 * pts[on] + 0.5 * (nb[on] * inv[on, None])
                keep = np.unique(e)
                remap = -np.ones(len(pts), int); remap[keep] = np.arange(len(keep))
                sk = pv.PolyData()
                sk.points = pts[keep]
                sk.lines = np.c_[np.full(len(e), 2), remap[e[:, 0]], remap[e[:, 1]]].ravel()
                pl.add_mesh(sk.tube(radius=SKEL_R, n_sides=6), color=SKEL_COL,
                            smooth_shading=True, specular=0.05)
            if hv is not None and len(hv):
                # spheres in world units, placed on the surface being rendered, so they follow
                # whichever geometry is shown rather than being pinned to the white one
                ball = pv.Sphere(radius=HINGE_R, theta_resolution=14, phi_resolution=14)
                pl.add_mesh(pv.PolyData(vw[hv]).glyph(geom=ball, scale=False, orient=False),
                            color=ns.C_3HG, smooth_shading=True,
                            specular=0.35, specular_power=20)
            pl.camera.enable_parallel_projection()
            pl.camera.position = tuple(mesh.center + direction * 400)
            pl.camera.focal_point = tuple(mesh.center)
            pl.camera.up = tuple(up)
            pl.camera.parallel_scale = ps
            pl.screenshot(f"{CACHE}/S{sid}_{SURF}{('_3hg%g' % MERGE_MM) if SHOW_3HG else ''}{'_sk' if SHOW_SKEL else ''}_{tag}.png", transparent_background=True)
        print(f"  Subject{sid}", flush=True)


def plane_axes(sid):
    """(centre, and the two in-plane unit vectors of each anatomical section)"""
    s_ = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s_}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v = nib.affines.apply_affine(tk, nib.freesurfer.read_geometry(f"{FS}/{s_}/surf/{H}.white")[0])
    lat, up = camera.frame(v, dk.read(s_)[0])
    ant = np.cross(up, lat)               # camera.py builds lat = ant x up, so this inverts it
    return v.mean(0), v, (ant, up, lat)          # in-plane horizontal, vertical, and the normal


def fovs():
    """One field of view, shared by all specimens and all three levels, so the sections are
    comparable down and across the plate."""
    w = h = 0.0
    for sid in SUBJ:
        c, v, (e0, e1, _) = plane_axes(sid)
        d = v - c
        w = max(w, np.abs(d @ e0).max()); h = max(h, np.abs(d @ e1).max())
    return 2.06 * w, 2.06 * h


def sections(sid, FOV):
    """the sagittal series, each level from its own source volume"""
    c, v, (e0, e1, nrm) = plane_axes(sid)
    proj = (v - c) @ nrm
    lo, hi = float(proj.min()), float(proj.max())        # medial face to lateral convexity
    fw, fh = FOV
    nx = NPIX
    ny = max(int(round(NPIX * fh / fw)), 8)
    gx = (np.arange(nx) - (nx - 1) / 2) * (fw / nx)
    gy = (np.arange(ny) - (ny - 1) / 2) * (fh / ny)
    A, B = np.meshgrid(gx, gy, indexing="xy")
    out = {}
    for tag, src in SLICES:
        img = nib.load(SRC[src].format(r=ROOT, f=FS, i=sid))
        vol, inv = np.asarray(img.dataobj, np.float32), np.linalg.inv(img.affine)
        frac = LEVEL_BY_SUBJ.get(sid, LEVEL)
        xyz = (c + (lo + frac * (hi - lo)) * nrm) + A[..., None] * e0 + B[..., None] * (-e1)
        ijk = nib.affines.apply_affine(inv, xyz.reshape(-1, 3)).T
        sl = ndi.map_coordinates(vol, ijk, order=1, mode="constant", cval=0).reshape(ny, nx)
        nz = sl[sl > 0]
        v0, v1 = (np.percentile(nz, [1, 99.5]) if nz.size > 100 else (0.0, 1.0))
        out[tag] = np.clip((sl - v0) / max(v1 - v0, 1e-6), 0, 1)
    return out


_BOX = None


def surf_box():
    """One crop box for every surface render. Cropping each to its own alpha would rescale the
    specimens independently and destroy the true-relative-size property; not cropping at all
    leaves the padding that the largest specimen needs around every smaller one."""
    global _BOX
    if _BOX is None:
        bb = [Image.open(f"{CACHE}/S{sid}_{SURF}{('_3hg%g' % MERGE_MM) if SHOW_3HG else ''}{'_sk' if SHOW_SKEL else ''}_{t}.png").getchannel("A").getbbox()
              for sid in SUBJ for t in VIEWS]
        _BOX = (min(b[0] for b in bb), min(b[1] for b in bb),
                max(b[2] for b in bb), max(b[3] for b in bb))
    return _BOX


def load(sid, tag):
    im = Image.open(f"{CACHE}/S{sid}_{SURF}{('_3hg%g' % MERGE_MM) if SHOW_3HG else ''}{'_sk' if SHOW_SKEL else ''}_{tag}.png")
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3])
    return np.asarray(bg.crop(surf_box()))


def block_geom(Wmm):
    """Every millimetre of the cohort block, computed before any canvas exists.

    The heights here are all fixed by image aspects, so the figure height is an OUTPUT of the
    layout. Returning it separately lets this block be dropped into a larger plate that has to
    know how tall it will be before it can create its canvas."""
    FOV = fovs()
    box = surf_box()
    surf_ar = (box[3] - box[1]) / (box[2] - box[0])
    ps = scale(1100 / 820)
    surf_mm_w = (box[2] - box[0]) * (2 * ps / 820)
    GX, GAP, TITLE, VGAP, ROWGAP = (SP[k] for k in ("GX", "GAP", "TITLE", "VGAP", "ROWGAP"))
    PWmm = (Wmm - 2 * GX) / 3
    # one scale for both rows, set by whichever row is the tighter fit
    ncell = max(len(VIEWS), len(SLICES))
    avail = (PWmm - (ncell - 1) * GAP) / ncell
    fpm = min(avail / surf_mm_w, avail / FOV[0])
    wsurf, hsurf = surf_mm_w * fpm, surf_mm_w * surf_ar * fpm
    wsec, hsec = FOV[0] * fpm, FOV[1] * fpm
    panel = TITLE + hsec + VGAP + hsurf
    return dict(FOV=FOV, PW=PWmm, GX=GX, GAP=GAP, TITLE=TITLE, VGAP=VGAP, ROWGAP=ROWGAP,
                wsurf=wsurf, hsurf=hsurf, wsec=wsec, hsec=hsec, panel=panel,
                total=2 * panel + ROWGAP)


def draw_block(fig, x0, ytop, Wmm, FWmm, FHmm):
    """Draw the six-specimen block with its top edge at `ytop` mm from the canvas bottom."""
    g = block_geom(Wmm)
    fx, fy = lambda v: v / FWmm, lambda v: v / FHmm
    for k, sid in enumerate(SUBJ):
        r, c = divmod(k, 3)
        px = x0 + c * (g["PW"] + g["GX"])
        ptop = ytop - r * (g["panel"] + g["ROWGAP"])
        fig.text(fx(px + g["PW"] / 2), fy(ptop - g["TITLE"] + 1.2), f"Subject {sid}",
                 fontsize=SUBJ_FS, ha="center", va="bottom", color=ns.INK)
        sec = sections(sid, g["FOV"])
        xo = px + (g["PW"] - len(SLICES) * g["wsec"] - (len(SLICES) - 1) * g["GAP"]) / 2
        for tag, _ in SLICES:
            ax = fig.add_axes([fx(xo), fy(ptop - g["TITLE"] - g["hsec"]),
                               fx(g["wsec"]), fy(g["hsec"])])
            ax.imshow(sec[tag], cmap="gray", vmin=0, vmax=1, interpolation="bilinear",
                      aspect="auto")
            ax.set_axis_off()
            xo += g["wsec"] + g["GAP"]
        ysurf = ptop - g["TITLE"] - g["hsec"] - g["VGAP"] - g["hsurf"]
        xs = px + (g["PW"] - len(VIEWS) * g["wsurf"] - (len(VIEWS) - 1) * g["GAP"]) / 2
        for j, tag in enumerate(VIEWS):
            ax = fig.add_axes([fx(xs + j * (g["wsurf"] + g["GAP"])), fy(ysurf),
                               fx(g["wsurf"]), fy(g["hsurf"])])
            ax.imshow(load(sid, tag), interpolation="lanczos"); ax.set_axis_off()
    return g["total"]


def compose():
    ns.nature()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONTS})
    MARG = 2.0
    FW = ns.W2 / ns.MM
    FH = block_geom(FW - 2 * MARG)["total"] + 1.5
    fig = plt.figure(figsize=(FW * ns.MM, FH * ns.MM), facecolor="white")
    draw_block(fig, MARG, FH, FW - 2 * MARG, FW, FH)
    for ext, dpi in (("png", 800), ("pdf", None)):
        fig.savefig(f"{OUT}/{NAME}.{ext}", dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    w, h = Image.open(f"{OUT}/{NAME}.png").size
    print(f"wrote {NAME}.png  {w / 800 * 25.4:.0f} x {h / 800 * 25.4:.0f} mm")


if __name__ == "__main__":
    if "--compose-only" not in sys.argv:
        render()
    compose()

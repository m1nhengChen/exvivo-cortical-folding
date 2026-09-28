#!/usr/bin/env python3
"""Replacement FIG2: fiber converges in the white matter beneath three-hinge gyri.
Depth profile of the DBSI anisotropic-fiber fraction from the cortex, through the white
surface, into the superficial white matter, beneath 3HG vs beneath matched gyral crowns.
Medial-wall 3HG excluded. Specimen is the unit."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from scipy import sparse, ndimage as ndi
from scipy.sparse.csgraph import dijkstra
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import dk
from hinge_ids import cluster_groups, group_mean, mirror_groups, DISK, DISK
import natstyle as ns
ns.use()
ROOT = (_env.ROOT + "/28211843"); FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet"); OUT = (_env.ROOT + "/surf/three_hinge/results")
H = "rh"; SUBJ = [1, 2, 3, 4, 5, 6]
DEPTHS = np.arange(-2.0, 5.01, 0.5)          # mm from white surface; <0 cortex, >0 white matter
SUBWM = (DEPTHS >= 0.5) & (DEPTHS <= 3.0)


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


def sample(img, xyz):
    ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz).T
    return ndi.map_coordinates(np.asarray(img.dataobj, np.float32), ijk, order=1, mode="constant", cval=np.nan)


def profiles(sid):
    s = f"Subject{sid}"
    mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
    tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    import pyvista as pv
    m = pv.PolyData(v.astype(float), np.c_[np.full(len(f), 3), f].ravel())
    m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    nrm = (tk[:3, :3] @ m.point_data["Normals"].T).T; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    vw = nib.affines.apply_affine(tk, v)
    if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm                                   # outward (toward pial)
    img = nib.load(f"{ROOT}/{s}/{s}_DBSI/fiber_ratio_map.nii.gz")
    prof = np.stack([sample(img, vw - d * nrm) for d in DEPTHS])   # d>0 -> into WM

    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc"); parc = dk.read(s)[0]; ctx = parc > 0
    hinge = np.atleast_1d(np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int))
    hinge = hinge[parc[hinge] > 0]                   # drop medial-wall 3HG
    dmin = dijkstra(mesh_graph(v.astype(float), f), indices=hinge.tolist(), min_only=True)
    lo = np.percentile(sulc[ctx], 25)
    pool = np.where((sulc <= lo) & (dmin > 8) & ctx)[0]
    # random, not index-even: see hinge_profile.py
    crown = np.sort(np.random.RandomState(sid).choice(pool, min(len(hinge), len(pool)),
                                                      replace=False))
    # Per gyral junction, not per landmark: landmarks closer than the estimator's non-overlap
    # distance are one junction sampled repeatedly. DISK, not the 6 mm default, because this
    # samples along a normal ray from the same 2 mm-disk landmark set experiment 1 uses - the same
    # support, so the same scale. Controls grouped to match.
    grp = cluster_groups(sid, hinge, merge=DISK); sz = [len(g) for g in grp]
    cg = mirror_groups(len(crown), sz)
    hpatch = group_mean(np.nanmean(prof[SUBWM][:, hinge], 0), grp)
    cpatch = group_mean(np.nanmean(prof[SUBWM][:, crown], 0), cg)
    Yh = np.array([np.nanmean(group_mean(prof[d][hinge], grp)) for d in range(prof.shape[0])])
    Yc = np.array([np.nanmean(group_mean(prof[d][crown], cg)) for d in range(prof.shape[0])])
    return Yh, Yc, hpatch, cpatch


def main():
    from scipy import stats
    Hs, Cs, HP, CP = [], [], [], []
    for s in SUBJ:
        hm, cm, hp, cp = profiles(s)
        Hs.append(hm); Cs.append(cm); HP.append(hp[np.isfinite(hp)]); CP.append(cp[np.isfinite(cp)])
    Hs, Cs = np.array(Hs), np.array(Cs)
    dH = Hs[:, SUBWM].mean(1); dC = Cs[:, SUBWM].mean(1)
    p = stats.wilcoxon(dH - dC).pvalue
    # z-score superficial-WM fiber within specimen for a group-comparison distribution
    HZ, CZ, hmz, cmz = [], [], [], []
    for i in range(6):
        both = np.concatenate([HP[i], CP[i]]); mu, sd = both.mean(), both.std() or 1
        HZ.append((HP[i]-mu)/sd); CZ.append((CP[i]-mu)/sd)
        hmz.append((HP[i].mean()-mu)/sd); cmz.append((CP[i].mean()-mu)/sd)
    allH, allC = np.concatenate(HZ), np.concatenate(CZ)

    fig = plt.figure(figsize=(9.6, 4.2), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.42, left=0.09, right=0.97, top=0.9, bottom=0.16)

    ax = fig.add_subplot(gs[0])
    ax.axvspan(0, DEPTHS.max(), color="#f0eeea", zorder=0)          # white-matter band
    ax.axvline(0, color=ns.INK, lw=0.8)
    for M, col, lab in [(Cs, ns.C_CROWN, "gyral crown"), (Hs, ns.C_3HG, "3HG")]:
        mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(6)
        ax.plot(DEPTHS, mu, color=col, lw=2.4, label=lab, solid_capstyle="round", zorder=3)
        ax.fill_between(DEPTHS, mu - se, mu + se, color=col, alpha=.18, lw=0, zorder=2)
    ax.set_xlabel("depth from white surface  (mm)")
    ax.set_ylabel("DBSI fiber fraction")
    ax.legend(loc="lower right", handlelength=1.2)
    ns.panel(ax, "a", x=-0.16, y=1.04)

    # difference curve: 3HG - crown fiber fraction across depth (mean +/- SEM over 6 specimens)
    axb = fig.add_subplot(gs[1])
    diff = Hs - Cs; mu = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(6)
    axb.axvspan(0, DEPTHS.max(), color="#f0eeea", zorder=0)
    axb.axhline(0, color=ns.INK, lw=0.8, zorder=1)
    axb.fill_between(DEPTHS, 0, mu, where=(mu > 0), color=ns.C_3HG, alpha=.14, zorder=1)
    axb.fill_between(DEPTHS, mu - se, mu + se, color=ns.C_3HG, alpha=.22, lw=0, zorder=2)
    axb.plot(DEPTHS, mu, color=ns.C_3HG, lw=2.4, solid_capstyle="round", zorder=3)
    axb.axvline(0, color=ns.INK, lw=0.8, zorder=1)
    axb.set_xlabel("depth from white surface  (mm)")
    axb.set_ylabel("3HG − crown  DBSI fiber fraction")
    axb.text(0.97, 0.10, f"P = {p:.3f}", transform=axb.transAxes, ha="right", va="bottom",
             fontsize=9, color=ns.INK)
    ns.panel(axb, "b", x=-0.22, y=1.04)
    for a in (ax, axb):
        for sp in ("top", "right"): a.spines[sp].set_visible(False)

    fig.savefig(f"{OUT}/FIG2_fiberconv.png", dpi=340); fig.savefig(f"{OUT}/FIG2_fiberconv.pdf")
    from PIL import Image
    print("wrote FIG2_fiberconv.png", Image.open(f"{OUT}/FIG2_fiberconv.png").size,
          "| sub-WM meandiff=%.3f p=%.3f" % ((dH - dC).mean(), p))


if __name__ == "__main__":
    main()

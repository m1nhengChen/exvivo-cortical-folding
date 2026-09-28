#!/usr/bin/env python3
"""Regional (DK) breakdown of the 3HG signature. For each DK region we compare 3HG to
ordinary gyral crowns WITHIN THE SAME REGION (so region is matched, folding is not touched),
per specimen, standardized by that specimen's cortical SD. Cells = mean across specimens
with >=MINN 3HG in the region; a dot marks features consistent in sign across those specimens."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, numpy as np, warnings, sys; warnings.filterwarnings("ignore")
import nibabel as nib
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from nilearn import plotting
from hinge_ids import cluster_groups, group_mean, mirror_groups, DISK
import dk, natstyle as ns
ns.use()

FS = (_env.ROOT + "/surf/fs"); GNET = (_env.ROOT + "/surf/three_hinge/gnet")
OUT = (_env.ROOT + "/surf/three_hinge/results"); H = "rh"; MID = [1, 2, 3]
FSA = os.environ["FREESURFER_HOME"] + "/subjects/fsaverage"
SUBJ = [1, 2, 3, 4, 5, 6]; MINN = 5
# One representative per physical family, with the DBSI compartment set kept complete. The rule
# is fixed in advance and does not look at the outcome: selecting by significance would be
# circular (the test runs on these same six specimens), would drop cortical thickness - the
# largest effect of all, |d| = 0.26, held out only by P = 0.060 at n = 6 - and would keep MD, AD
# and RD side by side although they correlate at r = 0.98-1.00 across specimens. AD and RD are
# therefore represented by MD; all nine metrics remain visible in the effect-size panel.
METRICS = [("thickness", "Cortical thickness"), ("FA", "FA"), ("ADC", "MD"),
           ("fiber_ratio", "DBSI fiber"), ("restricted_ratio", "DBSI restricted"),
           ("hindered_ratio", "DBSI hindered"), ("water_ratio", "DBSI free water")]
LOBE = {
    "F": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "precentral",
          "paracentral", "lateralorbitofrontal", "medialorbitofrontal", "parstriangularis",
          "parsopercularis", "parsorbitalis"],
    "P": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "T": ["superiortemporal", "middletemporal", "inferiortemporal", "fusiform", "entorhinal"],
    "O": ["lateraloccipital", "lingual", "cuneus"],
    "C": ["posteriorcingulate", "isthmuscingulate"]}


def clean_mid(s, name):
    return nib.load(f"{FS}/{s}/surf_maps/{H}.{name}.profile.mgh").get_fdata()[:, 0, :][:, MID].mean(1)


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


def collect():
    """Per region, per specimen: the standardized 3HG-crown diff for each metric plus the
    3HG / crown vertex counts, so the heatmap can apply a strict cutoff while the brain map
    paints every region that carries any 3HG."""
    i2n = dk.id2name()
    raw = {}   # region_name -> list of per-specimen dicts {nh, ncr, d:{metric:diff}}
    for sid in SUBJ:
        s = f"Subject{sid}"
        v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white"); v = v.astype(float)
        sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
        parc, _, _ = dk.read(s)
        thick = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.thickness")
        M = {"thickness": thick}
        for key, _ in METRICS:
            if key != "thickness":
                M[key] = clean_mid(s, key)
        hinge = np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int)
        hinge = hinge[parc[hinge] > 0]               # drop medial-wall 3HG before they seed dmin
        dmin = dijkstra(mesh_graph(v, f), indices=hinge.tolist(), min_only=True)
        # crown = the gyral-crest peaks (most-gyral quartile), SAME definition as Fig 1,
        # just restricted to each region -- otherwise the contrast sign disagrees with the
        # main figure because 'all gyral' includes gyral banks
        lo = np.percentile(sulc, 25)
        crown_all = (sulc <= lo) & (dmin > 6) & (parc > 0)
        for pid in np.unique(parc[parc > 0]):        # every cortical DK region, not just LOBE
            nm = i2n.get(pid)
            if nm in (None, "unknown", "corpuscallosum"):
                continue
            hin = hinge[parc[hinge] == pid]
            cro = np.where(crown_all & (parc == pid))[0]
            if len(hin) < 1 or len(cro) < 1:
                continue
            # per junction, not per landmark (see hinge_ids.py)
            g = cluster_groups(sid, hin)
            cgg = mirror_groups(len(cro), [len(x) for x in g])
            d = {}
            for key, _ in METRICS:
                sd = np.nanstd(M[key][parc == pid]) or 1
                d[key] = (np.nanmean(group_mean(M[key][hin], g)) -
                          np.nanmean(group_mean(M[key][cro], cgg))) / sd
            raw.setdefault(nm, []).append({"nh": len(hin), "ncr": len(cro), "d": d})
    return raw


def main():
    raw = collect()
    keys = [k for k, _ in METRICS]
    # heatmap (panel a): strict -- specimens with >=MINN 3HG AND >=MINN crown, region kept
    # if seen in >=4 such specimens (rigorous per-specimen statistics)
    per = {}
    for nm, recs in raw.items():
        good = [r for r in recs if r["nh"] >= MINN and r["ncr"] >= MINN]
        if len(good) >= 4:
            per[nm] = {k: np.array([r["d"][k] for r in good]) for k in keys}
    ordered, lobe_of = [], {}
    for lobe, names in LOBE.items():
        for nm in names:
            if nm in per:
                ordered.append(nm); lobe_of[nm] = lobe
    Z = np.full((len(ordered), len(keys)), np.nan)
    CONS = np.zeros_like(Z, bool)
    for i, nm in enumerate(ordered):
        for j, k in enumerate(keys):
            vals = per[nm][k]
            Z[i, j] = np.mean(vals)
            CONS[i, j] = np.all(vals > 0) or np.all(vals < 0)

    nr, nc = len(ordered), len(keys)
    vmax = np.nanpercentile(np.abs(Z), 96)

    # ---- fsaverage per-region vertex maps, one per metric ----
    coords, faces = nib.freesurfer.read_geometry(f"{FSA}/surf/{H}.inflated")
    sulc = nib.freesurfer.read_morph_data(f"{FSA}/surf/{H}.sulc")
    alab, _, anames = nib.freesurfer.read_annot(f"{FSA}/label/{H}.aparc.annot")
    anames = [n.decode() if isinstance(n, bytes) else n for n in anames]
    # brain maps (panel b) paint EVERY DK region that carries any 3HG, averaging over all
    # specimens with >=1 hinge, so only the true medial wall stays grey. Small regions rest
    # on few 3HG -- this panel is the spatial overview; panel a holds the rigorous statistics.
    allreg = {nm: {k: np.mean([r["d"][k] for r in recs]) for k in keys}
              for nm, recs in raw.items()}
    verts = {}                                   # metric key -> per-vertex value on fsaverage
    for k in keys:
        vv = np.full(len(coords), np.nan)        # NaN = grey, kept only for the medial wall
        for jj, nm in enumerate(anames):
            if nm in ("unknown", "corpuscallosum"):
                continue                         # medial wall stays grey
            vv[alab == jj] = allreg.get(nm, {}).get(k, 0.0)   # no 3HG -> neutral 0, not grey
        verts[k] = vv

    FW, FH = 8.6, 8.8
    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    ax = fig.add_axes([0.19, 0.065, 0.205, 0.80]); ax.set_anchor("NW")
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    # pcolormesh with white gaps for the clean modern heatmap look; row 0 at top
    im = ax.pcolormesh(np.ma.masked_invalid(Z), cmap=ns.DIV, norm=norm,
                       edgecolors="white", linewidth=2.2)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlim(0, nc); ax.set_ylim(nr, 0)
    ax.set_xticks(np.arange(nc) + 0.5)
    ax.set_xticklabels([lab for _, lab in METRICS], rotation=38, ha="left", fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(nr) + 0.5)
    ax.set_yticklabels([ns.cap(nm.replace("middle", "mid").replace("frontal", "front.")
                        .replace("parietal", "par.").replace("temporal", "temp.")
                        .replace("occipital", "occ.").replace("orbito", "orb."))
                        for nm in ordered], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # lobe brackets + labels on the RIGHT (clear of the region names), white lobe separators
    prev = None; start = 0
    LOBE_NAME = {"F": "Frontal", "P": "Parietal", "T": "Temporal", "O": "Occipital",
                 "C": "Cingulate"}
    bounds = []
    for i, nm in enumerate(ordered + [None]):
        lb = lobe_of.get(nm) if nm else None
        if lb != prev and prev is not None:
            ax.plot([nc + 0.12, nc + 0.12], [start + 0.12, i - 0.12], lw=1.6,
                    color=ns.INK, clip_on=False)
            ax.text(nc + 0.30, (start + i) / 2, LOBE_NAME[prev], rotation=270, va="center",
                    ha="center", fontsize=9, fontweight="bold")
            bounds.append(i); start = i
        if lb != prev:
            start = i; prev = lb
    for b in bounds[:-1] if bounds else []:
        ax.plot([0, nc], [b, b], color="white", lw=5.0, clip_on=False, zorder=5)
    cax = fig.add_axes([0.19, 0.035, 0.205, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("3HG − crown, within region  (SD)", fontsize=8, labelpad=3)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=2, width=0.6)

    # ---- right column: each surface rendered separately, whitespace CROPPED away, then
    # placed edge-to-edge so the panel is tight left-to-right (nilearn 3-D axes otherwise
    # carry large internal margins that no amount of gridspec spacing removes) ----
    import io
    from PIL import Image

    def render(vert, view):
        ff = plt.figure(figsize=(3, 3)); aa = ff.add_subplot(111, projection="3d")
        plotting.plot_surf_stat_map((coords, faces), vert, hemi="right", view=view, bg_map=sulc,
                                    cmap=ns.DIV, vmin=-vmax, vmax=vmax, symmetric_cbar=True,
                                    colorbar=False, axes=aa, figure=ff)
        aa.set_box_aspect((1, 1, 0.8), zoom=1.5)
        buf = io.BytesIO(); ff.savefig(buf, format="png", transparent=True, dpi=150,
                                       bbox_inches="tight", pad_inches=0)
        plt.close(ff); buf.seek(0)
        im = Image.open(buf); return np.asarray(im.crop(im.getbbox()))

    top, bot = 0.895, 0.055
    rh = (top - bot) / nc                          # row pitch
    img_h = rh * 0.76                              # brain height in figure fraction
    xb0, gap = 0.435, 0.004                        # panel-b start (just past lobe labels) & gap
    for j, (k, lab) in enumerate(METRICS):
        yc = top - (j + 0.5) * rh
        x = xb0
        for view in ("lateral", "medial"):
            arr = render(verts[k], view); ih, iw = arr.shape[:2]
            w = img_h * (iw / ih) * (FH / FW)      # preserve pixel aspect in figure coords
            axb = fig.add_axes([x, yc - img_h / 2, w, img_h])
            axb.imshow(arr); axb.axis("off"); axb.set_rasterized(True)
            x += w + gap
        fig.text((xb0 + x - gap) / 2, yc - img_h / 2 - 0.004, lab, ha="center", va="top",
                 fontsize=9.5, fontweight="bold")

    fig.savefig(f"{OUT}/FIG3_regional.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{OUT}/FIG3_regional.pdf", bbox_inches="tight", pad_inches=0.03)
    print("wrote FIG3_regional.png", Image.open(f"{OUT}/FIG3_regional.png").size, "|", len(ordered), "regions")


if __name__ == "__main__":
    main()

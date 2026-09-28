#!/usr/bin/env python3
"""Figure 4 - in-vivo validation and folding convergence.

 a  the 3HG - crown Cohen's d per living subject (KDE over 414 subjects) for the four tensor
    metrics, with the six ex-vivo specimens overlaid as ink dots on the same axis: the two
    experiments answer the same question with the same estimator, so they share one scale.
 b  the folding ladder in vivo: sulc-matched rung means (z within subject), mean +- s.e.m.
    over 414 subjects.
 c  the same ladder in the six ex-vivo hemispheres (tensor metrics), same colours and scale -
    the two panels differ in exactly one honest place: in vivo the crest -> junction step is
    emphatic but the 3HG -> 4HG+ rung does not continue the gradient (MD/RD flat, AD partial
    reversal); ex vivo, at 0.8 mm, the gradient runs through. That contrast is the figure's
    point, not something to hide: the continuum is what the resolution buys.
 d  the ex-vivo ladder for the full diffusion set (MD + the DBSI compartments) - the original
    discovery panel, moved here from the connectivity figure so that figure is pure wiring and
    this one is the complete folding-convergence story.
Data: invivo/invivo_pipeline.py, invivo/invivo_ladder.py -> derived/exp{1,2}_summary.npz;
      hinge_order.py -> results/hinge_order.npz; exvivo_parcel_crown.py (ex-vivo d,
      parcel-matched crown - the in-vivo control design).
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import natstyle as ns

RES = (_env.ROOT + "/surf/three_hinge/results")
DER = (_env.AABC + "/derived")
EXFA = (_env.ROOT + "/28211843/Subject6/Subject6_DTI/FA.nii.gz")
IVFA = f"{DER}/{_env.aabc_example()}/FA.nii.gz"
ROOTS = (_env.AABC + "/organized")
DPI = 1600
# one colour per metric across all three panels: falling metrics in the crown blues,
# the rising one in the 3HG orange - the same logic as the ex-vivo ladder panel
MCOL = {"MD": ns.BLUE, "AD": "#6E8DB6", "RD": ns.BLUE_L, "FA": ns.C_3HG}
METS = ["FA", "MD", "AD", "RD"]
RUNGS = ["crest", "3HG", "4HG+"]
EXV_KEY = {"MD": "ADC", "AD": "AD", "RD": "RD", "FA": "FA"}   # npz naming


def pfmt(p):
    """exact P, mantissa x 10^e below 0.001; the tiny ones are the result, not clutter"""
    if p >= 0.001:
        return f"$P$ = {p:.3f}"
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    if round(m) == 10:
        m, e = 1, e + 1
    return f"$P$ = {m:.0f}$\\times$10$^{{{e}}}$"


def panel_fa(fig, FH_MM):
    """a: the four tensor maps at both resolutions, on ANATOMICAL sagittal planes.

    The ex-vivo specimens were scanned in arbitrary poses - the volume axes are not anatomical
    axes - so the ex-vivo slice is sampled on the parcel-centroid-derived plane machinery of
    render_voxels (superior and anterior from centroid groups, the cut swept along their cross
    product). The in-vivo volume is sliced in canonical RAS with cerebellum and brainstem
    removed via wmparc, since the specimens have neither. Both views: anterior right, superior
    up, one slice per experiment shared by all four maps."""
    import nibabel as nib
    import scipy.ndimage as ndi
    import render_voxels as rv
    EXK = {"FA": "FA", "MD": "ADC", "AD": "AD", "RD": "RD"}
    EXD = os.path.dirname(EXFA)

    def clean(mk):
        L, n = ndi.label(mk)
        if n > 1:
            mk = L == (np.bincount(L.ravel())[1:].argmax() + 1)
        return mk

    def crop_all(imgs, mk):
        r, c = np.where(mk.any(1))[0], np.where(mk.any(0))[0]
        sl = (slice(max(r[0] - 2, 0), r[-1] + 3), slice(max(c[0] - 2, 0), c[-1] + 3))
        return {m: np.ma.masked_array(a[sl], ~mk[sl]) for m, a in imgs.items()}

    def polish(imgs, f):
        """display-only resampling: nearest-fill under the mask, cubic upsample, smooth
        silhouette. Shown at 1.5 mm the mask edge is a staircase and the panel reads as a
        matplotlib artefact rather than an image; values are not reinterpreted, only rendered."""
        mask = ~imgs["FA"].mask
        idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        m2 = ndi.zoom(ndi.gaussian_filter(mask.astype(float), 0.6), f, order=1) > 0.5
        return {m: np.ma.masked_array(ndi.zoom(a.data[tuple(idx)], f, order=3), ~m2)
                for m, a in imgs.items()}

    # ---- ex vivo: the anatomical plane ----
    sid = 6
    sup = rv.axis(sid, "sup", np.zeros(3)); ant = rv.axis(sid, "ant", np.zeros(3))
    ant = ant - (ant @ sup) * sup; ant /= np.linalg.norm(ant)
    nrm = np.cross(ant, sup); nrm /= np.linalg.norm(nrm)
    WALL, _ = rv.landmarks(sid, whole=True)
    c0 = WALL.mean(0)
    c = c0 + np.percentile((WALL - c0) @ nrm, 45) * nrm    # a mid-lateral cut
    p1, p2 = (WALL - c) @ ant, (WALL - c) @ sup
    t1 = np.arange(p1.min() - 3, p1.max() + 3, 0.45)
    t2 = np.arange(p2.min() - 3, p2.max() + 3, 0.45)
    U, V = np.meshgrid(t1, t2, indexing="xy")
    XYZ = c + U[..., None] * ant + V[..., None] * sup
    mk = rv.sample(nib.load(f"{rv.TR}/Subject{sid}/mask.nii.gz"), XYZ) > 0.85
    mk = ndi.binary_erosion(mk, iterations=2)
    exi = {m: rv.sample(nib.load(f"{EXD}/{EXK[m]}.nii.gz"), XYZ) for m in METS}
    mk = clean(mk & (exi["FA"] > 1e-9))                    # the DTI FOV is tighter than the mask
    ex = polish(crop_all({m: np.flipud(a) for m, a in exi.items()}, np.flipud(mk)), 2)

    # ---- in vivo: canonical RAS, cerebrum only ----
    ses = _env.aabc_example()
    ivv = {m: nib.as_closest_canonical(nib.load(f"{DER}/{ses}/{m}.nii.gz")) for m in METS}
    wp = nib.as_closest_canonical(nib.load(
        f"{ROOTS}/{ses}/sMRI/T1w/wmparc.nii.gz"))
    A = np.linalg.inv(wp.affine) @ ivv["FA"].affine
    sh = ivv["FA"].shape
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in sh], indexing="ij")
    src = (A @ np.stack([ii, jj, kk, np.ones_like(ii)]).reshape(4, -1))[:3].round().astype(int)
    ok = np.all((src >= 0) & (src < np.array(wp.shape)[:, None]), 0)
    lab = np.zeros(np.prod(sh)); W = wp.get_fdata()
    lab[ok] = W[src[0, ok], src[1, ok], src[2, ok]]
    cereb = np.isin(lab.reshape(sh), [7, 8, 46, 47, 15, 16])
    xi = int(sh[0] * 0.70)                                 # mid-lateral, matching the ex-vivo cut
    ivi = {m: np.rot90(v.get_fdata()[xi]) for m, v in ivv.items()}
    mkv = np.rot90(~cereb[xi]) & (ivi["FA"] > 1e-9)
    # two voxels off the rim: edge voxels carry partial-volume speckle (high FA at the mask
    # boundary, high diffusivity beside CSF) that upsampling would paint as a bright red
    # border. Largest component is kept AFTER the erosion - eroding first can detach the
    # temporal lobe and leave floating fragments.
    mkv = clean(ndi.binary_erosion(clean(mkv), iterations=2))
    iv = polish(crop_all(ivi, mkv), 4)

    cm = plt.get_cmap("turbo").copy(); cm.set_bad("white")
    X0, WC, GX = 0.048, 0.134, 0.009
    W_MM = ns.W2 / ns.MM
    ytop = 0.912
    for lab_, imgs in (("Ex vivo, 0.8 mm", ex), ("In vivo, 1.5 mm", iv)):
        hh = WC * (imgs["FA"].shape[0] / imgs["FA"].shape[1]) * W_MM / FH_MM
        # ONE window per row for the three diffusivities. Windowed independently, three highly
        # correlated maps are each stretched onto the full colour range and read as identical;
        # shared, the real ordering AD > MD > RD shows. In vivo the window is set on parenchyma
        # (85th percentile): CSF is ~3x tissue diffusivity, so a CSF-topped window compresses
        # the cortex-WM contrast into the bottom of the scale - CSF saturates instead, and the
        # cortical ribbon reads warm as it does ex vivo (GM > WM diffusivity in both).
        dpool = np.concatenate([imgs[m].compressed() for m in ("MD", "AD", "RD")])
        dwin = np.percentile(dpool, [2, 95])
        for j, m in enumerate(["FA", "MD", "AD", "RD"]):
            ax = fig.add_axes([X0 + j * (WC + GX), ytop - hh, WC, hh])
            img = imgs[m]
            if m == "FA":
                lo, hi = np.percentile(img.compressed(), [2, 98])
            elif lab_.startswith("Ex"):
                lo, hi = dwin              # shared: fixed tissue is near-isotropic, AD ~ RD
            else:
                # in vivo AD and RD differ threefold - a shared window saturates one and
                # buries the other; per-map parenchymal windows (85th pct keeps CSF out)
                # let MD and RD show the same warm cortical ribbon as ex vivo
                lo, hi = np.percentile(img.compressed(), [5, 85])
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="lanczos")
            ax.set_axis_off()
            if lab_.startswith("Ex"):
                ax.set_title(m, fontsize=7, pad=2.5)
        fig.text(X0 - 0.010, ytop - hh / 2, lab_, fontsize=6.2, rotation=90,
                 ha="center", va="center")
        ytop -= hh + 0.018


def panel_d(ax):
    z = np.load(f"{DER}/exp1_summary.npz")
    D = z["d"]                                              # subj x [FA MD AD RD]
    # SAME control design as the in-vivo pipeline: crown draws matched to the 3HG DK-parcel
    # histogram (exvivo_parcel_crown.py). The primary ex-vivo analysis (Fig. 2) keeps its
    # whole-hemisphere count-matched crown; overlaying the two experiments on one axis is only
    # honest if the estimator AND the control design agree, so this panel recomputes ex vivo
    # rather than mixing designs. Cost of the alignment: ex-vivo FA drops from +0.171
    # (P = 0.010) to +0.058 (P = 0.14, 5/6 positive) - reported as is.
    ze = np.load(f"{RES}/exvivo_parcel_crown_d.npz")
    exv = {m: ze["d"][:, list(ze["mets"]).index(m)] for m in METS}
    ax.axvline(0, color=ns.INK, lw=.6, zorder=3)
    from scipy.stats import gaussian_kde
    for i, m in enumerate(METS):
        y = len(METS) - 1 - i
        d = D[:, METS.index(m)]
        kd = gaussian_kde(d)
        xs = np.linspace(-0.75, 0.75, 240)
        dens = kd(xs); dens = dens / dens.max() * 0.38
        ax.fill_between(xs, y, y + dens, color=MCOL[m], alpha=.55, lw=0, zorder=2)
        ax.plot(xs, y + dens, color=MCOL[m], lw=.8, zorder=3)
        mu = d.mean()
        ax.plot([mu], [y], marker="o", ms=3.6, mfc="white", mec=MCOL[m], mew=.9,
                zorder=5, linestyle="none")
        # the six specimens, on the same axis: the point of the figure
        ax.scatter(exv[m], np.full(6, y - 0.17), s=7, color=ns.INK, zorder=4, lw=0)
        ax.plot([np.mean(exv[m])], [y - 0.17], marker="D", ms=2.6, color=ns.INK,
                zorder=5, linestyle="none")
        # both tests, right-aligned in the free right margin. P values are ink everywhere in
        # this figure (c-e set them in ink beside coloured metric names); which experiment each
        # belongs to is said by position - the upper one sits on the in-vivo ridge's baseline,
        # the lower one beside the six ex-vivo dots.
        ax.text(0.73, y + 0.14, pfmt(stats.ttest_1samp(d, 0).pvalue), ha="right", va="bottom",
                fontsize=5.4, color=ns.INK)
        ax.text(0.73, y - 0.30, pfmt(stats.ttest_1samp(exv[m], 0).pvalue), ha="right",
                va="bottom", fontsize=5.4, color=ns.INK)
    ax.set_yticks(range(len(METS)))
    ax.set_yticklabels(METS[::-1], fontsize=6.4)
    ax.set_ylim(-0.55, len(METS) - 0.25)
    ax.set_xlim(-0.75, 0.75); ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel("Cohen's d  (3HG − crown)", fontsize=7, labelpad=2)
    ax.tick_params(axis="y", length=0)
    # No legend. Its in-vivo handle was a grey square, which matches nothing on the panel -
    # the ridges are in the metric colours - and read as a puzzle. The two experiments are
    # named directly on the top row instead, each label beside the mark it names: "in vivo" at
    # the foot of the FA ridge, "ex vivo" beside the six FA dots. The legend text repeats it.
    # Both at the left edge of the top row, left-aligned on one x, each on the baseline of the
    # thing it names: "in vivo" on the ridge's baseline, "ex vivo" on the dot row. Floating them
    # next to the marks put the two words at different x on different lines and they read as
    # scattered, not as a key.
    yt = len(METS) - 1
    ax.text(-0.73, yt + 0.05, "in vivo", ha="left", va="bottom", fontsize=5.8, color=ns.INK)
    ax.text(-0.73, yt - 0.17, "ex vivo", ha="left", va="center", fontsize=5.8, color=ns.INK)


def ladder(ax, MU, SE, PV, letter, title, lx):
    x = np.arange(3)
    ax.axhline(0, color=ns.MUTE, lw=.5, ls=(0, (2.5, 2)), zorder=1)
    ends = []
    for m in METS:
        mu, se = MU[m], SE[m]
        ax.fill_between(x, mu - se, mu + se, color=MCOL[m], alpha=.15, lw=0, zorder=2)
        ax.plot(x, mu, color=MCOL[m], lw=1.1, marker="o", ms=2.4, mec="white", mew=.4,
                solid_capstyle="round", zorder=3)
        ends.append([mu[-1], m, MCOL[m]])
    GAP = 0.075
    ends.sort(key=lambda e: e[0])
    ypos = [e[0] for e in ends]
    for i in range(1, len(ypos)):
        ypos[i] = max(ypos[i], ypos[i - 1] + GAP)
    off = (np.mean(ypos) - np.mean([e[0] for e in ends]))
    for (v, lab, col), yy in zip(ends, np.array(ypos) - off):
        ax.annotate(lab, xy=(2.02, v), xytext=(2.22, yy), fontsize=6, color=col,
                    fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=.5, shrinkA=0, shrinkB=1.5))
    # The tested quantity is the end-to-end contrast z(4HG+) - z(crest) - NOT a fitted slope,
    # and in vivo the middle rung visibly deviates from a line, so the block must not say
    # 'linear trend'. Both panels are empty in the
    # upper-left corner, so the four P values sit there as a small block instead of hanging off
    # the end labels, where four two-line labels do not fit the y range.
    # rows end at y = 0.10: below that, the ex-vivo panel's ribbons pass through the left half
    ax.text(-0.13, 0.325, "4HG+ \u2212 crest", fontsize=5.4, color=ns.INK, va="top")
    for k, m in enumerate(METS):
        yy = 0.265 - k * 0.055
        ax.text(-0.13, yy, m, fontsize=5.6, color=MCOL[m], fontweight="bold", va="top")
        ax.text(0.36, yy, pfmt(PV[m]), fontsize=5.4, color=ns.INK, va="top")
    ax.set_xlim(-0.25, 2.75); ax.set_xticks(x)
    ax.set_xticklabels(["Crest", "3HG", "4HG+"], fontsize=6.4)
    ax.set_ylim(-0.30, 0.34); ax.set_yticks([-0.2, 0, 0.2])
    ax.set_ylabel("Microstructure (SD)", fontsize=7, labelpad=2)
    ax.set_xlabel("Folding convergence", fontsize=7, labelpad=2)
    ns.ptitle(ax, title, letter, lx=lx, y=1.14)


def main():
    ns.nature()
    FH_MM = 104
    fig = plt.figure(figsize=(ns.W2, FH_MM * ns.MM), facecolor="white")

    # ---- row 1: the measures, then the replication ----
    panel_fa(fig, FH_MM)
    axb = fig.add_axes([0.695, 0.575, 0.285, 0.340])
    panel_d(axb)
    # one letter line for the row, set by b's axes top
    yr1 = 0.575 + 0.340 * 1.14
    fig.text(0.020, yr1, "a", fontsize=8, fontweight="bold", va="bottom")
    fig.text(0.622, yr1, "b", fontsize=8, fontweight="bold", va="bottom")

    # ---- row 2: the three ladders ----
    z2 = np.load(f"{DER}/exp2_summary.npz")
    Z = z2["z"]                                             # subj x rung x [FA MD AD RD]
    MU = {m: Z[:, :, METS.index(m)].mean(0) for m in METS}
    SE = {m: Z[:, :, METS.index(m)].std(0, ddof=1) / np.sqrt(len(Z)) for m in METS}
    T = z2["trend"]
    PV = {m: stats.ttest_1samp(T[:, METS.index(m)], 0).pvalue for m in METS}
    RB, RH = 0.075, 0.345
    ladder(fig.add_axes([0.075, RB, 0.238, RH]), MU, SE, PV, "c",
           "In vivo", -0.26)

    ze = np.load(f"{RES}/hinge_order.npz")
    MUe = {m: np.array([ze[f"{EXV_KEY[m]}|{r}"].mean() for r in RUNGS]) for m in METS}
    SEe = {m: np.array([ze[f"{EXV_KEY[m]}|{r}"].std(ddof=1) / np.sqrt(6) for r in RUNGS])
           for m in METS}
    PVe = {m: stats.ttest_1samp(ze[f"{EXV_KEY[m]}|4HG+"] - ze[f"{EXV_KEY[m]}|crest"], 0).pvalue
           for m in METS}
    ladder(fig.add_axes([0.413, RB, 0.238, RH]), MUe, SEe, PVe, "d",
           "Ex vivo, tensor metrics", -0.26)
    import fig_hinge_order as ho
    axe = fig.add_axes([0.758, RB, 0.222, RH])
    ho.order_panel(axe, "")
    ns.ptitle(axe, "Ex vivo, MD and DBSI compartments", "e", lx=-0.26, y=1.14)

    for ext, dpi in (("png", DPI), ("pdf", None)):
        fig.savefig(f"{RES}/FIG4_invivo_folding.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    w, h = Image.open(f"{RES}/FIG4_invivo_folding.png").size
    print(f"wrote FIG4_invivo_folding.png  {w/DPI*25.4:.0f} x {h/DPI*25.4:.0f} mm")


if __name__ == "__main__":
    main()

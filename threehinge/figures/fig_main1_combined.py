#!/usr/bin/env python3
"""Combined main figure - the 3HG microstructural signature (FIG1), its depth resolution (FIG2)
and its regional breakdown (FIG3) as one Nature-spec figure. 183 mm double column.
 a  multimodal effect sizes, 3HG vs gyral crown          (left top)
 b  three-way comparison, 3HG / crown / fundus
 c  DBSI fiber fraction from cortex into superficial WM        (plate only)
 d  the 3HG - crown fiber difference across that depth        (plate only)
 c  the three sampling classes on one specimen's white surface  (signature figure only)
 e  region x metric heatmap of the within-parcel difference, with the same differences on the
    fsaverage surface alongside (one panel, right column)
 f  equivolumetric depth trajectories, with the tested within-parcel difference beneath
Data pipelines are imported unchanged from fig1_signature / fig_depth_combined / region_dk."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import io, sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.patheffects as pe
from PIL import Image
import nibabel as nib
import natstyle as ns
from fig1_signature import per_subject_d, zmeans, NICE, SUBJ
from fig_depth_combined import load_depth, ORDER, RES
import fig_fiberconv as fc
import render_classes as rc
import region_dk as rd
import dk
ns.nature()                                        # override the ns.use() the imports ran
OUT = RES; FW, FH = ns.W2, 185 * ns.MM
DPI = 1600                     # one raster resolution across the whole figure set


# Vertical remap, so the a-d block can be drawn on its own shorter canvas without every panel
# carrying two sets of coordinates. Identity for the full plate; for the a-d figure it rescales
# the band the block occupies to fill the page. Panel f is simply not called.
YLO, YHI = 0.0, 1.0


def _y(v):
    return (v - YLO) / (YHI - YLO)


def _ax(fig, rect):
    x, y, w, h = rect
    return fig.add_axes([x, _y(y), w, h / (YHI - YLO)])


def _txt(fig, x, y, *a, **k):
    return fig.text(x, _y(y), *a, **k)


def panel_a(fig, y0=0.635, letter="a", ty=1.06):
    mods, Dc = per_subject_d("crown")
    # Diffusion phenotype only. QSM (total / para / dia) and T2* are susceptibility and
    # relaxometry contrasts - a different physical story - and their per-specimen spread was
    # what the old (-1.35, 1.45) axis was sized for. They stay in the landmark source data.
    DROP = {"QSM", "QSM_pos", "QSM_neg", "T2star"}
    mods = [m for m in mods if m not in DROP]
    order = sorted(mods, key=lambda m: np.nanmean(Dc[m]))
    ax = _ax(fig, [0.115, y0, 0.19, 0.29])
    for i, m in enumerate(order):
        # sign, not class: ns.C_3HG/C_CROWN here would say 'crown' where the panel
        # means '3HG lower than crown'
        d = Dc[m]; mu = np.nanmean(d); col = ns.C_POS if mu > 0 else ns.C_NEG
        cons = np.all(d > 0) or np.all(d < 0)
        ax.plot([0, mu], [i, i], color=col, lw=.8, alpha=.5 if cons else .2, zorder=2)
        ax.scatter(d, np.full(6, i), s=5, color=col, alpha=.32 if cons else .16, zorder=3, lw=0)
        ax.scatter([mu], [i], s=19, color=col, zorder=4, edgecolor="white", linewidth=.4,
                   alpha=1 if cons else .4)
        # significance as stars, right margin: * P<0.05, ** P<0.01, *** P<0.001, blank = n.s.
        pv = stats.ttest_1samp(d[np.isfinite(d)], 0).pvalue
        st = "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < 0.05 else ""
        if st:
            ax.text(0.86, i, st, ha="right", va="center", fontsize=6.5, color=ns.INK)
    ax.axvline(0, color=ns.INK, lw=.5, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NICE.get(m, m) for m in order], fontsize=6.3)
    ax.set_ylim(-.6, len(order) - .4)
    ax.set_xlim(-0.55, 0.88); ax.set_xticks([-0.4, 0, 0.4, 0.8])
    ax.set_xlabel("Cohen's d  (3HG − crown)", fontsize=6.5, labelpad=2)
    ax.tick_params(axis="y", length=0)
    # ty: the two panels share a row, so their letters share one baseline
    _txt(fig, 0.012, y0 + 0.29 * ty, letter, fontsize=8, fontweight="bold", va="bottom")
    return ax


def panel_b(fig, y0=0.635, letter="b", ty=1.06):
    acc = zmeans()
    keys = [k for k, _ in rd.METRICS]      # same selection as the regional panel, same order
    ax = _ax(fig, [0.425, y0, 0.17, 0.29])
    y = np.arange(len(keys))[::-1]; h = .25
    for gi, (g, col, lab) in enumerate([("hinge", ns.C_3HG, "3HG"),
                                        ("crown", ns.C_CROWN, "Crown"),
                                        ("fundus", ns.C_FUND, "Fundus")]):
        m = [np.mean(acc[g][k]) for k in keys]
        se = [np.std(acc[g][k]) / np.sqrt(6) for k in keys]
        ax.barh(y + (gi - 1) * h, m, h * .9, color=col, alpha=.92, label=lab, zorder=2,
                xerr=se, error_kw=dict(lw=.4, ecolor="#666", capsize=.8))
    ax.axvline(0, color=ns.INK, lw=.5)
    ax.set_yticks(y); ax.set_yticklabels([NICE[k] for k in keys], fontsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Standardized mean (z)", fontsize=6.5, labelpad=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=.9,
              columnspacing=.9, labelspacing=.3, borderpad=.1, fontsize=5.8, frameon=False)
    ns.nletter(ax, letter, x=-0.52, y=ty)


def panel_match(fig, ytop=0.945, h=0.162):
    """The sampling rule, above the three classes it produces (both are panel a).

    Left, what a 3HG is: the GyralNet crest network around one junction, with the junction's own
    arms marked and each landmark drawn as the 2 mm geodesic disk the estimator averages over.
    Right, how the control is matched: the fold-depth distribution of the cortex with the three
    drawn classes on it. The crown draw is the one the analysis uses, not a cartoon of one, so
    the match is shown as evidence rather than asserted."""
    # One row, one band. All four components share the top edge at ytop and the row height ZH,
    # which the two magnified patches set (their aspect is fixed by the render); the histogram
    # and the key are fitted into that same band rather than each choosing its own height.
    X0, X1 = 0.030, 0.598          # stop clear of d's row labels
    GAPX, ZG = 0.020, 0.010
    ZH = 0.104                     # row height, raw; 9 key lines at 5.4 pt fit in it
    img = Image.open(f"{OUT}/_classes/S{rc.SID}_zoom.png")
    zw = ZH * (img.size[0] / img.size[1]) * (FH / FW)     # undistorted width for that height
    xh = X0 + 2 * zw + ZG + GAPX                          # histogram column starts here
    wk = 0.110                                            # key column
    wh = X1 - wk - GAPX - xh                              # histogram column
    y0 = ytop - ZH
    h = ZH

    # ---- the same patch of cortex on the 3HG and on the Crown surface ------------------------
    # Both are true magnifications of the boxed region of the surfaces below: one camera, one
    # field of view, each showing only what its own surface carries. Side by side they are the
    # matching rule made visible - one anatomy, two draws, the crown on the crests between the
    # junctions rather than at them.
    axZ = []
    for k, (name, lab, col) in enumerate((("zoom", "3HG", rc.PAINT["3HG"]),
                                          ("zoom_crown", "Crown", rc.PAINT["crown"]))):
        img = Image.open(f"{OUT}/_classes/S{rc.SID}_{name}.png")
        bg = Image.new("RGB", img.size, "white"); bg.paste(img, mask=img.split()[3])
        ax = _ax(fig, [X0 + k * (zw + ZG), y0, zw, ZH])
        ax.imshow(np.asarray(bg), interpolation="lanczos"); ax.set_axis_off()
        for sp in ax.spines.values():            # a frame, so each reads as a cut-out
            sp.set_visible(True); sp.set_linewidth(.6); sp.set_color(ns.INK)
        ax.set_frame_on(True)
        ax.text(0.5, -0.07, lab, transform=ax.transAxes, fontsize=6.4, color=col,
                fontweight="bold", ha="center", va="top")
        axZ.append(ax)

    # ---- fold depth: where each class is drawn from -------------------------------------------
    # Cohort level, not pooled: each hemisphere contributes one density curve and the panel shows
    # the mean over the six with its s.e.m., so the unit here is the specimen, as everywhere else
    # in the paper.
    z = np.load(f"{OUT}/_classes/match_sulc.npz")
    bins = z["bins"]; xc = (bins[:-1] + bins[1:]) / 2
    LBL = 0.024                    # raw height under the axis for the x label
    ax = _ax(fig, [xh, y0 + LBL, wh, ZH - LBL])
    ctx = z["ctx"]
    ax.fill_between(xc, ctx.mean(0), color="#E3E0DB", lw=0, zorder=1)
    for k, col in (("crown", ns.C_CROWN), ("fundus", ns.C_FUND), ("hinge", ns.C_3HG)):
        m = z[k].mean(0); se = z[k].std(0, ddof=1) / np.sqrt(len(z[k]))
        ax.fill_between(xc, m - se, m + se, color=col, alpha=.22, lw=0, zorder=2)
        ax.plot(xc, m, color=col, lw=0.9, zorder=3, solid_capstyle="round")
    ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(bottom=0); ax.set_yticks([])
    ax.set_xticks([z["q25"], z["q75"]]); ax.set_xticklabels(["25th", "75th"], fontsize=5.6)
    ax.set_xlabel("Fold depth (sulc percentile)", fontsize=6.2, labelpad=1.2)
    ax.tick_params(axis="x", length=1.5, pad=1.0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    yl = ax.get_ylim()[1]
    for xx, ha, lab, col in ((-1.34, "right", "3HG", ns.C_3HG),
                             (-0.80, "left", "Crown", ns.C_CROWN),
                             (1.30, "left", "Fundus", ns.C_FUND)):
        ax.text(xx, yl * .80, lab, ha=ha, va="center", fontsize=5.8, color=col,
                fontweight="bold")
    xl = 0.13
    i = int(np.argmin(np.abs(xc - xl)))
    ytop_ = yl * 0.50
    ax.plot([xl, xl], [ctx.mean(0)[i] + yl * 0.02, ytop_ - yl * 0.015], color="#B9B5AE",
            lw=.45, zorder=2)
    ax.text(xl, ytop_, "Cortex", ha="center", va="bottom", fontsize=5.8, color="#9A968F",
            fontweight="bold", zorder=4)

    # ---- the key, in the same band ----------------------------------------------------------
    rows = (("Number", "same $n$"),
            ("Fold depth", "most-gyral quartile"),
            ("Distance", "$\\geq$ 8 mm from a 3HG"),
            ("Region", "same parcel"))
    xk = X1 - wk
    LS = 0.0115                    # line pitch, raw: 9 lines span the band
    _txt(fig, xk, ytop, "Crown matched on", ha="left", va="top", fontsize=5.8,
         color=ns.C_CROWN, fontweight="bold")
    for i, (k, v) in enumerate(rows):
        yy = ytop - LS * (1.3 + 2 * i)
        _txt(fig, xk + 0.004, yy, k, ha="left", va="top", fontsize=5.4, color=ns.INK,
             fontweight="bold")
        _txt(fig, xk + 0.010, yy - LS, v, ha="left", va="top", fontsize=5.4, color=ns.INK)
    return axZ


def panels_cd(fig, ytop=0.588, letter="c", zoom_ax=None):
    """Panel c: where the three sampling classes actually sit, on one specimen's white surface.

    This replaces the sub-white-matter fiber profile pair. The class DEFINITIONS are the ones the
    measurement uses, so the control becomes inspectable rather than a sentence in the methods:
    crown sits on the crests beside the hinges, not on the banks, which is why the contrast is
    hinge-versus-crest and not gyrus-versus-sulcus. Each landmark is drawn as its 2 mm geodesic
    disk - the estimator's actual support.

    The crown and fundus DRAWS are stratified to the 3HG per-parcel histogram, which the measured
    draws are not. See render_classes.py: a uniform draw is regionally lopsided (density CV 0.96
    against 0.37 for 3HG) and reads as a rendering fault. Illustration only; the legend must not
    call this the analysed sample."""
    from render_classes import CLASSES, SID, PAINT
    box = None
    ims = []
    for name, _ in CLASSES:
        im = Image.open(f"{OUT}/_classes/S{SID}_{name}.png")
        b = im.getchannel("A").getbbox()
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3]))
        ims.append(im)
    # one crop box for all three, or the brains would be scaled independently
    out = []
    for im in ims:
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3])
        out.append(np.asarray(bg.crop(box)))
    # The row is width-limited, not height-limited: the render is 1.54:1, so three of them side by
    # side run out of horizontal room long before they fill the band under panels a,b. Take the
    # full width the left column has (out to the heatmap gutter at 0.685) and centre the result
    # vertically in what is left, instead of hanging it from a fixed top edge with the slack
    # dumped underneath.
    # X1 stops short of the heatmap's row labels, which reach ~0.075 to the left of its axes at
    # 0.685 and were being overrun by the third brain
    X0, X1, GAPX = 0.030, 0.565, 0.006
    W = (X1 - X0 - 2 * GAPX) / 3
    Hbox = W * (out[0].shape[0] / out[0].shape[1]) * (FW / FH)
    # The class labels go UNDER their brains, clear of the letter line above.
    # The letter sits on panel d's baseline: panels_fg lifts its letter clear of the rotated
    # metric headers, and the two top-row letters on two different lines read as a mistake.
    # anchored to the ytop it is given. It used to derive its own top from panels_fg's letter
    # line, which silently ignored the argument and put the brains on top of the matching row.
    traw = ytop
    tfig = _y(traw)
    TITLE = 0.014                                          # clearance under the letter line
    y0 = traw - TITLE - Hbox
    for k, ((name, col), img) in enumerate(zip(CLASSES, out)):
        ax = _ax(fig, [X0 + k * (W + GAPX), y0, W, Hbox])
        ax.imshow(img, interpolation="lanczos"); ax.set_axis_off()
        if zoom_ax is not None and name in ("3HG", "crown"):
            # one camera for every class render, so the callout coordinates are shared
            co = np.load(f"{OUT}/_classes/S{rc.SID}_callout.npz")
            cx, cy, hw = float(co["x"]) - box[0], float(co["y"]) - box[1], float(co["half"])
            hy = hw * 0.77                                  # the zoom is 4:3, not square
            ax.add_patch(Rectangle((cx - hw, cy - hy), 2 * hw, 2 * hy, fill=False,
                                   edgecolor=ns.INK, lw=.6, zorder=5))
            zax = zoom_ax[0 if name == "3HG" else 1]
            for xc, fx in ((cx - hw, 0.0), (cx + hw, 1.0)):
                fig.add_artist(ConnectionPatch(
                    xyA=(fx, 0.0), coordsA=zax.transAxes,
                    xyB=(xc, cy - hy), coordsB=ax.transData,
                    color=ns.INK, lw=.4, alpha=.6, zorder=1))
        # label in the render colour, not the palette colour, so the label and the patches it
        # names are the same blue; panel b keeps the palette for its legend
        _txt(fig, X0 + k * (W + GAPX) + W / 2, y0 - 0.003, ns.cap(name), fontsize=7.5,
             color=PAINT[name], fontweight="bold", ha="center", va="top")
        if k == 0:
            fig.text(0.012, _y(0.945), letter, fontsize=8, fontweight="bold",
                     va="bottom")
    print(f"  panel c: {out[0].shape[1]}x{out[0].shape[0]} px per brain, "
          f"{W * FW / ns.MM:.0f} mm wide each", flush=True)


def panels_cd_fiber(fig):
    """Panels c and d of the full plate: DBSI fiber fraction from cortex into superficial WM.

    This is the pair the three-class surface render replaced when the a-d figure was split out.
    The surface render answers "where is the control", which the standalone signature figure
    needs; the full plate has room for both questions, and the fiber convergence is a result
    rather than a definition, so on the plate it keeps the two panels. c is the raw pair of
    depth profiles, d their difference - the quantity the Wilcoxon actually tests, over the
    0.5-3.0 mm sub-white-matter band."""
    z = np.load(f"{OUT}/_depthAB_cache.npz"); Hs, Cs = z["Hs"], z["Cs"]
    D = fc.DEPTHS
    pAB = stats.wilcoxon(Hs[:, fc.SUBWM].mean(1) - Cs[:, fc.SUBWM].mean(1)).pvalue
    for tag, x0, w in (("c", 0.115, 0.175), ("d", 0.425, 0.170)):
        ax = _ax(fig, [x0, 0.400, w, 0.155])
        ax.axvspan(0, D.max(), color="#f2f0ec", zorder=0)
        ax.axvline(0, color=ns.INK, lw=.5)
        if tag == "c":
            for M, col, lab in [(Cs, ns.C_CROWN, "crown"), (Hs, ns.C_3HG, "3HG")]:
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(6)
                ax.plot(D, mu, color=col, lw=1.1, label=lab, solid_capstyle="round", zorder=3)
                ax.fill_between(D, mu - se, mu + se, color=col, alpha=.16, lw=0, zorder=2)
            ax.set_ylabel("DBSI fiber fraction", fontsize=6.5, labelpad=2)
            ax.legend(loc="lower right", handlelength=.9, fontsize=5.8, borderpad=.2,
                      labelspacing=.3, handletextpad=.4)
            ns.nletter(ax, "c", x=-0.30, y=1.06)
        else:
            diff = Hs - Cs; mu = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(6)
            ax.axhline(0, color=ns.INK, lw=.5, zorder=1)
            ax.fill_between(D, 0, mu, where=(mu > 0), color=ns.C_3HG, alpha=.13, zorder=1)
            ax.fill_between(D, mu - se, mu + se, color=ns.C_3HG, alpha=.22, lw=0, zorder=2)
            ax.plot(D, mu, color=ns.C_3HG, lw=1.1, solid_capstyle="round", zorder=3)
            ax.set_ylabel("Fiber-fraction difference", fontsize=6.5, labelpad=2)
            ax.text(.96, .08, f"$P$ = {pAB:.3f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6)
            ns.nletter(ax, "d", x=-0.34, y=1.06)
        y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + .16 * (y1 - y0))
        ax.text(.14, .965, "Cortex", transform=ax.transAxes, ha="center", va="top", fontsize=5.8)
        ax.text(.64, .965, "White matter", transform=ax.transAxes, ha="center", va="top",
                fontsize=5.8)
        ax.set_xlabel("Distance from white surface (mm)", fontsize=6.5, labelpad=2)
    print(f"  panels c,d: sub-WM fiber 3HG>crown in "
          f"{int((Hs[:, fc.SUBWM].mean(1) > Cs[:, fc.SUBWM].mean(1)).sum())}/6, P = {pAB:.4f}",
          flush=True)


def panel_e(fig):
    """Panel f: the raw depth trajectories, with the tested difference on a strip beneath.

    Two complaints have to be answered at once. The trajectory is the point - how each metric
    runs from white to pial - so it stays. But on an axis wide enough to hold that trajectory
    (MD runs 0.8 to 1.35) a 0.12 SD offset between the two classes is a line width, and the panel
    reads as a null while the statistics report an effect. The strip carries dWP, the quantity
    depth_stats.py actually tests: 3HG minus that parcel's own gyral crown, standardized by the
    within-parcel SD and averaged over parcels. Same grammar as panels c and d."""
    rho, Yh, Yc, dWP = load_depth()
    TOP, STRIP, GAP, BOT = 0.148, 0.052, 0.012, 0.088
    for i, (m, lab) in enumerate(ORDER):
        x0 = 0.055 + i * 0.161
        ax = fig.add_axes([x0, BOT + STRIP + GAP, 0.126, TOP])
        for Y, col in [(Yc[m], ns.C_CROWN), (Yh[m], ns.C_3HG)]:
            mu = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(6)
            ax.plot(rho, mu, color=col, lw=1.0, solid_capstyle="round", zorder=3)
            ax.fill_between(rho, mu - se, mu + se, color=col, alpha=.16, lw=0, zorder=2)
        # the two curves are close by construction; shading between them makes the offset an
        # area rather than a gap the eye has to measure
        ax.fill_between(rho, Yc[m].mean(0), Yh[m].mean(0), color=ns.MUTE, alpha=.22, lw=0,
                        zorder=1)
        ax.set_title(lab, fontsize=6.5, pad=2)
        if m == "MD":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.yaxis.get_offset_text().set(size=5, color=ns.INK)
        ax.set_xticks([0, .5, 1]); ax.set_xticklabels([])
        ax.tick_params(labelsize=5.5)
        if i == 0:
            ns.nletter(ax, "f", x=-0.38, y=1.12)

        sx = fig.add_axes([x0, BOT, 0.126, STRIP])
        D = dWP[m]; mu = np.nanmean(D, 0); se = np.nanstd(D, 0, ddof=1) / np.sqrt(D.shape[0])
        sx.axhline(0, color=ns.INK, lw=.4, ls=(0, (2.5, 2)), zorder=1)
        # Colour by the sign AT EACH DEPTH, not by the sign of the whole curve. Colouring the
        # whole line by its mean made FA blue and fiber orange for a reason the reader cannot
        # see, and hid the very thing those two metrics are reported for - that they cross zero.
        # Same device as panel d: area above the line is 3HG-ward, below it is crown-ward.
        sx.fill_between(rho, 0, mu, where=(mu > 0), color=ns.C_3HG, alpha=.30, lw=0,
                        interpolate=True, zorder=2)
        sx.fill_between(rho, 0, mu, where=(mu < 0), color=ns.C_CROWN, alpha=.30, lw=0,
                        interpolate=True, zorder=2)
        sx.fill_between(rho, mu - se, mu + se, color=ns.MUTE, alpha=.20, lw=0, zorder=2)
        sx.plot(rho, mu, color=ns.INK, lw=.9, solid_capstyle="round", zorder=3)
        sx.set_ylim(-0.42, 0.42); sx.set_yticks([-0.3, 0, 0.3])
        sx.set_xticks([0, .5, 1]); sx.set_xticklabels(["white", "0.5", "pial"], fontsize=5.5)
        sx.tick_params(labelsize=5)
        sx.set_xlabel("Cortical depth", fontsize=6.5, labelpad=2)
        if i == 0:
            sx.set_ylabel("3HG − crown\n(SD)", fontsize=5.8, labelpad=2)


def panels_fg(fig, letter="e"):
    raw = rd.collect()
    keys = [k for k, _ in rd.METRICS]
    per = {}
    for nm, recs in raw.items():
        good = [r for r in recs if r["nh"] >= rd.MINN and r["ncr"] >= rd.MINN]
        if len(good) >= 4:
            per[nm] = {k: np.array([r["d"][k] for r in good]) for k in keys}
    ordered, lobe_of = [], {}
    for lobe, names in rd.LOBE.items():
        for nm in names:
            if nm in per:
                ordered.append(nm); lobe_of[nm] = lobe
    Z = np.array([[np.mean(per[nm][k]) for k in keys] for nm in ordered])
    nr, nc = Z.shape
    # Fixed scale, not a per-run percentile. A data-driven limit silently rescales the panel
    # every time the underlying numbers change - the merge to junction-level values shifted it -
    # so the same colour meant a different effect size before and after. +-0.5 SD covers the
    # observed range and is the same number the colourbar is labelled with.
    vmax = 0.5

    # ---- f: heatmap ----
    # In the standalone a-d figure the left column ends at the b/c x labels (~0.435 raw); a
    # heatmap running 0.10 lower made d visibly taller than everything beside it. The plate has
    # panel f underneath, so there the full height is right.
    # HT leaves the colourbar room under the heatmap instead of pushing it off the canvas, and
    # HB stops the column short of the page edge; in the plate the full height is right.
    HT, HB = (0.930, 0.385) if YLO == 0.0 else (0.900, 0.400)
    ax = _ax(fig, [0.685, HB, 0.115, HT - HB])
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    Zc = np.clip(Z, -vmax, vmax)          # clip, so nothing saturates off-scale silently
    im = ax.pcolormesh(np.ma.masked_invalid(Zc), cmap=ns.DIV, norm=norm,
                       edgecolors="white", linewidth=.9)
    ax.set_aspect("auto"); ax.invert_yaxis()
    ax.set_xticks(np.arange(nc) + .5)
    ax.set_xticklabels([lab for _, lab in rd.METRICS], rotation=40, ha="left", fontsize=5.2)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(nr) + .5)
    ax.set_yticklabels([ns.cap(nm.replace("middle", "mid").replace("frontal", "front.")
                        .replace("parietal", "par.").replace("temporal", "temp.")
                        .replace("occipital", "occ.").replace("orbito", "orb.")
                        ) for nm in ordered], fontsize=5.0)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    LN = {"F": "Frontal", "P": "Parietal", "T": "Temporal", "O": "Occipital",
          "C": "Cingulate"}
    prev, start = None, 0
    for i, nm in enumerate(ordered + [None]):
        lb = lobe_of.get(nm) if nm else None
        if lb != prev and prev is not None:
            ax.plot([nc + .12, nc + .12], [start + .12, i - .12], lw=.9, color=ns.INK,
                    clip_on=False)
            ax.text(nc + .42, (start + i) / 2, LN[prev], rotation=270, va="center", ha="center",
                    fontsize=5.2, fontweight="bold")
            start = i
        if lb != prev:
            start = i; prev = lb
    cax = _ax(fig, [0.755, HB - 0.048, 0.155, 0.009])   # clear of the heatmap's bottom edge
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("3HG − crown, within region (SD)", fontsize=5.8, labelpad=2)
    cb.set_ticks([-vmax, 0, vmax])       # the surfaces share this scale, so state its ends
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.3, length=1.5, width=.5)
    # The rotated metric headers sit above the heatmap's top edge and reach roughly 8 mm up, so
    # the letter has to clear them rather than sit at a fixed fraction - which put it inside them.
    # The offset is in mm because the two figures this function serves have different canvases.
    ytitle = _y(HT) + (9.8 * ns.MM) / (FH * (YHI - YLO))
    fig.text(0.607, ytitle, letter, fontsize=8, fontweight="bold", va="bottom")

    # ---- g: fsaverage maps ----
    FSA = os.environ["FREESURFER_HOME"] + "/subjects/fsaverage"
    coords, faces = nib.freesurfer.read_geometry(f"{FSA}/surf/rh.inflated")
    sulc = nib.freesurfer.read_morph_data(f"{FSA}/surf/rh.sulc")
    alab, _, anames = nib.freesurfer.read_annot(f"{FSA}/label/rh.aparc.annot")
    anames = [n.decode() if isinstance(n, bytes) else n for n in anames]
    allreg = {nm: {k: np.mean([r["d"][k] for r in recs]) for k in keys}
              for nm, recs in raw.items()}
    from nilearn import plotting

    def render(vert, view):
        ff = plt.figure(figsize=(3, 3)); aa = ff.add_subplot(111, projection="3d")
        plotting.plot_surf_stat_map((coords, faces), vert, hemi="right", view=view, bg_map=sulc,
                                    cmap=ns.DIV, vmin=-vmax, vmax=vmax, symmetric_cbar=True,
                                    colorbar=False, axes=aa, figure=ff)
        aa.set_box_aspect((1, 1, .8), zoom=1.5)
        buf = io.BytesIO()
        ff.savefig(buf, format="png", transparent=True, dpi=500, bbox_inches="tight",
                   pad_inches=0)
        plt.close(ff); buf.seek(0)
        imr = Image.open(buf); return np.asarray(imr.crop(imr.getbbox()))

    top, bot = HT, HB                                # brains alongside the heatmap, same span
    rh_ = (top - bot) / len(rd.METRICS)
    for j, (k, lab) in enumerate(rd.METRICS):
        vv = np.full(len(coords), np.nan)
        for jj, nm in enumerate(anames):
            if nm in ("unknown", "corpuscallosum"):
                continue
            vv[alab == jj] = allreg.get(nm, {}).get(k, 0.0)
        yc = top - j * rh_ - rh_ / 2
        img_h = rh_ * .60
        arrs = [render(vv, view) for view in ("lateral", "medial")]
        ws = [img_h * (a.shape[1] / a.shape[0]) * (FH / FW) for a in arrs]
        x = 0.912 - (sum(ws) + 0.003) / 2            # brains column right of the heatmap
        for arr, w in zip(arrs, ws):
            axb = _ax(fig, [x, yc - img_h / 2, w, img_h])
            axb.imshow(arr); axb.axis("off"); axb.set_rasterized(True)
            x += w + 0.003
        _txt(fig, 0.912, yc - img_h / 2 - 0.004, lab, ha="center", va="top", fontsize=5.5)

def main(with_f=True):
    global YLO, YHI
    if with_f:
        YLO, YHI = 0.0, 1.0
        H = FH
    else:
        # the a-d block runs from just under the colour bar to the top of the canvas
        YLO, YHI = 0.300, 1.0
        H = FH * (YHI - YLO)
    fig = plt.figure(figsize=(FW, H), facecolor="white")
    # Two different figures out of one panel set.
    #   plate:     a,b effect sizes | c,d fiber convergence | e regional | f depth
    #   signature: a  the three sampling classes, on top, because it is the definition the rest of
    #              the figure rests on and the reader needs it before the numbers
    #              b,c effect sizes | d regional
    if with_f:
        # a and b take their letter baseline from e, which computes its own line to clear the
        # rotated metric headers; three top-row letters, one line
        ty = (0.930 + 9.8 * ns.MM / FH - 0.635) / 0.29
        panel_a(fig, ty=ty); panel_b(fig, ty=ty)
        panels_cd_fiber(fig); panels_fg(fig, "e"); panel_e(fig)
    else:
        # the matching row the co-author asked for sits above the class surfaces: the reader needs
        # to see how a 3HG is found and what its control satisfies before any number is shown
        # the sampling rule and the classes it produces are one panel: the reader needs the
        # first to read the second, and two letters would imply two claims
        axZ = panel_match(fig, ytop=0.945)
        panels_cd(fig, ytop=0.812, letter="a", zoom_ax=axZ)
        panel_a(fig, y0=0.335, letter="b")
        panel_b(fig, y0=0.335, letter="c")
        panels_fg(fig, "d")
    name = "FIG1_combined" if with_f else "FIG1_signature_ad"
    for ext, dpi in (("png", DPI), ("pdf", None)):     # tight: no blank strip above or below
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    w, h = Image.open(f"{OUT}/{name}.png").size
    print(f"wrote {name}.png  {w/DPI*25.4:.0f} x {h/DPI*25.4:.0f} mm")


if __name__ == "__main__":
    main("--no-f" not in sys.argv)

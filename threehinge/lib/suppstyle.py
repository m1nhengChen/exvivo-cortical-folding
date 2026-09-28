#!/usr/bin/env python3
"""Classic-matplotlib styling for the supplementary robustness figures.

The look the user pointed at: saturated tab-blue / orange / yellow-green fills with black
edges, violins carrying a hollow black inner boxplot with red '+' fliers, a full four-spine
frame, framed legends with round swatches, and star brackets for significance. Fonts stay
on natstyle's (Liberation Sans at Nature sizes) so the supplement still matches the mains.
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np
from scipy import stats as _st
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

BLUE, ORANGE, GREEN, GRAY = "#1f77b4", "#ff7f0e", "#9acd32", "#8c8c8c"


def frame(ax, lw=0.8):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw); sp.set_color("black")
    ax.tick_params(color="black")


def stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def violin_box(ax, center, vals, color, width=0.8, tf=None, itf=None):
    """violin (black edge, saturated fill) + hollow black inner box + red '+' fliers.
    tf/itf: forward/inverse transforms so the kde is done in e.g. log space."""
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    tf = tf if tf is not None else (lambda x: x)
    itf = itf if itf is not None else (lambda x: x)
    t = tf(v)
    if len(v) > 1 and np.ptp(t) > 0:
        kd = _st.gaussian_kde(t)
        tt = np.linspace(t.min() - 0.32 * np.ptp(t), t.max() + 0.32 * np.ptp(t), 140)
        dd = kd(tt); dd = dd / dd.max() * (width / 2)
        ax.fill_betweenx(itf(tt), center - dd, center + dd, facecolor=color,
                         edgecolor="black", linewidth=0.6, zorder=2)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    inl = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
    out = v[(v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)]
    lo, hi = inl.min(), inl.max()
    bw = width * 0.16
    ax.plot([center, center], [lo, q1], color="black", lw=0.7, zorder=4)
    ax.plot([center, center], [q3, hi], color="black", lw=0.7, zorder=4)
    for yy in (lo, hi):
        ax.plot([center - bw * 0.7, center + bw * 0.7], [yy, yy], color="black", lw=0.7,
                zorder=4)
    ax.add_patch(Rectangle((center - bw, q1), 2 * bw, q3 - q1, facecolor="none",
                           edgecolor="black", linewidth=0.7, zorder=5))
    ax.plot([center - bw, center + bw], [med, med], color="black", lw=0.9, zorder=6)
    if len(out):
        ax.plot(np.full(len(out), center), out, ls="none", marker="+", color="red",
                ms=3.2, mew=0.8, zorder=6, clip_on=False)


def sig_bracket(ax, x1, x2, y, text, h=0.012, fontsize=7):
    """example-figure significance bracket: down-ticks at both ends, stars above"""
    span = np.diff(ax.get_ylim())[0]
    ax.plot([x1, x1, x2, x2], [y - h * span, y, y, y - h * span], color="black", lw=0.8,
            zorder=7, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.15 * h * span, text, ha="center", va="bottom",
            fontsize=fontsize, zorder=7)


def legend_swatch(ax, entries, **kw):
    """framed legend with the examples' round colour swatches"""
    kw.setdefault("frameon", True); kw.setdefault("edgecolor", "black")
    kw.setdefault("facecolor", "white"); kw.setdefault("framealpha", 1)
    kw.setdefault("fontsize", 6); kw.setdefault("borderpad", 0.5)
    kw.setdefault("labelspacing", 0.45); kw.setdefault("handletextpad", 0.5)
    leg = ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=5.2, mfc=c,
                                    mec="black", mew=0.6, label=t) for c, t in entries], **kw)
    leg.get_frame().set_linewidth(0.7)
    return leg

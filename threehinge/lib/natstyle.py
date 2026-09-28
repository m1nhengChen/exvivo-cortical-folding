#!/usr/bin/env python3
"""Nature-style matplotlib defaults: Helvetica-like type, NPG palette, minimal chrome."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Nature Publishing Group (ggsci npg) palette
NPG = dict(red="#E64B35", cyan="#4DBBD5", teal="#00A087", navy="#3C5488",
           salmon="#F39B7F", slate="#8491B4", mint="#91D1C2", crimson="#DC0000",
           brown="#7E6148", taupe="#B09C85")
INK = "#1a1a1a"; GRID = "#e6e6e6"; MUTE = "#9aa0a6"

# roles
# Class colours: warm = 3HG-ward, cool = crown-ward, held consistent across every figure.
#
# These were a saturated advertising orange (#EA8121) against a strong blue (#1F6FB4). Journal
# figures in this family sit at much lower chroma - a coral rather than an orange, a slate blue
# rather than a primary one - which is what stops a six-panel plate from looking like a poster.
# The hues and their meanings are unchanged, so nothing downstream has to be relabelled.
BLUE = "#4E6E9F"; ORANGE = "#DE7A5A"; BLUE_L = "#9DB4D0"
TEAL = "#4FA894"                                   # third class, when one is needed
# In a sensitivity figure the published configuration is drawn in INK and every variant in colour,
# so the reference is found without reading labels. Class figures keep the warm/cool pair above.
C_REF = INK
C_3HG = ORANGE; C_CROWN = BLUE; C_FUND = BLUE_L
import os as _os


def _rgb2lab(rgb):
    import numpy as _np
    M = _np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
                   [.0193339, .1191920, .9503041]]); WP = _np.array([.95047, 1.0, 1.08883])
    c = _np.asarray(rgb, float)
    xyz = _np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4) @ M.T / WP
    f = _np.where(xyz > (6 / 29) ** 3, _np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return _np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                      200 * (f[..., 1] - f[..., 2])], -1)


def _lab2rgb(lab):
    import numpy as _np
    M = _np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
                   [.0193339, .1191920, .9503041]]); WP = _np.array([.95047, 1.0, 1.08883])
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    g = lambda t: _np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))
    lin = _np.stack([g(fx), g(fy), g(fz)], -1) * WP @ _np.linalg.inv(M).T
    lin = _np.clip(lin, 0, 1)
    return _np.where(lin <= .0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - .055)


def lab_diverging(neg, pos, centre="#f4f2f0", Lc=96.0, n=256, name="lab_div"):
    """Diverging map built in CIELAB: end -> neutral centre -> end, so lightness is monotone on
    each wing and the two wings can be given the same chroma. This is how the low-chroma maps
    that match the page's class colours are made; a hand-picked hex ramp cannot promise either."""
    import numpy as _np
    from matplotlib.colors import to_rgb
    A, B, C = _rgb2lab(to_rgb(neg)), _rgb2lab(to_rgb(pos)), _rgb2lab(to_rgb(centre)).copy()
    C[0] = Lc
    t = _np.linspace(0, 1, n // 2)[:, None]
    return LinearSegmentedColormap.from_list(name, _lab2rgb(_np.vstack([A * (1 - t) + C * t,
                                                                        C * (1 - t) + B * t])))


# Diverging map for SIGNED DIFFERENCES. Deliberately NOT on the class hues: across the paper warm
# and cool mean 3HG and crown - class identity - so a 3HG-minus-crown map drawn in those same two
# colours makes a cool cell read as "crown" when it means "3HG lower here". Plum (hue ~290) and
# slate-teal (hue ~190) sit well away from both coral (~40) and slate blue (~215), and at chroma
# C* 23-31 they match the page's own low-chroma palette - every brighter candidate, however
# well-constructed, jumped off the plate. Built in CIELAB so lightness is monotone on each wing
# and the two ends are equally weighted (L* 43 / 49). Teal is the POSITIVE end, as it has been
# since the BrBG version, so nothing written about the figure changes polarity.
DIV = lab_diverging("#7A5A8A", "#4C7C82", name="plum_teal")
C_NEG, C_POS = "#7A5A8A", "#4C7C82"        # DIV's two ends, for marks that carry a sign not a class

# trial hook: DIV_CMAP=<matplotlib name> or "lab:<neg hex>:<pos hex>" swaps the signed-difference
# map for one build without touching the default
if _os.environ.get("DIV_CMAP"):
    import matplotlib.pyplot as _plt
    from matplotlib.colors import to_hex as _to_hex
    _spec = _os.environ["DIV_CMAP"]
    DIV = (lab_diverging(*_spec.split(":")[1:3]) if _spec.startswith("lab:")
           else _plt.get_cmap(_spec))
    C_NEG, C_POS = _to_hex(DIV(0.0)), _to_hex(DIV(1.0))
# single-hue ramp on the same slate, for value (not class) scales
SEQ = LinearSegmentedColormap.from_list(
    "slate_seq", ["#f4f2f0", "#D9E1EB", "#9DB4D0", "#6E8DB6", BLUE, "#3A5379", "#22334C"])
C_FA = NPG["navy"]; C_FIBER = NPG["teal"]; C_RESTR = NPG["red"]


def cap(s):
    """Capitalise a standalone label. str.capitalize() would turn '3HG' into '3hg'."""
    return s[:1].upper() + s[1:]


def use():
    mpl.rcParams.update({
        "font.family": "Nimbus Sans", "font.size": 11,
        "axes.edgecolor": INK, "axes.linewidth": 0.9, "axes.labelcolor": INK,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 9.5, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def panel(ax, letter, x=-0.15, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="left")


def title(ax, s, pad=8):
    ax.set_title(s, loc="left", fontsize=12, fontweight="bold", color=INK, pad=pad)


# ---- Nature figure specification -------------------------------------------------------------
# Widths are fixed by the journal: 89 mm single column, 120 mm 1.5 column, 183 mm double column;
# max height 247 mm. Type is 5-7 pt sans (never below 5 pt at final size), panel letters 8 pt bold.
# Kept separate from use() so existing figures are unaffected.
MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM


def nature():
    mpl.rcParams.update({
        "font.family": "Nimbus Sans", "font.size": 7,
        "axes.edgecolor": INK, "axes.linewidth": 0.5, "axes.labelcolor": INK,
        "axes.labelsize": 7, "axes.titlesize": 7, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 6, "legend.frameon": False,
        "lines.linewidth": 1.0, "patch.linewidth": 0.4,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def nletter(ax, letter, x=-0.16, y=1.02):
    """Nature panel letter: bold lowercase, 8 pt, flush left above the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def ptitle(ax, text, letter=None, lx=-0.30, y=1.06, size=7.5, lsize=8, loc="center"):
    """Descriptive panel title and its letter, drawn on ONE baseline.

    Journal figures in this family caption every panel with a sentence saying what it shows, not
    just an axis label. The letter has to be set with the title rather than separately: placed by
    two different calls at two different y values it ends up floating above and away from the text
    it belongs to, which is what nletter() at y = 1.15 was doing.

    Both are drawn in axes coordinates with va="bottom", so they share a baseline whatever the
    panel height. lx is the letter's x in axes coordinates - negative, to clear the tick labels."""
    tx = {"center": 0.5, "left": 0.0}[loc]
    ax.text(tx, y, text, transform=ax.transAxes, ha={"center": "center", "left": "left"}[loc],
            va="bottom", fontsize=size, color=INK)
    if letter is not None:
        ax.text(lx, y, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=lsize,
                fontweight="bold", color=INK)

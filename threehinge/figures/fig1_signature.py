#!/usr/bin/env python3
"""Figure 1 (Nature-style): 3HG carry a coherent multimodal microstructural signature."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import natstyle as ns
ns.use()

OUT = (_env.ROOT + "/surf/three_hinge/results"); SUBJ = [1, 2, 3, 4, 5, 6]
NICE = {"FA": "FA", "fiber_ratio": "DBSI fiber", "restricted_ratio": "DBSI restricted",
        "hindered_ratio": "DBSI hindered", "water_ratio": "DBSI free water", "AD": "AD",
        "RD": "RD", "ADC": "MD", "thickness": "Cortical thickness", "QSM": "QSM",
        "QSM_pos": "QSM+", "QSM_neg": "QSM−", "T2star": "T2*"}


def crop(path):
    im = Image.open(path); bbox = im.getbbox()
    return np.asarray(im.crop(bbox))


def per_subject_d(ref):
    z0 = np.load(f"{OUT}/Subject1_landmarks.npz", allow_pickle=True); mods = [str(m) for m in z0["mods"]]
    D = {m: [] for m in mods}
    for sid in SUBJ:
        z = np.load(f"{OUT}/Subject{sid}_landmarks.npz", allow_pickle=True)
        h, r = z["hinge"], z[ref]
        for k, m in enumerate(mods):
            a, b = h[k], r[k]; a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            D[m].append((a.mean() - b.mean()) / sp if sp else np.nan)
    return mods, {m: np.array(v) for m, v in D.items()}


def zmeans():
    z0 = np.load(f"{OUT}/Subject1_landmarks.npz", allow_pickle=True); mods = [str(m) for m in z0["mods"]]
    acc = {g: {m: [] for m in mods} for g in ("hinge", "crown", "fundus")}
    for sid in SUBJ:
        z = np.load(f"{OUT}/Subject{sid}_landmarks.npz", allow_pickle=True)
        for k, m in enumerate(mods):
            allv = np.concatenate([z[g][k] for g in ("hinge", "crown", "fundus")])
            mu, sd = np.nanmean(allv), np.nanstd(allv) or 1
            for g in ("hinge", "crown", "fundus"):
                acc[g][m].append((np.nanmean(z[g][k]) - mu) / sd)
    return acc


def main():
    fig = plt.figure(figsize=(9.2, 5.0), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.32,
                           left=0.13, right=0.985, top=0.86, bottom=0.14)

    # ---- a: forest, 3HG vs crown ----
    mods, Dc = per_subject_d("crown")
    order = sorted(mods, key=lambda m: np.nanmean(Dc[m]))
    axB = fig.add_subplot(gs[0])
    for i, m in enumerate(order):
        d = Dc[m]; mu = np.nanmean(d); col = ns.C_3HG if mu > 0 else ns.C_CROWN
        cons = np.all(d > 0) or np.all(d < 0)
        axB.plot([0, mu], [i, i], color=col, lw=1.0, alpha=.5 if cons else .2, zorder=2)
        axB.scatter(d, np.full(6, i), s=13, color=col, alpha=.32 if cons else .16,
                    zorder=3, edgecolor="none")
        axB.scatter([mu], [i], s=52, color=col, zorder=4, edgecolor="white", linewidth=0.8,
                    alpha=1 if cons else .4)
    axB.axvline(0, color=ns.INK, lw=0.7, zorder=1)
    axB.set_yticks(range(len(order))); axB.set_yticklabels([NICE.get(m, m) for m in order])
    axB.set_ylim(-0.6, len(order)-0.4); axB.set_xlim(-1.35, 1.45)
    axB.set_xlabel("effect size   (Cohen's d,  3HG − gyral crown)")
    axB.tick_params(axis="y", length=0)
    ns.panel(axB, "a", x=-0.42, y=1.12)

    # ---- b: three-way ----
    acc = zmeans()
    keys = ["thickness", "FA", "fiber_ratio", "restricted_ratio", "hindered_ratio", "ADC"]
    axC = fig.add_subplot(gs[1]); y = np.arange(len(keys))[::-1]; h = 0.25
    for gi, (g, col, lab) in enumerate([("hinge", ns.C_3HG, "3HG"),
                                        ("crown", ns.C_CROWN, "gyral crown"),
                                        ("fundus", ns.C_FUND, "sulcal fundus")]):
        off = (gi - 1) * h
        m = [np.mean(acc[g][k]) for k in keys]; se = [np.std(acc[g][k]) / np.sqrt(6) for k in keys]
        axC.barh(y + off, m, h*0.9, color=col, alpha=0.92, label=lab, zorder=2,
                 xerr=se, error_kw=dict(lw=0.7, ecolor="#666", capsize=1.5))
    axC.axvline(0, color=ns.INK, lw=0.7)
    axC.set_yticks(y); axC.set_yticklabels([NICE[k] for k in keys]); axC.tick_params(axis="y", length=0)
    axC.set_xlabel("standardized mean   (z within specimen)")
    # legend above the panel (horizontal) so it never sits on the bars
    axC.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               handlelength=1.0, columnspacing=1.3, labelspacing=0.3, borderpad=0.2)
    ns.panel(axC, "b", x=-0.34, y=1.12)

    fig.savefig(f"{OUT}/FIG1_signature.png", dpi=340)
    fig.savefig(f"{OUT}/FIG1_signature.pdf")
    print("wrote FIG1_signature.png", Image.open(f"{OUT}/FIG1_signature.png").size)


if __name__ == "__main__":
    main()

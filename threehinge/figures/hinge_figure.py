#!/usr/bin/env python3
"""Two-panel summary for the Subject-level 3HG proof of concept:
  (left)  grouped bars: 3HG vs crown vs fundus for a few interpretable modalities
          (z-scored within modality so different units share an axis);
  (right) geodesic ring profiles X(d) around 3HG (0-2/2-4/4-6/6-8 mm)."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = (_env.ROOT + "/surf/three_hinge/results")
BAR_MODS = ["thickness", "FA", "restricted_ratio", "QSM", "T2star"]


def load_table(sid):
    s = f"Subject{sid}"
    rows = {}
    with open(f"{OUT}/{s}_hinge_table.tsv") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            rows[p[0]] = [float(x) for x in p[1:]]
    return s, rows


def main(sid):
    s, rows = load_table(sid)
    prof = np.load(f"{OUT}/{s}_ring_profile.npz")

    fig, (axb, axp) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")

    # ---- left: z-scored group means ----
    labels, hz, cz, fz = [], [], [], []
    for m in BAR_MODS:
        if m not in rows:
            continue
        hm, _, cm, _, fm, _, _ = rows[m]
        mu = np.mean([hm, cm, fm]); sd = np.std([hm, cm, fm]) or 1.0
        labels.append(m); hz.append((hm-mu)/sd); cz.append((cm-mu)/sd); fz.append((fm-mu)/sd)
    x = np.arange(len(labels)); w = 0.26
    axb.bar(x-w, hz, w, label="3HG", color="#d1495b")
    axb.bar(x,   cz, w, label="gyral crown", color="#2a78d6")
    axb.bar(x+w, fz, w, label="sulcal fundus", color="#edae49")
    axb.axhline(0, color="#888", lw=.7)
    axb.set_xticks(x); axb.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axb.set_ylabel("z within modality"); axb.set_title(f"{s}: 3HG vs crown vs fundus", fontweight="bold")
    axb.legend(fontsize=8, frameon=False)

    # ---- right: ring profiles (each z-scored across rings for shared axis) ----
    r = prof["rings"]
    for m in ["QSM", "T2star", "FA", "restricted_ratio", "thickness"]:
        if m not in prof.files:
            continue
        y = prof[m].astype(float)
        yz = (y - np.nanmean(y)) / (np.nanstd(y) or 1.0)
        axp.plot(r, yz, marker="o", label=m)
    axp.set_xlabel("geodesic distance from 3HG (mm)")
    axp.set_ylabel("z across rings"); axp.set_title("Ring profile X(d)", fontweight="bold")
    axp.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(f"{OUT}/{s}_hinge_summary.png", dpi=180)
    from PIL import Image
    print("wrote", f"{OUT}/{s}_hinge_summary.png", Image.open(f"{OUT}/{s}_hinge_summary.png").size)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)

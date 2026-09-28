#!/usr/bin/env python3
"""Summarise the partial-volume / geometry validation: does the 3HG - crown depth effect survive
restricting to high-GM-purity voxels, and adjusting for the WM and non-brain fractions?

For each metric the depth profile of the standardized 3HG - crown difference is reduced to its
mean over the nine depths, per specimen, and tested against zero (n = 6). Three versions are
compared side by side: raw, high-purity subset, and partial-volume-adjusted."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import _env

import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
D = (_env.ROOT + "/surf/three_hinge/depth")
SUBJ = [1, 2, 3, 4, 5, 6]
KEY = ["MD", "hindered", "water", "fiber", "restricted", "FA", "AD", "RD", "FD",
       "DBSI_MD", "DBSI_FA", "DBSI_AD", "DBSI_RD", "fib1_FA", "fib1_AD", "fib1_RD"]


def main():
    Z = [np.load(f"{D}/Subject{s}_depth_pv.npz") for s in SUBJ]
    mets = [str(m) for m in Z[0]["mets"]]
    print("partial volume of the sampled diffusion voxels (mean over depths and specimens):")
    for k, lab in (("gm", "GM"), ("wm", "WM")):
        h = np.mean([z[f"{k}_h"].mean() for z in Z]); c = np.mean([z[f"{k}_c"].mean() for z in Z])
        print(f"  {lab} fraction   3HG {h:.3f}   crown {c:.3f}   difference {h - c:+.3f}")
    print(f"  samples with GM fraction >= 0.70: {100 * np.mean([z['npure'].mean() for z in Z]):.0f}%")

    keep = [m for m in KEY if m in mets]
    for pre, title in (("wp", "WITHIN-PARCEL contrast (the one the published depth statistics test)"),
                       ("", "count-matched whole-hemisphere contrast (cross-check)")):
        print(f"\n=== {title} ===")
        print(f"{'metric':11s}" + "".join(f"{t:>22s}" for t in ("raw", "GM purity >= 0.70",
                                                               "PV-adjusted")))
        print(f"{'':11s}" + "".join(f"{'d':>8s}{'6/6':>5s}{'P':>9s}" for _ in range(3)))
        for m in keep:
            report(Z, m, pre)


def report(Z, m, pre):
        line = f"{m:11s}"
        for k in ("raw", "pure", "adj"):
            key = f"{pre}{k}_{m}"
            if key not in Z[0].files:
                line += f"{'--':>22s}"; continue
            v = np.array([np.nanmean(z[key]) for z in Z])
            if np.all(np.isnan(v)):
                line += f"{'--':>22s}"; continue
            p = stats.ttest_1samp(v[np.isfinite(v)], 0).pvalue
            cons = int(np.all(np.sign(v[np.isfinite(v)]) == np.sign(v[np.isfinite(v)][0])))
            line += f"{np.nanmean(v):+8.3f}{cons:5d}{p:9.4f}"
        print(line)


def tail():
    print("\nd = standardized 3HG - crown difference averaged over the nine equivolumetric depths;")
    print("6/6 = the six specimens agree in sign; P = two-sided one-sample t over specimens.")


if __name__ == "__main__":
    main()
    tail()

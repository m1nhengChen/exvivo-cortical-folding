#!/usr/bin/env python3
"""Formal 3HG main effect and 3HG x depth interaction for the FIG2 depth profiles.

The unit is the specimen (n=6). For each metric, dWP[j,k] is the region-matched, within-parcel
SD-normalized 3HG-minus-crown difference for specimen j at equivolumetric depth k (rho 0.1..0.9,
white->pial), so it is already the contrast of interest: a one-sample test on dWP IS the
3HG effect, and a depth-contrast on dWP IS the 3HG x depth interaction.

  main effect      H0: mean_k dWP[j,k] = 0            -> one-sample test on the depth mean
  interaction      H0: dWP[j,k] does not vary with k  -> (i) linear contrast in rho (directed)
                                                         (ii) omnibus non-flatness of the profile

PRIMARY inference is parametric on the specimen-level contrast (one-sample t for main and linear,
one-way repeated-measures F with Greenhouse-Geisser for omnibus), BH-FDR across the six metrics
within each family. Parametric is primary because at n=6 an exact sign-flip test is RESOLUTION-
LIMITED: the smallest attainable two-sided p is 2/64 = 0.031, so BH over six metrics can never
return q < 0.19 no matter how large the effect. The exact sign-flip p and the sign-consistency
count (x/6) are reported alongside as distribution-free support, not as the decision rule."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import _env

import sys, itertools, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from statsmodels.stats.multitest import multipletests

DEPTH = (_env.ROOT + "/surf/three_hinge/depth")
RES = (_env.ROOT + "/surf/three_hinge/results")
SUBJ = [1, 2, 3, 4, 5, 6]
ORDER = [("fiber", "DBSI fiber"), ("MD", "MD"), ("hindered", "DBSI hindered"),
         ("water", "DBSI free water"), ("restricted", "DBSI restricted"), ("FA", "FA")]
SIGNS = np.array(list(itertools.product([1, -1], repeat=len(SUBJ))), float)   # 64 x 6, exact


def t1(x):
    """one-sample t statistic along axis -1 (studentized -> stable under sign-flip)."""
    return x.mean(-1) / (x.std(-1, ddof=1) / np.sqrt(x.shape[-1]) + 1e-300)


def signflip_signed(y):
    """exact two-sided sign-flip test on per-specimen scalars y (n,). Returns p, t_obs."""
    obs = t1(y)
    null = t1(SIGNS * y)                       # 64 studentized statistics
    return (np.abs(null) >= np.abs(obs) - 1e-12).mean(), obs


def rm_anova(X):
    """one-way repeated-measures F for depth-dependence of the contrast X (n specimens x k depths),
    Greenhouse-Geisser corrected. This IS the 3HG x depth interaction: X is already 3HG - crown."""
    n, k = X.shape
    SS_d = n * ((X.mean(0) - X.mean()) ** 2).sum()
    SS_e = ((X - X.mean(0) - X.mean(1, keepdims=True) + X.mean()) ** 2).sum()
    df1, df2 = k - 1, (n - 1) * (k - 1)
    F = (SS_d / df1) / (SS_e / df2)
    M = np.eye(k) - 1.0 / k                                # GG epsilon from the centred cov
    ev = np.linalg.eigvalsh(M @ np.cov((X - X.mean(0)).T) @ M)[1:]
    eps = float(np.clip(ev.sum() ** 2 / ((k - 1) * (ev ** 2).sum()), 1.0 / (k - 1), 1.0))
    return stats.f.sf(F, df1 * eps, df2 * eps), F, eps


def signflip_omnibus(C):
    """exact sign-flip test of H0: group-mean profile is zero. C is (n, k), rows already
    depth-centered. Statistic = sum_k (mean_j C[j,k])^2 (nonnegative -> one-sided by construction)."""
    stat = lambda M: (M.mean(0) ** 2).sum()
    obs = stat(C)
    null = np.array([stat(s[:, None] * C) for s in SIGNS])
    return (null >= obs - 1e-12).mean(), obs


def main():
    rho = np.load(f"{DEPTH}/Subject5_depth.npz")["rho"]
    w = rho - rho.mean(); w = w / (w ** 2).sum()          # linear-contrast weights (slope in 1/depth)
    rows = []
    for m, lab in ORDER:
        D = np.array([np.load(f"{DEPTH}/Subject{s}_depth.npz")[f"dWP_{m}"] for s in SUBJ], float)
        mean_j = D.mean(1)                                 # per-specimen depth-averaged effect
        slope_j = D @ w                                    # per-specimen depth gradient
        C = D - D.mean(1, keepdims=True)                   # depth-centered -> pure interaction part

        p_main_sf, _ = signflip_signed(mean_j)
        p_lin_sf, _ = signflip_signed(slope_j)
        p_omni_sf, _ = signflip_omnibus(C)

        t_main, p_main = stats.ttest_1samp(mean_j, 0)      # primary: main effect
        t_lin, p_lin = stats.ttest_1samp(slope_j, 0)       # primary: directed interaction
        p_omni, F_omni, eps = rm_anova(D)                  # primary: omnibus interaction

        rows.append(dict(
            metric=lab, key=m,
            main_mean=mean_j.mean(), main_sem=mean_j.std(ddof=1) / np.sqrt(6),
            main_dz=mean_j.mean() / mean_j.std(ddof=1), main_consist=int(np.sum(np.sign(mean_j) == np.sign(mean_j.mean()))),
            t_main=t_main, p_main=p_main, p_main_signflip=p_main_sf,
            slope=slope_j.mean(), slope_sem=slope_j.std(ddof=1) / np.sqrt(6),
            slope_dz=slope_j.mean() / slope_j.std(ddof=1), slope_consist=int(np.sum(np.sign(slope_j) == np.sign(slope_j.mean()))),
            t_interaction_linear=t_lin, p_interaction_linear=p_lin, p_interaction_linear_signflip=p_lin_sf,
            F_interaction_omnibus=F_omni, gg_eps=eps, p_interaction_omnibus=p_omni,
            p_interaction_omnibus_signflip=p_omni_sf,
            deep=D[:, :3].mean(), mid=D[:, 3:6].mean(), superficial=D[:, 6:].mean(),
            var_ratio=(C.mean(0) ** 2).sum() / ((D.mean(0) ** 2).sum() + 1e-300)))

    for src, dst in [("p_main", "q_main"),
                     ("p_interaction_linear", "q_interaction_linear"),
                     ("p_interaction_omnibus", "q_interaction_omnibus")]:
        q = multipletests([r[src] for r in rows], method="fdr_bh")[1]  # BH across the six metrics
        for r, qi in zip(rows, q): r[dst] = qi

    hdr = ["metric", "main_mean", "main_sem", "main_dz", "main_consist", "t_main", "p_main",
           "q_main", "p_main_signflip", "slope", "slope_sem", "slope_dz", "slope_consist",
           "t_interaction_linear", "p_interaction_linear", "q_interaction_linear",
           "p_interaction_linear_signflip", "F_interaction_omnibus", "gg_eps",
           "p_interaction_omnibus", "q_interaction_omnibus", "p_interaction_omnibus_signflip",
           "deep", "mid", "superficial", "var_ratio"]
    with open(f"{RES}/depth_stats.tsv", "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[h]:.4g}" if isinstance(r[h], float) else str(r[h]) for h in hdr) + "\n")

    star = lambda q: "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"
    print("\n3HG MAIN EFFECT   H0: depth-averaged 3HG-crown difference = 0   (n=6 specimens)")
    print(f"{'metric':16s}{'mean+-SEM':>17s}{'dz':>7s}{'sign':>6s}{'t':>7s}{'p':>8s}{'q':>8s}{'p_perm':>8s}")
    for r in rows:
        print(f"{r['metric']:16s}{r['main_mean']:+8.3f} +-{r['main_sem']:5.3f}{r['main_dz']:+7.2f}"
              f"{r['main_consist']:5d}/6{r['t_main']:7.2f}{r['p_main']:8.4f}{r['q_main']:8.4f}"
              f"{r['p_main_signflip']:8.3f}  {star(r['q_main'])}")

    print("\n3HG x DEPTH INTERACTION   H0: the difference does not vary with cortical depth")
    print(f"{'metric':16s}{'slope+-SEM':>17s}{'sign':>6s}{'p_lin':>8s}{'q_lin':>8s}"
          f"{'F_omni':>8s}{'p_omni':>8s}{'q_omni':>8s}{'p_perm':>8s}")
    for r in rows:
        print(f"{r['metric']:16s}{r['slope']:+8.3f} +-{r['slope_sem']:5.3f}"
              f"{r['slope_consist']:5d}/6{r['p_interaction_linear']:8.4f}{r['q_interaction_linear']:8.4f}"
              f"{r['F_interaction_omnibus']:8.2f}{r['p_interaction_omnibus']:8.4f}"
              f"{r['q_interaction_omnibus']:8.4f}{r['p_interaction_omnibus_signflip']:8.3f}  "
              f"{star(min(r['q_interaction_linear'], r['q_interaction_omnibus']))}")

    print("\nCLASSIFICATION (q<0.05)")
    for r in rows:
        M = r["q_main"] < .05
        I = min(r["q_interaction_linear"], r["q_interaction_omnibus"]) < .05
        cls = ("depth-INVARIANT offset" if M and not I else "depth-DEPENDENT (offset + gradient)"
               if M and I else "depth-DEPENDENT only (no net offset)" if I else "no effect")
        print(f"  {r['metric']:16s} main={'Y' if M else 'n'} interaction={'Y' if I else 'n'}   {cls}"
              f"   | deep {r['deep']:+.2f}  mid {r['mid']:+.2f}  superficial {r['superficial']:+.2f}")
    print(f"\nwrote {RES}/depth_stats.tsv")
    print("q = BH-FDR across the six metrics within each family; p_perm = exact sign-flip (64 patterns,")
    print("floor 0.031 signed / 0.016 omnibus) shown as distribution-free support only.")


if __name__ == "__main__":
    main()

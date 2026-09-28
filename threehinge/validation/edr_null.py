#!/usr/bin/env python3
"""Distance-constrained (EDR-style) null for the hinge network.

Connection probability falls off steeply with distance (the exponential distance rule), so a
connectivity excess is only evidence of specific organization if it survives a null that keeps
the wiring-distance structure. Here every 3HG-3HG node pair is matched to gyral-crown node pairs
from the SAME specimen with
    - the same unordered DK parcel pair (source and target region matched; fallback to the same
      lobe pair, then to distance alone, when the parcel pair has no crown counterpart),
    - the same inter-node Euclidean distance (+- TOL mm),
    - identical patch geometry by construction (every node is the same 3 mm sub-WM sphere, and
      each 3HG node has a same-parcel crown partner, so the null pool is opportunity-matched).
The null connection probability for a 3HG pair is the mean connection indicator over its matched
crown pairs, and

    E_HH = (# observed 3HG-3HG connections) / (sum of matched-null probabilities).

The same construction with the seed-normalized streamline fraction F instead of the binary
indicator is reported as a
sensitivity. Specimen is the unit (n = 6); the six log E_HH are tested against zero."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
import dk
TR = (_env.ROOT + "/surf/three_hinge/tracto")
VAL = (_env.ROOT + "/surf/three_hinge/validate")
SUBJ = [1, 2, 3, 4, 5, 6]; TOL = 5.0
LOBE = {"frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                    "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
                    "precentral", "paracentral", "frontalpole"],
        "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
        "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                     "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"],
        "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
        "cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate",
                      "isthmuscingulate"],
        "insula": ["insula"]}
NAME2LOBE = {n: l for l, ns_ in LOBE.items() for n in ns_}


ID2NAME = dk.id2name()                               # {DK id: parcel name}


def lobe_of(pid):
    name = ID2NAME.get(int(pid), ID2NAME.get(int(pid) - 2000, ""))
    return NAME2LOBE.get(str(name).replace("ctx-rh-", ""), "other")


def run(sid, weighted=False):
    z = np.load(f"{TR}/Subject{sid}/conn_matrix.npz")
    kind, parc, D = z["kind"], z["parc"], z["nodedist"]
    # Mb, not M: with SIFT2 weights a connection can carry an arbitrarily small number, so
    # (M > 0) would test floating point rather than whether a streamline exists. Mb is the
    # unweighted count of the same streamlines. Falls back to M for tractograms saved before
    # weights existed.
    CNT = z["Mb"] if "Mb" in z.files else z["M"]
    A = ((CNT + CNT.T) > 0).astype(float)
    # the weighted variant must use F, the seed-normalized fraction: 3HG patches carry up to 25%
    # more seeded streamlines than their crown partners, so raw counts would score that difference
    # as connectivity.
    val = (z["F"] + z["F"].T) if weighted else A
    lob = np.array([lobe_of(p) for p in parc])

    hh_i, hh_j = np.triu_indices(len(kind), 1)
    is_hh = (kind[hh_i] == 0) & (kind[hh_j] == 0)
    is_cc = (kind[hh_i] == 1) & (kind[hh_j] == 1)

    def keys(idx_i, idx_j, f):
        return np.array(["|".join(sorted((str(f[a]), str(f[b])))) for a, b in zip(idx_i, idx_j)])

    ci, cj = hh_i[is_cc], hh_j[is_cc]
    cD, cV = D[ci, cj], val[ci, cj]
    cP = keys(ci, cj, parc)
    cL = keys(ci, cj, lob)

    hi, hj = hh_i[is_hh], hh_j[is_hh]
    hD, hV = D[hi, hj], val[hi, hj]
    hP = keys(hi, hj, parc)
    hL = keys(hi, hj, lob)

    exp, tier = np.zeros(len(hi)), np.zeros(len(hi), int)
    for k in range(len(hi)):
        m = (cP == hP[k]) & (np.abs(cD - hD[k]) <= TOL)
        if m.sum() >= 3:
            tier[k] = 1
        else:
            m = (cL == hL[k]) & (np.abs(cD - hD[k]) <= TOL)
            if m.sum() >= 3:
                tier[k] = 2
            else:
                m = np.abs(cD - hD[k]) <= TOL / 2
                tier[k] = 3
        exp[k] = cV[m].mean() if m.any() else np.nan
    ok = np.isfinite(exp)
    E = hV[ok].sum() / max(exp[ok].sum(), 1e-12)
    return E, hV[ok].sum(), exp[ok].sum(), np.bincount(tier[ok], minlength=4)[1:]


def unmatched(sid, weighted=False):
    """The SAME estimator and the SAME contrast with the matching switched off: every 3HG-3HG pair
    against the pooled mean over all crown-crown pairs. This is the only legitimate reference for
    E_HH -- comparing E_HH with the seed-level target contrast of Fig. 4 would change the
    estimator, the comparison class and the matching all at once."""
    z = np.load(f"{TR}/Subject{sid}/conn_matrix.npz")
    kind = z["kind"]
    C2 = z["Mb"] if "Mb" in z.files else z["M"]
    val = (z["F"] + z["F"].T) if weighted else ((C2 + C2.T) > 0).astype(float)
    i, j = np.triu_indices(len(kind), 1)
    v = val[i, j]
    hh = (kind[i] == 0) & (kind[j] == 0)
    cc = (kind[i] == 1) & (kind[j] == 1)
    return v[hh].mean() / v[cc].mean()


def calibrate(sid, nrep=20):
    """Feed the machinery a KNOWN-null case: a random half of the crown pairs plays the 3HG role
    and is matched against the other half. Unbiased matching must return E ~ 1."""
    z = np.load(f"{TR}/Subject{sid}/conn_matrix.npz")
    kind, parc, D = z["kind"], z["parc"], z["nodedist"]
    C3 = z["Mb"] if "Mb" in z.files else z["M"]
    A = ((C3 + C3.T) > 0).astype(float)
    lob = np.array([lobe_of(p) for p in parc])
    ii, jj = np.triu_indices(len(kind), 1)
    cc = np.where((kind[ii] == 1) & (kind[jj] == 1))[0]
    ci, cj = ii[cc], jj[cc]
    K = np.array(["|".join(sorted((str(parc[a]), str(parc[b])))) for a, b in zip(ci, cj)])
    L = np.array(["|".join(sorted((str(lob[a]), str(lob[b])))) for a, b in zip(ci, cj)])
    Dv, Av = D[ci, cj], A[ci, cj]
    rng = np.random.RandomState(sid); out = []
    for _ in range(nrep):
        half = rng.rand(len(cc)) < 0.5
        exp = np.zeros(half.sum()); hK, hL, hD = K[half], L[half], Dv[half]
        oK, oL, oD, oA = K[~half], L[~half], Dv[~half], Av[~half]
        for k in range(half.sum()):
            m = (oK == hK[k]) & (np.abs(oD - hD[k]) <= TOL)
            if m.sum() < 3:
                m = (oL == hL[k]) & (np.abs(oD - hD[k]) <= TOL)
                if m.sum() < 3:
                    m = np.abs(oD - hD[k]) <= TOL / 2
            exp[k] = oA[m].mean() if m.any() else np.nan
        ok = np.isfinite(exp)
        out.append(Av[half][ok].sum() / max(exp[ok].sum(), 1e-12))
    return float(np.mean(out)), float(np.std(out, ddof=1))


def main():
    res = {}
    for weighted, lab in ((False, "binary connection indicator"), ((True), "streamline-count weighted")):
        print(f"\n=== E_HH vs distance+region+opportunity matched null ({lab}) ===")
        print(f"{'':6s}{'observed':>9s}{'expected':>10s}{'E_HH':>7s}{'match tier 1/2/3 (%)':>22s}")
        Es, Us = [], []
        for sid in SUBJ:
            E, o, e, t = run(sid, weighted)
            Es.append(E); Us.append(unmatched(sid, weighted))
            tp = 100 * t / t.sum()
            print(f"S{sid:<5d}{o:9.0f}{e:10.1f}{E:7.2f}{tp[0]:10.0f} /{tp[1]:3.0f} /{tp[2]:3.0f}")
        Es, Us = np.array(Es), np.array(Us)
        res["E_weighted" if weighted else "E_binary"] = Es
        res["U_weighted" if weighted else "U_binary"] = Us
        t_, p_ = stats.ttest_1samp(np.log(Es), 0)
        tu, pu = stats.ttest_1samp(np.log(Us), 0)
        td, pd = stats.ttest_rel(np.log(Es), np.log(Us))
        print(f"group  geometric mean {np.exp(np.log(Es).mean()):.2f}x   "
              f"t(5) = {t_:.2f}, P = {p_:.4f}, {int((Es > 1).sum())}/6 > 1")
        print(f"       unmatched reference {np.exp(np.log(Us).mean()):.2f}x  t(5) = {tu:.2f}, "
              f"P = {pu:.4f}; matching raises it in {int((Es > Us).sum())}/6, "
              f"paired t(5) = {td:.2f}, P = {pd:.4f}")
    return res


if __name__ == "__main__":
    res = main()
    print("\n=== calibration: crown half vs crown half (expect E ~ 1) ===")
    cal = {}
    for sid in SUBJ:
        m, sd = calibrate(sid)
        cal[sid] = m
        print(f"S{sid}: E = {m:.3f} +- {sd:.3f}")
    np.savez(f"{VAL}/edr_null.npz", subj=np.array(SUBJ), cal=np.array([cal[s] for s in SUBJ]), **res)
    print("\n=== bias-corrected E_HH (binary), per-specimen E / calibration E ===")
    Ec = np.array([run(sid, False)[0] / cal[sid] for sid in SUBJ])
    t_, p_ = stats.ttest_1samp(np.log(Ec), 0)
    print("  " + "  ".join(f"S{s}={e:.2f}" for s, e in zip(SUBJ, Ec)))
    print(f"  geometric mean {np.exp(np.log(Ec).mean()):.2f}x   t(5) = {t_:.2f}, P = {p_:.4f}, "
          f"{int((Ec > 1).sum())}/6 > 1")

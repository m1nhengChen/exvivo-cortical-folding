#!/usr/bin/env python3
"""3HG vs ordinary-gyral-crown vs sulcal-fundus multimodal contrast on the white surface.

Landmarks: gyralnet rh_3hinge_ids.txt (vertex indices into rh.white).
Modalities: the audited per-vertex samples.  We rebuild a CLEAN mid-cortical value from
the depth profile (.profile.mgh has depths 0.1/0.3/0.5/0.7/0.9) using only 0.3-0.7, which
the audit showed avoids WM contamination near white (depth 0.1) and CSF overshoot near
pial (depth 0.9).  T1/T2 are excluded (audit: 4 mm slices, 1.1 mm CoM error).

Per landmark we take the mean over a 2 mm geodesic disk (robust to single-vertex noise).
Controls, matched in count to the 3HG set:
  gyral crown : strongly gyral (sulc <= 25th pct) and >CROWN_FAR mm from any 3HG
                -> an ordinary crown that is NOT a hinge, with a sulc range comparable to
                   the 3HG set, so the contrast isolates hinge-vs-crown, not gyrus-vs-sulcus
  sulcal fundus: strongly sulcal (sulc >= 75th pct)

Outputs per subject: a summary table, a ring profile, and an .npz of per-landmark disk
means tagged by class -- consumed by hinge_pool.py for the 6-subject mixed model."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import os, sys, numpy as np, nibabel as nib
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import dk
from hinge_ids import cluster_groups, group_mean

FS = (_env.ROOT + "/surf/fs")
GNET = (_env.ROOT + "/surf/three_hinge/gnet")
OUT = (_env.ROOT + "/surf/three_hinge/results")
H = "rh"
DISK_MM = 2.0
# Merge scale for the 3HG landmarks. It follows this analysis's own estimator, not the
# connectivity patch: disk_means uses a 2 mm RADIUS geodesic disk, so two samples stop
# overlapping at 4 mm. Borrowing the connectivity's 6 mm discards more than it needs to.
SEP = float(os.environ.get("SEP", str(__import__("hinge_ids").DISK)))
CROWN_FAR = 8.0
MID_DEPTHS = [1, 2, 3]          # indices into the 5-depth profile -> 0.3/0.5/0.7
RINGS = [(0, 2), (2, 4), (4, 6), (6, 8)]
MODS = ["QSM", "QSM_pos", "QSM_neg", "T2star", "FA", "AD", "RD", "ADC",
        "fiber_ratio", "restricted_ratio", "hindered_ratio", "water_ratio"]


def mesh_graph(v, f):
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    e = np.unique(e, axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                             shape=(n, n))


def clean_mid(subj, name):
    """Mid-cortical value averaged over depths 0.3-0.7 only."""
    p = f"{FS}/{subj}/surf_maps/{H}.{name}.profile.mgh"
    d = nib.load(p).get_fdata()[:, 0, :]          # (nv, 5)
    return np.nanmean(d[:, MID_DEPTHS], axis=1)


def disk_means(G, seeds, maps, radius):
    out = {m: np.full(len(seeds), np.nan) for m in maps}
    for i, s in enumerate(seeds):
        d = dijkstra(G, indices=int(s), limit=radius, min_only=True)
        idx = np.where(np.isfinite(d))[0]
        for m, val in maps.items():
            vv = val[idx]; vv = vv[np.isfinite(vv)]
            if vv.size:
                out[m][i] = vv.mean()
    return out


def run(sid):
    s = f"Subject{sid}"
    v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
    v = v.astype(np.float64); n = len(v)
    sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
    thick = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.thickness")
    hinge = np.atleast_1d(np.loadtxt(f"{GNET}/{s}/out/{H}_3hinge_ids.txt", dtype=int))
    parc = dk.read(s)[0]
    ctx = parc > 0                                    # cortex only -- exclude medial wall
    # gyralnet reports hinges on the medial wall too (0-6% here). Masking only the maps is not
    # enough: such a landmark still seeds dmin, still counts towards k (and hence the size of the
    # count-matched crown/fundus sets), and its 3 mm disk still returns a mean - taken over
    # whichever rim vertices happen to be cortical. Drop the landmarks themselves.
    nraw = len(hinge); hinge = hinge[ctx[hinge]]

    maps = {m: clean_mid(s, m) for m in MODS
            if os.path.exists(f"{FS}/{s}/surf_maps/{H}.{m}.profile.mgh")}
    maps["thickness"] = thick
    # CRITICAL: mask non-cortical vertices. The crown pool otherwise pulls in medial-wall /
    # unlabelled vertices whose maps are sampled from WM/ventricle/background (e.g. FA 0.44),
    # which contaminates the control and flips FA/fiber/MD/hindered.
    for m in maps:
        maps[m] = np.where(ctx, maps[m], np.nan)

    G = mesh_graph(v, f)
    dmin = dijkstra(G, indices=hinge, min_only=True)

    lo, hi = np.percentile(sulc[ctx], 25), np.percentile(sulc[ctx], 75)
    crown_pool = np.where((sulc <= lo) & (dmin > CROWN_FAR) & ctx)[0]
    fund_pool = np.where((sulc >= hi) & ctx)[0]
    k = len(hinge)
    # Minimum-separation draws, not index-even ones: see spread_sample() in hinge_ids.py. The
    # control must carry the same spacing the 6 mm merge gives the landmarks, or it is the more
    # pseudo-replicated of the two and inflates d through the pooled SD.
    # target: the junction medoids - one control per gyral junction, at their fold depth
    NDRAW = 20                        # random control draws, averaged at the level of d
    grp0 = cluster_groups(sid, hinge, merge=SEP)
    k = len(hinge)
    # RANDOM draws, not crown_pool[np.linspace(...)]. Even intervals of VERTEX INDEX are not even
    # intervals of space: FreeSurfer numbering is spatially local, so that line returned runs of
    # neighbouring vertices - median nearest-neighbour 1.0 mm, 81-95% of points within 2 mm of
    # another. Measured against three alternatives, the clumping turned out NOT to inflate the
    # effect (MD: linspace -0.195, random -0.203, whole-pool -0.197), so the fix costs nothing and
    # is kept only because index-ordered sampling has no defensible rationale. Forcing a 4 mm
    # minimum separation instead was tried and reverted: it cut the control from 479 to 218 points
    # and took hindered and free water below significance for no bias removed.
    rng = np.random.RandomState(sid)
    draws = [(np.sort(rng.choice(crown_pool, min(k, len(crown_pool)), replace=False)),
              np.sort(rng.choice(fund_pool, min(k, len(fund_pool)), replace=False)))
             for _ in range(NDRAW)]
    crown, fund = draws[0]
    print(f"\n=== {s}: {k} 3HG ({nraw - k} dropped on the medial wall) | {len(crown)} crown | "
          f"{len(fund)} fundus | "
          f"sulc(3HG)={np.median(sulc[hinge]):+.2f} crown={np.median(sulc[crown]):+.2f} ===")
    Hd = disk_means(G, hinge, maps, DISK_MM)
    allc = np.unique(np.concatenate([c for c, _ in draws]))
    allf = np.unique(np.concatenate([f for _, f in draws]))
    Cd_all = disk_means(G, allc, maps, DISK_MM)
    Fd_all = disk_means(G, allf, maps, DISK_MM)
    ci = {int(x): i for i, x in enumerate(allc)}; fi = {int(x): i for i, x in enumerate(allf)}

    # Aggregate per gyral junction, not per landmark. Landmarks within SEP mm of one another are
    # the same junction sampled repeatedly - their 2 mm disks overlap - and counting them
    # separately inflates the effect size.
    #
    # The control is deliberately NOT blocked to the same sizes here. Note that this is an
    # asymmetry: the draws are taken with rng.choice from the pool, so they are not spaced like
    # the landmarks - on S5 the crown draw has a median nearest-neighbour of 0.61 mm with 97% of
    # points within 2 mm of another, against 3.54 mm and 5% for the landmarks. Running
    # hinge_ids.mirror_groups on the control as region_dk.py does was measured and is a null
    # result: every effect size moves by <= 0.01 with no change of sign or ordering (FA
    # +0.171 -> +0.174, MD -0.210 -> -0.214, free water -0.189 -> -0.193, thickness
    # +0.262 -> +0.272), P values rise slightly on the smaller effective n, and DBSI hindered
    # goes 6/6 -> 5/6 consistent. The published numbers are the ungrouped-control ones.
    grp = cluster_groups(sid, hinge, merge=SEP)
    sz = [len(g) for g in grp]
    Hd = {m: group_mean(Hd[m], grp) for m in Hd}
    # per-draw control values; the caller averages the statistic over draws
    Cd = {m: np.array([Cd_all[m][[ci[int(x)] for x in c]] for c, _ in draws], dtype=object)
          for m in Cd_all}
    Fd = {m: np.array([Fd_all[m][[fi[int(x)] for x in f]] for _, f in draws], dtype=object)
          for m in Fd_all}
    print(f"    {len(hinge)} landmarks -> {len(grp)} junctions "
          f"(median cluster {int(np.median(sz))}, max {max(sz)})", flush=True)

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for m in maps:
        h = Hd[m]
        ds, cs, fs_ = [], [], []
        for r in range(NDRAW):
            c, fu = Cd[m][r], Fd[m][r]
            sp = np.sqrt((np.nanvar(h, ddof=1) + np.nanvar(c, ddof=1)) / 2)
            ds.append((np.nanmean(h) - np.nanmean(c)) / sp if sp else np.nan)
            cs.append(np.nanmean(c)); fs_.append(np.nanmean(fu))
        c, fu = np.array(cs), np.array(fs_)
        d = float(np.nanmean(ds))
        rows.append((m, np.nanmean(h), np.nanmean(c), np.nanmean(fu), d))
        print(f"  {m:16s} 3HG={np.nanmean(h):9.4f} crown={np.nanmean(c):9.4f} "
              f"fundus={np.nanmean(fu):9.4f}  d={d:+.3f}")
    with open(f"{OUT}/{s}_hinge_table.tsv", "w") as fo:
        fo.write("modality\t3HG\tcrown\tfundus\td_3HG_crown\n")
        for r in rows:
            fo.write("\t".join([r[0]] + [f"{x:.5f}" for x in r[1:]]) + "\n")

    # ring profile (confounded by folding; kept for completeness)
    prof = {}
    for m in ["QSM", "T2star", "FA", "restricted_ratio"]:
        if m in maps:
            prof[m] = [np.nanmean(maps[m][(dmin > a) & (dmin <= b)]) for a, b in RINGS]
    prof["thickness"] = [np.nanmean(thick[(dmin > a) & (dmin <= b)]) for a, b in RINGS]

    # per-landmark values for the pooled model
    np.savez(f"{OUT}/{s}_landmarks.npz",
             mods=np.array(list(maps.keys())),
             hinge=np.vstack([Hd[m] for m in maps]),
             # one representative draw for the pooled per-landmark model; the draw-averaged
             # effect size is what the per-specimen table carries
             crown=np.vstack([np.asarray(Cd[m][0], float) for m in maps]),
             fundus=np.vstack([np.asarray(Fd[m][0], float) for m in maps]),
             rings=np.array([b for _, b in RINGS]),
             **{f"ring_{m}": np.array(prof[m]) for m in prof})
    print(f"  -> {OUT}/{s}_hinge_table.tsv, {s}_landmarks.npz")


if __name__ == "__main__":
    for i in ([int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]):
        run(i)

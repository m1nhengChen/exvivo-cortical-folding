#!/usr/bin/env python3
"""Individualized-geometry validation: does a specimen's OWN 3HG configuration align with its
structural wiring better than another specimen's 3HG configuration would?

For host specimen s, the hinge-network enrichment E(H, C_s) is computed with the published
estimator, once with the host's own 3HG set H_s and once with each donor's 3HG set H_s'
transferred onto the host cortex through the FreeSurfer spherical registration (sphere.reg is
fsaverage-aligned, so nearest-vertex transfer s' -> s is the standard surface mapping; transferred
vertices are deduplicated and medial-wall vertices dropped). Everything downstream - crown
matching, patch geometry, tolerances - is identical to validate/seed_control.py, and the host's
tractogram is fixed, so the ONLY thing that varies is whose folding configuration supplies the
landmarks.

Donors come from two cohorts. The five other DWI specimens (Subject1-6) are within-cohort. Six
FLASH ex-vivo hemispheres (Subject7-12, Dryad doi:10.5061/dryad.dbrv15fgk) have surfaces but no
diffusion data, so they can only ever be donors; they were reconstructed by a different pipeline
from a different acquisition, which makes them the stronger control - if a host's own configuration
still wins against them, the advantage cannot come from anything the DWI cohort shares.

The FLASH cohort yields systematically more 3HG landmarks than the DWI cohort (median 695 vs 440
on comparable meshes), and the target contrast does rise with landmark count (Spearman rho = +0.49
within host, positive in 6/6). That must be adjusted for, but NOT by subsampling every
configuration to a common count: the diagonal of this matrix is meant to BE the host's own
configuration, and a random subset of it is a different object, degraded by an amount that depends
on nothing but which unrelated donor happened to be smallest. Instead the count is regressed out -
within each host, log E is fitted on log(landmark count) across all twelve configurations and own
is ranked by its residual. The primary test is then distribution-free: under the null the own
residual rank is uniform on 1..12 in each host, so the sum of the six ranks has an exact null
distribution.

Reported per host: E_own, the eleven E_foreign, and the rank of own among the twelve. Group test:
paired t on log E, own vs mean-foreign, n = 6 hosts, and separately against each donor cohort. Both
the target contrast (f_hh/f_hc) and the coverage-free seed contrast (f_hh/f_ch) are computed; the
latter is the safer comparison because donor hinge sets differ in size and hence in target
coverage."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, time, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
from scipy import sparse, stats
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
import dk
FS = (_env.ROOT + "/surf/fs"); GNET = (_env.ROOT + "/surf/three_hinge/gnet")
TR = (_env.ROOT + "/surf/three_hinge/tracto")
OUT = (_env.ROOT + "/surf/three_hinge/validate")
H = "rh"; R = 3.0; INWARD = 2.0; NEAR = 4.0
SUBJ = [1, 2, 3, 4, 5, 6]                     # hosts: the specimens that have tractography
FLASH = [7, 8, 9, 10, 11, 12]                 # donor-only: FLASH surfaces, no diffusion
DONOR = SUBJ + FLASH


def mesh_graph(v, f):
    e = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1); n = len(v)
    return sparse.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))


_PARC = {}


def donor_parc(sid):
    if sid not in _PARC:
        _PARC[sid] = dk.read(f"Subject{sid}")[0]
    return _PARC[sid]


def sphere(sid):
    v, _ = nib.freesurfer.read_geometry(f"{FS}/Subject{sid}/surf/{H}.sphere.reg")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Host:
    def __init__(self, sid):
        s = f"Subject{sid}"; self.sid = sid
        dwi = nib.load(f"{TR}/{s}/dwi.nii.gz"); self.dinv = np.linalg.inv(dwi.affine)
        self.shp = dwi.shape[:3]
        trk = load_tractogram(f"{TR}/{s}/whole_brain.trk", dwi, bbox_valid_check=False)
        trk.to_space(Space.RASMM); SL = trk.streamlines
        self.ep0 = np.array([sl[0] for sl in SL]); self.ep1 = np.array([sl[-1] for sl in SL])
        self.vox2sl = defaultdict(set)
        for i, sl in enumerate(SL):
            for a, b, c in np.round(nib.affines.apply_affine(self.dinv, sl)).astype(int):
                if 0 <= a < self.shp[0] and 0 <= b < self.shp[1] and 0 <= c < self.shp[2]:
                    self.vox2sl[(a, b, c)].add(i)
        rr = int(np.ceil(R)); zoom = np.mean(dwi.header.get_zooms()[:3])
        self.off = [(a, b, c) for a in range(-rr, rr + 1) for b in range(-rr, rr + 1)
                    for c in range(-rr, rr + 1) if a * a + b * b + c * c <= (R / zoom) ** 2]

        mgh = nib.load(f"{FS}/{s}/mri/wm.mgz")
        tk = mgh.header.get_affine() @ np.linalg.inv(mgh.header.get_vox2ras_tkr())
        v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/{H}.white")
        import pyvista as pv
        m = pv.PolyData(v.astype(float), np.c_[np.full(len(f), 3), f].ravel())
        m.compute_normals(cell_normals=False, point_normals=True, inplace=True)
        nrm = (tk[:3, :3] @ m.point_data["Normals"].T).T
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        vw = nib.affines.apply_affine(tk, v)
        if np.mean(np.sum((vw - vw.mean(0)) * nrm, 1)) < 0:
            nrm = -nrm
        self.patch_w = vw - INWARD * nrm
        self.G = mesh_graph(v.astype(float), f)
        self.sulc = nib.freesurfer.read_morph_data(f"{FS}/{s}/surf/{H}.sulc")
        self.parc = dk.read(s)[0]; self.ctx = self.parc > 0
        self.sph_tree = cKDTree(sphere(sid))
        self._ids = {}

    def transfer(self, donor):
        """donor 3HG vertex ids -> host vertex ids via the fsaverage-aligned spheres.

        The medial wall is masked on BOTH sides. GyralNet runs on the whole surface and happily
        reports hinges on the medial wall, where there is no real folding - 0-6% of the landmarks
        in the DWI cohort but 7-17% in the FLASH cohort. Dropping them only after transfer, by the
        host's parcellation, is not enough: the two subjects' medial-wall boundaries do not
        coincide on the sphere, so a donor's medial-wall hinge can land just inside the host's
        cortex, on the callosal rim where streamlines are dense. Masking on the donor first makes
        every configuration a set of genuine cortical landmarks before it is compared."""
        hv = np.loadtxt(f"{GNET}/Subject{donor}/out/{H}_3hinge_ids.txt", dtype=int)
        hv = hv[donor_parc(donor)[hv] > 0]                    # donor-side medial wall
        if donor == self.sid:
            out = hv
        else:
            _, idx = self.sph_tree.query(sphere(donor)[hv])
            out = np.unique(idx)
        return out[self.parc[out] > 0]                        # host-side medial wall

    def ids_of(self, vid):
        if vid not in self._ids:
            cij = np.round(nib.affines.apply_affine(self.dinv, self.patch_w[vid])).astype(int)
            S = set()
            for o in self.off:
                S |= self.vox2sl.get((cij[0] + o[0], cij[1] + o[1], cij[2] + o[2]), set())
            self._ids[vid] = list(S)
        return self._ids[vid]

    def enrich(self, hinge, seed_rng):
        dmin = dijkstra(self.G, indices=hinge.tolist(), min_only=True)
        lo = np.percentile(self.sulc[self.ctx], 25)
        pool = (self.sulc <= lo) & (dmin > 6) & self.ctx
        by = {pid: np.where(pool & (self.parc == pid))[0] for pid in np.unique(self.parc[hinge])}
        crown = np.array([by[self.parc[h]][seed_rng.randint(len(by[self.parc[h]]))]
                          if len(by[self.parc[h]]) else -1 for h in hinge])
        ok = crown >= 0; hm, cm = hinge[ok], crown[ok]
        th = cKDTree(self.patch_w[hm]); tc = cKDTree(self.patch_w[cm])
        nh0 = th.query(self.ep0)[0] < NEAR; nh1 = th.query(self.ep1)[0] < NEAR
        nc0 = tc.query(self.ep0)[0] < NEAR; nc1 = tc.query(self.ep1)[0] < NEAR

        def frac(seeds, n0, n1):
            vals = []
            for vid in seeds:
                ids = self.ids_of(vid)
                if len(ids) < 3:
                    continue
                cw = self.patch_w[vid]
                far0 = np.linalg.norm(self.ep0[ids] - cw, axis=1) > np.linalg.norm(self.ep1[ids] - cw, axis=1)
                vals.append(np.mean(np.where(far0, n0[ids], n1[ids])))
            return np.nanmean(vals)

        f_hh = frac(hm, nh0, nh1); f_hc = frac(hm, nc0, nc1); f_ch = frac(cm, nh0, nh1)
        return f_hh / f_hc, f_hh / f_ch, len(hm)


def resid_rank(E, N):
    """Own's rank among all configurations after log E is regressed on log(landmark count).
    Regressing beats subsampling: nothing is discarded and the diagonal stays the real own set."""
    out = []
    for i in range(len(E)):
        x, y = np.log(N[i]), np.log(E[i])
        out.append(int(stats.rankdata(-(y - np.polyval(np.polyfit(x, y, 1), x)))[i]))
    return out


def exact_rank_p(ranks, k):
    """Exact P(sum of ranks <= observed) under independent Uniform{1..k} ranks per host."""
    d = np.zeros(k + 1); d[1:] = 1.0 / k
    tot = np.array([1.0])
    for _ in range(len(ranks)):
        tot = np.convolve(tot, d)
    return int(sum(ranks)), float(tot[:int(sum(ranks)) + 1].sum())


def main():
    nd = len(DONOR)
    Epub = np.full((6, nd), np.nan); Eseed = np.full((6, nd), np.nan)   # [host, donor]
    NH = np.full((6, nd), np.nan)                                       # transferred hinge count
    for hi, hs in enumerate(SUBJ):
        t0 = time.time(); host = Host(hs)
        for di, ds in enumerate(DONOR):
            ep, es, n = host.enrich(host.transfer(ds), np.random.RandomState(97 * hs + ds))
            Epub[hi, di], Eseed[hi, di], NH[hi, di] = ep, es, n
            tag = " (own)" if ds == hs else (" [FLASH]" if ds in FLASH else "")
            print(f"host S{hs} <- donor S{ds}{tag:9s}: n={n:4d}  E_target={ep:5.2f}  "
                  f"E_seed={es:5.2f}", flush=True)
        print(f"  host S{hs} done in {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{OUT}/own_vs_foreign.npz", Epub=Epub, Eseed=Eseed, nhinge=NH,
             subj=np.array(SUBJ), donor=np.array(DONOR), flash=np.array(FLASH))

    dsel = {"all 11 donors": [j for j in range(nd)],
            "5 DWI donors": [j for j in range(len(SUBJ))],
            "6 FLASH donors": [j for j in range(len(SUBJ), nd)]}
    for name, E in (("target contrast (f_hh/f_hc)", Epub), ("seed contrast (f_hh/f_ch)", Eseed)):
        own = np.array([E[i, i] for i in range(6)])
        print(f"\n=== {name} ===")
        print("  own    :", "  ".join(f"{v:5.2f}" for v in own))
        for lab, cols in dsel.items():
            fore = np.array([np.nanmean([E[i, j] for j in cols if j != i]) for i in range(6)])
            t, p = stats.ttest_rel(np.log(own), np.log(fore))
            print(f"  vs {lab:15s}", "  ".join(f"{v:5.2f}" for v in fore),
                  f"| own > foreign {int((own > fore).sum())}/6 | t(5) = {t:5.2f}, P = {p:.4f}")
        rho = [stats.spearmanr(NH[i], E[i]).statistic for i in range(6)]
        print(f"  Spearman(landmark count, E) within host: " +
              " ".join(f"{v:+.2f}" for v in rho) + f" | mean {np.mean(rho):+.2f}")
        raw = [int(stats.rankdata(-E[i])[i]) for i in range(6)]
        res = resid_rank(E, NH)
        print(f"  rank of own among {nd}:      raw {raw}   count-adjusted {res}")
        srank, prank = exact_rank_p(res, nd)
        print(f"  PRIMARY exact rank test on the count-adjusted ranks: sum = {srank}, "
              f"null mean {6 * (nd + 1) / 2:.0f}, one-sided P = {prank:.4f}")


if __name__ == "__main__":
    main()

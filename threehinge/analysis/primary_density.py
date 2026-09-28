#!/usr/bin/env python3
"""Are 3HGs denser outside primary cortices? Density = junctions per cm2 of cortical white
surface, primary (precentral, postcentral, pericalcarine, transversetemporal - the DK proxies
for M1, S1, V1, A1) vs the rest of the cortex, per hemisphere; the test is a one-sample t on
the per-hemisphere log density ratio. Usage:
    primary_density.py --exvivo
    primary_density.py --invivo SESSION [SESSION...]   (appends one npz per session)
"""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

import sys, os, numpy as np, nibabel as nib, warnings; warnings.filterwarnings("ignore")

PRIM = ("precentral", "postcentral", "pericalcarine", "transversetemporal")


def vertex_areas(v, f):
    a = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ta = 0.5 * np.linalg.norm(a, axis=1)
    va = np.zeros(len(v))
    for k in range(3):
        np.add.at(va, f[:, k], ta / 3)
    return va


def tally(names, lab, ctx, va, hids):
    """(n, area_mm2) for primary and non-primary cortex"""
    isprim_name = np.array([any(n.endswith(p) for p in PRIM) for n in names])
    vprim = isprim_name[lab] & ctx
    vnon = ~isprim_name[lab] & ctx
    return (int(vprim[hids].sum()), float(va[vprim].sum()),
            int(vnon[hids].sum()), float(va[vnon].sum()))


def exvivo():
    FS = (_env.ROOT + "/surf/fs")
    GNET = (_env.ROOT + "/surf/three_hinge/gnet")
    out = []
    for sid in [1, 2, 3, 4, 5, 6]:
        s = f"Subject{sid}"
        v, f = nib.freesurfer.read_geometry(f"{FS}/{s}/surf/rh.white")
        lab, _, names = nib.freesurfer.read_annot(f"{FS}/{s}/label/rh.aparc.annot")
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        bad = {i for i, n in enumerate(names) if n in ("unknown", "corpuscallosum")}
        ctx = ~np.isin(lab, list(bad)) & (lab >= 0)
        hids = np.atleast_1d(np.loadtxt(f"{GNET}/{s}/out/rh_3hinge_ids.txt", dtype=int))
        hids = hids[ctx[hids]]
        va = vertex_areas(v.astype(float), f)
        np_, ap, nn, an = tally(names, lab, ctx, va, hids)
        out.append((np_, ap, nn, an))
        print(f"{s}: primary {np_} 3HG / {ap/100:.1f} cm2 = {np_/ap*100:.3f}/cm2   "
              f"non-primary {nn} / {an/100:.1f} cm2 = {nn/an*100:.3f}/cm2   "
              f"ratio {(nn/an)/(np_/ap):.2f}", flush=True)
    np.savez("results/primary_density_exvivo.npz", data=np.array(out))


def invivo(sessions):
    import pyvista as pv
    ROOT = (_env.AABC + "/organized")
    DERIV = (_env.AABC + "/derived")
    for ses in sessions:
        S = f"{ROOT}/{ses}"
        row = np.zeros(4)
        try:
            for h in ("lh", "rh"):
                G = f"{S}/postprocessing_tong/gii_native"
                white = pv.read(f"{G}/surf/{h}.white.resample.163842.vtk")
                v = np.asarray(white.points, float)
                f = white.faces.reshape(-1, 4)[:, 1:]
                lab, _, names = nib.freesurfer.read_annot(
                    f"{G}/label/{h}.aparc.resample.163842.annot")
                names = [n.decode() if isinstance(n, bytes) else n for n in names]
                bad = {i for i, n in enumerate(names) if n in ("unknown", "corpuscallosum")}
                ctx = ~np.isin(lab, list(bad)) & (lab >= 0)
                hids = np.atleast_1d(np.loadtxt(
                    f"{G}/gyralnet_island_resample_163842_scanner/{h}_3hinge_ids.txt", dtype=int))
                hids = hids[ctx[hids]]
                va = vertex_areas(v, f)
                np_, ap, nn, an = tally(names, lab, ctx, va, hids)
                row += np.array([np_, ap, nn, an])
        except Exception as e:
            print(f"{ses} FAILED {type(e).__name__}", flush=True); continue
        os.makedirs(f"{DERIV}/{ses}", exist_ok=True)
        np.save(f"{DERIV}/{ses}/primary_density.npy", row)
        print(f"{ses} {int(row[0])}/{row[1]/100:.0f} vs {int(row[2])}/{row[3]/100:.0f} "
              f"ratio {(row[2]/row[3])/(row[0]/row[1]):.2f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--exvivo":
        exvivo()
    else:
        invivo(sys.argv[2:])

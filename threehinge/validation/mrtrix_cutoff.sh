#!/bin/bash
# Calibrate the iFOD2 FOD amplitude cutoff for these ex-vivo FODs.
#
# MRtrix's default (0.1) is set for in-vivo CSD, where FODs are larger. Here the l0 term averages
# ~0.17 in the mask and the default terminates tracking almost immediately: a 20k-streamline probe
# comes back with a median length of 9.8 mm, shorter than the 6 mm minimum node separation, so most
# of the tractogram cannot connect two nodes at all and only registers proximity.
#
# This sweeps the cutoff on a small -select so the length distribution can be read per specimen,
# BEFORE any contrast is computed from it. The choice must come from the tractogram looking like a
# tractogram, not from which value makes the enrichment largest.
set -e
# MRtrix3 must be on PATH, or MRTRIX_BIN point at its bin directory
[ -n "${MRTRIX_BIN:-}" ] && export PATH="$MRTRIX_BIN:$PATH"
_PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
ROOT_="${EXVIVO_ROOT:-$_PROJECT}"
TR=$ROOT_/surf/three_hinge/tracto
PY=${PYTHON:-python}
N=${N:-20000}
for S in "$@"; do
  w=$TR/Subject$S/mrtrix
  for CUT in 0.10 0.07 0.05 0.03 0.02; do
    f=$w/probe_$CUT.tck
    [ -f $f ] || tckgen $w/fod.mif $f -algorithm iFOD2 -seed_image $w/wm.mif \
        -select $N -cutoff $CUT -minlength 2 -maxlength 300 -quiet -force
  done
done
TR=$TR $PY - "$@" <<'PYEOF'
import sys, numpy as np, nibabel as nib
import os; TR=os.environ["TR"]
CUTS=["0.10","0.07","0.05","0.03","0.02"]
print(f"{'cutoff':>7s}" + "".join(f"{'S'+s:>26s}" for s in sys.argv[1:]))
print(f"{'':>7s}" + "".join(f"{'median  >20mm  >50mm  seeds':>26s}" for _ in sys.argv[1:]))
rows={}
for c in CUTS:
    cells=[]
    for s in sys.argv[1:]:
        w=f"{TR}/Subject{s}/mrtrix/probe_{c}.tck"
        t=nib.streamlines.load(w)
        sl=t.streamlines
        L=np.array([np.linalg.norm(np.diff(np.asarray(x),axis=0),axis=1).sum() for x in sl])
        tot=int(t.header.get("total_count", 0) or 0)
        cells.append(f"{np.median(L):7.1f}{100*(L>20).mean():6.0f}%{100*(L>50).mean():6.0f}%{tot:7d}")
        rows.setdefault(c,[]).append((np.median(L),(L>20).mean(),tot))
    print(f"{c:>7s}" + "".join(f"{x:>26s}" for x in cells))
print("\nseeds = tracking attempts needed for 20,000 accepted streamlines")
PYEOF

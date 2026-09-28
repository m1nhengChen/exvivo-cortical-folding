#!/bin/bash
# Independent-implementation check: MRtrix3 iFOD2 + SIFT2.
#
# Not a parameter variation of the published pipeline - different software, a different response
# estimator, a different tracking algorithm (probabilistic with second-order integration over the
# FOD, rather than first-order stepping between three discrete peaks) and, with SIFT2, a principled
# answer to the sampling-density question that the covariate analysis in FIGS_gyralbias could only
# argue about: SIFT2 gives each streamline a weight so that reconstructed streamline density
# matches the FOD lobe amplitudes, which is the quantity a per-node seed count only stands in for.
#
# Single-shell b6000 ex-vivo, so tournier single-tissue response and csd; no 5tt, matching the
# published run's lack of anatomical constraint.
set -e
# MRtrix3 must be on PATH, or MRTRIX_BIN point at its bin directory
[ -n "${MRTRIX_BIN:-}" ] && export PATH="$MRTRIX_BIN:$PATH"
_PROJECT="$(cd "$(dirname "$0")/../../.." && pwd)"
ROOT_="${EXVIVO_ROOT:-$_PROJECT}"
TR=$ROOT_/surf/three_hinge/tracto
ROOT=$ROOT_/28211843
PY=${PYTHON:-python}
NTRACK=${NTRACK:-400000}
# FOD amplitude cutoff, fixed at 0.05 by validate/mrtrix_cutoff.sh BEFORE any contrast was computed
# from these tractograms. MRtrix's default of 0.1 is set for in-vivo FOD amplitudes; on these
# ex-vivo FODs it terminates tracking so early that the median streamline is 6-26 mm depending on
# specimen - at or below the 6 mm minimum node separation, so most of the tractogram cannot connect
# two nodes and only registers proximity. Below 0.03 the median exceeds the length of the
# hemisphere itself, which is tracking through noise. 0.05 is the only value in between that gives
# a plausible length distribution (median 29-63 mm) AND the most consistent one across specimens
# (2.2x spread, against 4.4x at the default).
CUTOFF=${CUTOFF:-0.05}
for S in "$@"; do
  d=$TR/Subject$S; w=$d/mrtrix; mkdir -p $w
  echo "===== Subject$S ====="
  [ -f $w/bvecs_fsl.txt ] || $PY -c "
import numpy as np; np.savetxt('$w/bvecs_fsl.txt', np.loadtxt('$d/bvecs.txt').T, fmt='%.10f')"
  [ -f $w/dwi.mif ] || mrconvert $d/dwi.nii.gz $w/dwi.mif \
      -fslgrad $w/bvecs_fsl.txt $d/bvals.txt -datatype float32 -quiet -force
  [ -f $w/response.txt ] || dwi2response tournier $w/dwi.mif $w/response.txt \
      -mask $d/mask.nii.gz -quiet -force -scratch $w/tmp
  [ -f $w/fod.mif ] || dwi2fod csd $w/dwi.mif $w/response.txt $w/fod.mif \
      -mask $d/mask.nii.gz -lmax 8 -quiet -force
  [ -f $w/wm.mif ] || mrcalc $ROOT/Subject$S/Subject${S}_Mask/DWI_GM_WM_mask.nii.gz 2 -eq \
      $w/wm.mif -quiet -force
  [ -f $w/tracks.tck ] || tckgen $w/fod.mif $w/tracks.tck -algorithm iFOD2 \
      -seed_image $w/wm.mif -select $NTRACK -cutoff $CUTOFF -minlength 2 -maxlength 300 \
      -quiet -force
  [ -f $w/sift2.txt ] || tcksift2 $w/tracks.tck $w/fod.mif $w/sift2.txt -quiet -force
  echo "  tracks: $(tckinfo $w/tracks.tck 2>/dev/null | grep -i count | tr -s ' ' | head -2 | tr '\n' ' ')"
  echo "  sift2 weights: $(wc -l < $w/sift2.txt)"
done
echo "ALL DONE"

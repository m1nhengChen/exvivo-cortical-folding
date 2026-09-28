#!/bin/bash
# Re-place the white surface against the real image instead of the label-derived
# pseudo-T1.  Starts from rh.orig, which is already topology-corrected; since
# mris_place_surface only deforms, genus 0 is preserved without repeating the
# topology fix.
#   ./refine_white.sh Subject5 [brain.image.mgz]
set -e
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer}
source $FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
export SUBJECTS_DIR=${SUBJECTS_DIR:-${EXVIVO_FS:-$(cd "$(dirname "$0")/../.." && pwd)/surf/fs}}

S=$1
VOL=${2:-brain.image.mgz}
SUF=${3:-image}
H=rh
cd $SUBJECTS_DIR/$S

mris_autodet_gwstats --o surf/$H.autodet.${SUF}.dat \
    --i mri/$VOL --wm mri/wm.mgz --surf surf/$H.orig
mris_place_surface --adgws-in surf/$H.autodet.${SUF}.dat --wm mri/wm.mgz \
    --seg mri/aseg.presurf.mgz --threads ${THREADS:-4} --invol mri/$VOL \
    --$H --i surf/$H.orig --o surf/$H.white.${SUF} --white --nsmooth ${NSM:-5} ${EXTRA}
echo "### [$S] -> surf/$H.white.${SUF}"

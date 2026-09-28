#!/bin/bash
# Place the pial with FreeSurfer's deformable model against mri/brain.pialvol.mgz, the
# label map whose collapsed sulci have been cut open (pial_fs.py). mris_place_surface
# provides the tangential and normal spring terms needed to contract at the sulcal fundi.
#
#   ./pial_fs.sh Subject5
set -e
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer}
source $FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
export SUBJECTS_DIR=${SUBJECTS_DIR:-${EXVIVO_FS:-$(cd "$(dirname "$0")/../.." && pwd)/surf/fs}}

S=$1
H=rh
cd $SUBJECTS_DIR/$S

mris_autodet_gwstats --o surf/$H.autodet.pialvol.dat \
    --i mri/brain.pialvol.mgz --wm mri/wm.mgz --surf surf/$H.white
mris_place_surface --adgws-in surf/$H.autodet.pialvol.dat --wm mri/wm.mgz \
    --seg mri/aseg.presurf.mgz --threads ${THREADS:-4} --invol mri/brain.pialvol.mgz \
    --$H --i surf/$H.white --o surf/$H.pial.fs --pial --nsmooth 0 \
    --repulse-surf surf/$H.white --white-surf surf/$H.white
echo "### [$S] -> surf/$H.pial.fs"

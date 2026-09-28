#!/bin/bash
# Spherical registration + DK parcellation + thickness, without a dataset parcellation.
#
# For subjects with no dataset parcellation, pose cannot come from parcel centroids;
# sphere_rotsearch.py seats the sphere by searching SO(3) against fsaverage's folding, using
# only the subject's own curvature.
#
# The medial wall is fsaverage's, pulled back through the registration.
#
#   ./finalize_noparc.sh 8
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer}
source $FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
export SUBJECTS_DIR=${SUBJECTS_DIR:-${EXVIVO_FS:-$(cd "$(dirname "$0")/../.." && pwd)/surf/fs}}
N=$1; S=Subject$N; H=rh; D=$SUBJECTS_DIR/$S; T=${THREADS:-8}

echo "=== [$S] curv / sulc / inflate / sphere"
cd $D
mris_smooth -n 3 -nw -seed 1234 surf/$H.white surf/$H.smoothwm
mris_inflate surf/$H.smoothwm surf/$H.inflated                    # writes $H.sulc
mris_curvature -w -seed 1234 surf/$H.white && mv surf/$H.white.H surf/$H.curv
mris_sphere -seed 1234 -threads $T surf/$H.inflated surf/$H.sphere

echo "=== [$S] pose by rotation search against fsaverage (no parcellation)"
# A single rotation search can seat the sphere just outside mris_register's +-64 deg basin.
# Run several seeds, register each, and keep the one whose start-to-registered angle is
# smallest -- the seed that started nearest the basin centre.
cd "$HERE"
ATLAS=$FREESURFER_HOME/average/$H.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif
best=""; bestang=999
for seed in 0 1 2 3 4; do
  python3 sphere_rotsearch.py $N --seed $seed --out-suffix .s$seed
  mris_register -curv -threads $T $D/surf/$H.sphere.rotsearch.s$seed $ATLAS $D/surf/$H.sphere.reg.s$seed >/dev/null 2>&1 || continue
  ang=$(python3 -c "
import numpy as np, nibabel as nib
u=lambda X: X/np.maximum(np.linalg.norm(X,axis=-1,keepdims=True),1e-12)
p,_=nib.freesurfer.read_geometry('$D/surf/$H.sphere.rotsearch.s$seed')
r,_=nib.freesurfer.read_geometry('$D/surf/$H.sphere.reg.s$seed')
a,b=u(p-p.mean(0)),u(r-r.mean(0))
print(f'{np.median(np.degrees(np.arccos(np.clip((a*b).sum(1),-1,1)))):.2f}')")
  echo "   seed $seed: start->reg $ang deg"
  awk "BEGIN{exit !($ang < $bestang)}" && { bestang=$ang; best=$seed; }
done
echo "=== [$S] best seed $best (angle $bestang deg)"
cp $D/surf/$H.sphere.rotsearch.s$best $D/surf/$H.sphere.prealign
cp $D/surf/$H.sphere.reg.s$best $D/surf/$H.sphere.reg

echo "=== [$S] cortex label / medial wall from fsaverage"
cd "$HERE"
python3 fs_annot.py cortex $N

echo "=== [$S] FreeSurfer DK"
cd $D
mris_ca_label -l label/$H.cortex.label -seed 1234 $S $H surf/$H.sphere.reg \
    $FREESURFER_HOME/average/$H.curvature.buckner40.filled.desikan_killiany.2010-03-25.gcs \
    label/$H.aparc.raw.annot
cd "$HERE"
python3 fs_annot.py clean $N

echo "=== [$S] thickness"
cd $D
mris_thickness $S $H surf/$H.thickness
echo "=== [$S] DONE"

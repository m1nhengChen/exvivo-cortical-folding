#!/bin/bash
# Stage 2: tessellate the white solid and fix its topology.
#
#     prep_volumes.py  ->  mri/filled.mgz
#     make_surfaces.sh ->  surf/rh.orig      topologically correct (genus 0), on the *label*
#                                            boundary (noise and all)
#     refine_white.sh  ->  surf/rh.white     deformed onto the real image (brain.image.b50.mgz)
#     pial_fs.sh       ->  surf/rh.pial      deformed onto the sulci-opened label map
#     rebuild_one.sh   ->  everything else   curv, sulc, inflate, sphere, register, parcellate
#
# mris_place_surface only deforms, and deformation preserves topology, so rh.orig's genus 0
# carries through to rh.white.  Topology correction is done once here; downstream stages
# iterate without repeating it.
#
# Usage: ./make_surfaces.sh Subject1
set -e
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer}
source $FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
export SUBJECTS_DIR=${SUBJECTS_DIR:-${EXVIVO_FS:-$(cd "$(dirname "$0")/../.." && pwd)/surf/fs}}

HERE="$(cd "$(dirname "$0")" && pwd)"
S=$1
H=rh                      # right hemisphere
cd $SUBJECTS_DIR/$S
T=${THREADS:-$(n=$(nproc); echo $((n>8?8:n)))}

echo "### [$S] tessellate"
mri_pretess mri/filled.mgz 255 mri/brain.mgz mri/filled-pretess.mgz
mri_tessellate mri/filled-pretess.mgz 255 surf/$H.orig.nofix

# Decimate before topology correction.  mri_tessellate puts a vertex on every boundary voxel
# corner (~500k at 0.5 mm), oversampling past what the image resolves.  Collapsing to DECIMATE
# faces (quadric edge collapse, topology-preserving) roughly halves the vertex count, removes
# voxel-scale speckle, and shrinks the defect mris_fix_topology must correct.
if [ -n "$DECIMATE" ]; then
  mris_convert surf/$H.orig.nofix surf/$H.orig.nofix.full.vtk
  python3 "$HERE/decimate.py" surf/$H.orig.nofix.full.vtk surf/$H.orig.nofix.dec.vtk $DECIMATE
  mris_convert surf/$H.orig.nofix.dec.vtk surf/$H.orig.nofix
fi
mris_extract_main_component surf/$H.orig.nofix surf/$H.orig.nofix

echo "### [$S] topology correction"
mris_smooth -nw -seed 1234 surf/$H.orig.nofix surf/$H.smoothwm.nofix
mris_inflate -no-save-sulc surf/$H.smoothwm.nofix surf/$H.inflated.nofix
mris_sphere -q -p 6 -a 128 -seed 1234 surf/$H.inflated.nofix surf/$H.qsphere.nofix
mris_fix_topology -mgz -sphere qsphere.nofix -ga -seed 1234 -threads $T $S $H
mris_remove_intersection surf/$H.orig surf/$H.orig

echo "### [$S] -> surf/$H.orig   (next: ./refine_white.sh $S brain.image.b50.mgz b50)"
mris_euler_number surf/$H.orig 2>&1 | grep -iE "euler|genus|total defect" | head -2

#!/bin/bash
# One command, raw ex-vivo hemisphere MRI -> cortical surfaces + parcellation, as VTK.
#
#   ./run_pipeline.sh <input.nii.gz> [options]
#
# Self-contained: everything is created under a work directory (default <out>/work), and the
# pipeline scripts in ./pipeline are run in place.  Nothing outside this src/ tree is read or
# written except the external tools (FreeSurfer, ANTs, docker) and the output directory.
#
# Stages:
#   segmentation (FLASH GM/WM nnU-Net, docker) -> pseudo-T1 volumes -> tent mapping ->
#   tessellate+decimate+topology -> white on the image -> remesh -> RecoverSulci pial (docker)
#   -> spherical registration + DK parcellation + thickness (no dataset parcellation) -> VTK.
#
# Options:
#   --seg <mask.nii.gz>   Skip DL segmentation; use this GM/WM mask (GM=1, WM=2) directly.
#                         Use for T2w input, or any modality you already have a seg for.
#   --id <N>              Integer working slot (default 90), must be unused in the work dir.
#   --out <dir>          Output directory for the VTK (default: alongside the input).
#   --work <dir>         Work directory (default: <out>/work).
#   --threads <T>        Threads for FreeSurfer stages (default 8).
#   --keep-work          Keep the work directory after finishing.
#
# Requirements: FreeSurfer 7.x at $FREESURFER_HOME, ANTs (N4), and docker with images
#   pulks/docker_hippogang_exvivo_segm:v1.4.5 and ibeatgroup/ibeat_v2:release210.  --seg skips
#   the first; the pial stage always needs the second.
set -eo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
PIPE="$SRC/pipeline"
SEGDOCK=pulks/docker_hippogang_exvivo_segm:v1.4.5
export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer}

INPUT=""; SEGMASK=""; ID=90; OUTDIR=""; WORKDIR=""; THREADS=8; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seg) SEGMASK=$2; shift 2;;
    --id) ID=$2; shift 2;;
    --out) OUTDIR=$2; shift 2;;
    --work) WORKDIR=$2; shift 2;;
    --threads) THREADS=$2; shift 2;;
    --keep-work) KEEP=1; shift;;
    -*) echo "unknown option $1"; exit 2;;
    *) INPUT=$1; shift;;
  esac
done
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { echo "usage: run_pipeline.sh <input.nii.gz> [--seg mask] [--id N] [--out dir]"; exit 2; }
[ -z "$OUTDIR" ] && OUTDIR=$(dirname "$(readlink -f "$INPUT")")/surfaces_Subject$ID
[ -z "$WORKDIR" ] && WORKDIR=$OUTDIR/work
mkdir -p "$OUTDIR" "$WORKDIR"

# self-contained work layout; every pipeline script reads these
export EXVIVO_WORK="$WORKDIR"
export EXVIVO_FS="$WORKDIR/fs"
export EXVIVO_CSF="$WORKDIR/csf"
export EXVIVO_IB="$WORKDIR/ibeat"
export EXVIVO_DATA="$OUTDIR"                 # export_vtk writes straight to the output
export SUBJECTS_DIR="$EXVIVO_FS"
export SEG="$WORKDIR/seg05/{s}.nii.gz"       # {s} -> Subject{N}
export THREADS
mkdir -p "$EXVIVO_FS" "$EXVIVO_CSF" "$EXVIVO_IB" "$WORKDIR/seg05"

S=Subject$ID
D=$EXVIVO_FS/$S
N4=$EXVIVO_CSF/S$ID.t2w_n4.nii.gz
SEGOUT=$WORKDIR/seg05/$S.nii.gz
NNWORK=$WORKDIR/nnunet_$ID

[ -e "$D" ] && { echo "!! $D already exists -- pick a free --id"; exit 1; }
# fsaverage must be visible in SUBJECTS_DIR for rotation search and cortex-label transfer
[ -e "$EXVIVO_FS/fsaverage" ] || ln -s "$FREESURFER_HOME/subjects/fsaverage" "$EXVIVO_FS/fsaverage"
set +u +e; source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1 || true; set -u -e
echo "############ run_pipeline  $S  $(date +%F' '%T)  input=$INPUT"

# ---- 0. segmentation -> seg05/Subject{N}.nii.gz  +  csf/S{N}.t2w_n4.nii.gz ----------------
if [ -n "$SEGMASK" ]; then
  echo "== [0] using supplied segmentation, N4-correcting the input"
  cp "$SEGMASK" "$SEGOUT"
  python3 - "$INPUT" "$SEGOUT" "$N4" <<'PY'
import sys, os, subprocess, shutil, numpy as np, nibabel as nib
from scipy import ndimage as ndi
inp, seg, out = sys.argv[1:4]
im=nib.load(inp); sg=nib.load(seg)
a=np.asarray(im.dataobj).astype(np.float32)
ijk=np.indices(sg.shape).reshape(3,-1); xyz=nib.affines.apply_affine(sg.affine, ijk.T)
src=nib.affines.apply_affine(np.linalg.inv(im.affine), xyz).T
r=ndi.map_coordinates(a,src,order=1,mode="constant",cval=0).reshape(sg.shape)
raw=out.replace(".nii.gz","")+".raw.nii.gz"; msk=out.replace(".nii.gz","")+".n4mask.nii.gz"
nib.Nifti1Image(r, sg.affine).to_filename(raw)
lab=np.asarray(sg.dataobj); brain=ndi.binary_dilation(lab>0,iterations=3)
nib.Nifti1Image(brain.astype(np.uint8), sg.affine).to_filename(msk)
n4=next((c for c in ("/opt/ants/bin/N4BiasFieldCorrection","N4BiasFieldCorrection")
         if (os.path.exists(c) if os.path.isabs(c) else shutil.which(c))), None)
assert n4, "N4BiasFieldCorrection not found"
subprocess.run([n4,"-d","3","-i",raw,"-x",msk,"-o",out,"-s","2","-c","[100x100x100x100,1e-7]","-b","[120]"],check=True,capture_output=True)
print("   N4 done ->", out)
PY
else
  echo "== [0a] segmentation prep (crop + spoof spacing for nnU-Net)"
  rm -rf "$NNWORK"; mkdir -p "$NNWORK"
  python3 "$PIPE/seg_prep.py" "$INPUT" "$NNWORK"
  echo "== [0b] FLASH GM/WM nnU-Net  (docker $SEGDOCK)  $(date +%T)"
  docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --privileged --rm \
    -v "$NNWORK":/data/exvivo/ "$SEGDOCK" \
    /bin/bash -c "bash /src/commands_nnunet_inference.sh exvivo_flash_gm_wm_segm" >"$NNWORK/seg.log" 2>&1
  NN=$NNWORK/data_for_inference/output_from_nnunet_inference/case.nii.gz
  [ -f "$NN" ] || { echo "!! segmentation produced no output; see $NNWORK/seg.log"; tail -5 "$NNWORK/seg.log"; exit 1; }
  echo "== [0c] segmentation post (world placement + 0.5mm + N4 + relabel)  $(date +%T)"
  python3 "$PIPE/seg_post.py" "$NNWORK" "$NN" "$SEGOUT" "$N4"
fi

# ---- 1-3. volumes + tent mapping ----------------------------------------------------------
echo "== [1] pseudo-T1 volumes";   python3 "$PIPE/prep_volumes.py" $ID
echo "== [2] tissue mask";         python3 "$PIPE/make_tissue.py" $ID
echo "== [3] tent mapping (b50)";  python3 "$PIPE/wm_target.py" $ID --blend 0.5

# ---- 4. tessellate + adaptive decimate + topology -----------------------------------------
echo "== [4] tessellate + topology  $(date +%T)"
mri_pretess $D/mri/filled.mgz 255 $D/mri/brain.mgz $D/mri/filled-pretess.mgz >/dev/null 2>&1
mri_tessellate $D/mri/filled-pretess.mgz 255 $D/surf/rh.orig.nofix >/dev/null 2>&1
NF=$(mris_info $D/surf/rh.orig.nofix 2>/dev/null | grep -iE "faces" | grep -oE "[0-9]+" | head -1)
DEC=$(python3 -c "print(round($NF*0.325))")      # 1.0 mm edge = 0.325 x full faces
echo "   full faces $NF -> DECIMATE $DEC"
DECIMATE=$DEC "$PIPE/make_surfaces.sh" $S

# ---- 5-7. white + remesh + RecoverSulci pial ----------------------------------------------
echo "== [5] white on b50  $(date +%T)";  "$PIPE/refine_white.sh" $S brain.image.b50.mgz b50
echo "== [6] remesh white";               python3 "$PIPE/remesh_white.py" $ID --edge 0.5 --surf white.b50 --taubin 5 --out white.clean
cp $D/surf/rh.white.clean $D/surf/rh.white
echo "== [7] RecoverSulci + pial  (docker ibeat)  $(date +%T)"
python3 "$PIPE/pial_fs.py" $ID
"$PIPE/pial_fs.sh" $S
cp $D/surf/rh.pial.fs $D/surf/rh.pial

# ---- 8. sphere reg + DK parcellation + thickness (no dataset parcellation) -----------------
echo "== [8] registration + parcellation + thickness  $(date +%T)"
"$PIPE/finalize_noparc.sh" $ID

# ---- 9. export to VTK (written straight to the output dir via EXVIVO_DATA) -----------------
echo "== [9] export VTK"
python3 "$SRC/export_vtk.py" $ID
mv "$OUTDIR/$S"/rh.*.vtk "$OUTDIR"/ 2>/dev/null; rmdir "$OUTDIR/$S" 2>/dev/null || true

[ $KEEP -eq 0 ] && rm -rf "$WORKDIR"
echo "############ DONE  $(date +%F' '%T)  ->  $OUTDIR"
ls -1 "$OUTDIR"/*.vtk 2>/dev/null
